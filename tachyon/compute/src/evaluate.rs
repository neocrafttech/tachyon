/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::collections::HashMap;
use std::error::Error;

use gpu::column as gpu_column;

use crate::bit_vector::BitBlock;
use crate::codegen::{CodeBlock, CodeGen};
use crate::column::Column;
use crate::data_type::DataType;
use crate::error::ErrorMode;
use crate::expr::{Expr, SchemaContext};
use crate::operator::Operator;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Execution target for expression evaluation.
pub enum Device {
    /// Evaluate expressions on the GPU backend.
    GPU,
}

/// Evaluates an expression against a set of input columns.
///
/// Returns a vector of output columns. Row-wise expressions produce one output
/// column with one value per input row. Aggregate expressions produce a single
/// row output.
pub async fn evaluate<B: BitBlock>(
    device: Device, error_mode: ErrorMode, expr: &Expr, columns: &[Column<B>],
) -> Result<Vec<Column<B>>, Box<dyn Error>> {
    match device {
        Device::GPU => evaluate_gpu(expr, error_mode, columns).await,
    }
}

async fn evaluate_gpu<B: BitBlock>(
    expr: &Expr, error_mode: ErrorMode, columns: &[Column<B>],
) -> Result<Vec<Column<B>>, Box<dyn Error>> {
    let column_map: HashMap<String, (u16, DataType)> = columns
        .iter()
        .enumerate()
        .map(|(idx, col)| (col.name.clone(), (idx as u16, col.data_type)))
        .collect();

    let schema_context = SchemaContext::new().with_columns(&column_map).with_error_mode(error_mode);
    match expr {
        Expr::Aggregate { op, arg, distinct } => {
            evaluate_gpu_aggregate::<B>(*op, arg.as_ref(), *distinct, &schema_context, columns)
                .await
        }
        _ => evaluate_gpu_row::<B>(expr, &schema_context, columns).await,
    }
}

async fn evaluate_gpu_row<B: BitBlock>(
    expr: &Expr, schema_context: &SchemaContext, columns: &[Column<B>],
) -> Result<Vec<Column<B>>, Box<dyn Error>> {
    if let Expr::Column(name) = expr {
        if let Some((idx, DataType::Str)) = schema_context.lookup(name).copied() {
            return Ok(vec![columns[idx as usize].clone()]);
        }
    }

    let mut code_block = CodeBlock::default();
    expr.to_nvrtc::<B>(schema_context, &mut code_block)?;

    let size = columns[0].len();
    let input_cols =
        columns.iter().map(|col| col.to_gpu_column()).collect::<Result<Vec<_>, _>>()?;

    let mut output_cols = Vec::<gpu_column::Column>::new();
    let result_type = expr.infer_type(schema_context)?;

    let gpu_col = if result_type == DataType::Str {
        let row_capacity = estimate_string_row_capacity(expr, schema_context, columns)?;
        gpu_column::Column::new_uninitialized_string::<B>(
            size,
            row_capacity * size,
            size.div_ceil(B::BITS),
        )?
    } else {
        gpu_column::Column::new_uninitialized::<B>(
            size * result_type.native_size(),
            size.div_ceil(B::BITS),
            size,
        )?
    };
    output_cols.push(gpu_col);

    gpu::launch::<B>(code_block.code(), &input_cols, &output_cols).await?;

    let result_cols = output_cols
        .into_iter()
        .map(|col| -> Result<_, Box<dyn Error>> {
            Column::from_gpu_column(&col, "r0", result_type)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(result_cols)
}

fn max_string_len_column<B: BitBlock>(col: &Column<B>) -> Result<usize, Box<dyn Error>> {
    let values = col
        .data_as_slice::<gpu_column::StringView>()
        .ok_or("String expression requires encoded StringView columns")?;
    Ok(values.iter().map(|sv| sv.size as usize).max().unwrap_or(0))
}

fn estimate_string_row_capacity<B: BitBlock>(
    expr: &Expr, schema_context: &SchemaContext, columns: &[Column<B>],
) -> Result<usize, Box<dyn Error>> {
    match expr {
        Expr::Column(name) => {
            let (idx, dt) = schema_context
                .lookup(name)
                .copied()
                .ok_or_else(|| format!("unknown column: {}", name))?;
            if dt != DataType::Str {
                return Err("Expected string column".into());
            }
            Ok(max_string_len_column(&columns[idx as usize])?)
        }
        Expr::Call { name, args } => match name.as_str() {
            "lower" | "lower_case" | "upper" | "upper_case" => {
                if args.len() != 1 {
                    return Err(format!("{} expects 1 argument", name).into());
                }
                estimate_string_row_capacity(&args[0], schema_context, columns)
            }
            "substring" => {
                if args.len() != 3 {
                    return Err("substring expects 3 arguments".into());
                }
                let base_cap = estimate_string_row_capacity(&args[0], schema_context, columns)?;
                let requested = match &args[2] {
                    Expr::Literal(crate::expr::Literal::I32(v)) => (*v).max(0) as usize,
                    Expr::Literal(crate::expr::Literal::I64(v)) => (*v).max(0) as usize,
                    _ => base_cap,
                };
                Ok(base_cap.min(requested))
            }
            "concat" => {
                if args.len() != 2 {
                    return Err("concat expects 2 arguments".into());
                }
                let left = estimate_string_row_capacity(&args[0], schema_context, columns)?;
                let right = estimate_string_row_capacity(&args[1], schema_context, columns)?;
                Ok(left + right)
            }
            _ => Err(format!("Unsupported string function for output sizing: {}", name).into()),
        },
        _ => Err("Unable to infer string output buffer size for this expression".into()),
    }
}

async fn evaluate_gpu_aggregate<B: BitBlock>(
    op: Operator, arg: &Expr, distinct: bool, schema_context: &SchemaContext, columns: &[Column<B>],
) -> Result<Vec<Column<B>>, Box<dyn Error>> {
    if distinct {
        return Err("DISTINCT aggregates are not supported yet".into());
    }

    let (col_idx, col_type) = match arg {
        Expr::Column(col_name) => schema_context
            .lookup(col_name)
            .copied()
            .ok_or_else(|| format!("unknown column: {}", col_name))?,
        _ => return Err("Aggregate argument must be a column reference".into()),
    };

    let result_type =
        Expr::Aggregate { op, arg: Box::new(arg.clone()), distinct }.infer_type(schema_context)?;
    let code = build_aggregate_nvrtc_code::<B>(
        op,
        col_idx,
        col_type,
        result_type,
        schema_context.error_mode() == ErrorMode::Ansi,
    )?;
    let input_cols =
        columns.iter().map(|col| col.to_gpu_column()).collect::<Result<Vec<_>, _>>()?;

    let mut output_cols = Vec::<gpu_column::Column>::new();
    let size = 1usize;
    let gpu_col = gpu_column::Column::new_uninitialized::<B>(
        size * result_type.native_size(),
        size.div_ceil(B::BITS),
        size,
    )?;
    output_cols.push(gpu_col);

    gpu::launch_aggregate::<B>(&code, &input_cols, &output_cols).await?;

    let result_cols = output_cols
        .into_iter()
        .map(|col| -> Result<_, Box<dyn Error>> {
            Column::from_gpu_column(&col, "r0", result_type)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(result_cols)
}

fn build_aggregate_nvrtc_code<B: BitBlock>(
    op: Operator, col_idx: u16, col_type: DataType, result_type: DataType, ansi_error_mode: bool,
) -> Result<String, Box<dyn Error>> {
    let input_kernel_type = col_type.kernel_type();
    let output_kernel_type = result_type.kernel_type();
    let bits_type = B::C_TYPE;

    let code = match op {
        Operator::Min => format!(
            "\taggregate_codegen::min<TypeKind::{input_kernel_type}, TypeKind::{output_kernel_type}, {bits_type}>(ctx, input, output, num_rows, {col_idx});\n"
        ),
        Operator::Max => format!(
            "\taggregate_codegen::max<TypeKind::{input_kernel_type}, TypeKind::{output_kernel_type}, {bits_type}>(ctx, input, output, num_rows, {col_idx});\n"
        ),
        Operator::Sum => format!(
            "\taggregate_codegen::sum<{ansi_error_mode}, TypeKind::{input_kernel_type}, TypeKind::{output_kernel_type}, {bits_type}>(ctx, input, output, num_rows, {col_idx});\n"
        ),
        Operator::Avg => format!(
            "\taggregate_codegen::avg<{ansi_error_mode}, TypeKind::{input_kernel_type}, TypeKind::{output_kernel_type}, {bits_type}>(ctx, input, output, num_rows, {col_idx});\n"
        ),
        Operator::Count => format!(
            "\taggregate_codegen::count<TypeKind::{input_kernel_type}, {bits_type}>(ctx, input, output, num_rows, {col_idx});\n"
        ),
        _ => return Err(format!("Unsupported aggregate operator: {:?}", op).into()),
    };

    Ok(code)
}
