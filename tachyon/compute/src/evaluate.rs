/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::collections::HashMap;
use std::error::Error;

use gpu::cuda_launcher;
use gpu::ffi::column as gpu_column;

use crate::bit_vector::BitBlock;
use crate::codegen::{CodeBlock, CodeGen};
use crate::column::Column;
use crate::data_type::DataType;
use crate::error::ErrorMode;
use crate::expr::{Expr, SchemaContext};
use crate::operator::Operator;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    GPU,
}

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
    let mut code_block = CodeBlock::default();
    expr.to_nvrtc::<B>(schema_context, &mut code_block)?;

    let size = columns[0].len();
    let input_cols =
        columns.iter().map(|col| col.to_gpu_column()).collect::<Result<Vec<_>, _>>()?;

    let mut output_cols = Vec::<gpu_column::Column>::new();
    let result_type = expr.infer_type(schema_context)?;

    let gpu_col = gpu_column::Column::new_uninitialized::<B>(
        size * result_type.native_size(),
        size.div_ceil(B::BITS),
        size,
    )?;
    output_cols.push(gpu_col);

    cuda_launcher::launch::<B>(code_block.code(), &input_cols, &output_cols).await?;

    let result_cols = output_cols
        .into_iter()
        .map(|col| -> Result<_, Box<dyn Error>> {
            Column::from_gpu_column(&col, "r0", result_type)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(result_cols)
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

    cuda_launcher::launch_aggregate::<B>(&code, &input_cols, &output_cols).await?;

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
