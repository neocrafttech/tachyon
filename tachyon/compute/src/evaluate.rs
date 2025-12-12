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
    let mut code_block = CodeBlock::default();
    expr.to_nvrtc::<B>(&schema_context, &mut code_block)?;

    let size = columns[0].len();
    let input_cols =
        columns.iter().map(|col| col.to_gpu_column()).collect::<Result<Vec<_>, _>>()?;

    let mut output_cols = Vec::<gpu_column::Column>::new();
    let result_type = expr.infer_type(&schema_context)?;

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
