/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::collections::HashMap;

use crate::codegen::{CodeBlock, CodeGen};
use crate::column::Column;
use crate::data_type::DataType;
use crate::expr::{Expr, SchemaContext};

pub fn evaluate(expr: &Expr, columns: Vec<Column>) {
    let column_map: HashMap<String, (u16, DataType)> = columns
        .into_iter()
        .enumerate()
        .map(|(idx, col)| (col.name, (idx as u16, col.data_type)))
        .collect();

    let schema_context = SchemaContext::new().with_columns(&column_map);
    let mut code_block = CodeBlock::default();
    let code = expr.to_nvrtc(&schema_context, &mut code_block).expect("Code to be generated");
    println!("Code is {:?} {:?}", code, code_block.code())
}
