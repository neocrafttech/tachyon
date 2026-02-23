//! Compute engine APIs for expression parsing, planning, and evaluation.
//!
//! This crate exposes the core DataFrame compute primitives used by Tachyon,
//! including expression ASTs, column containers, parser helpers, and GPU-backed
//! expression evaluation.

/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

/// Bitset utilities used for null tracking in columns.
pub mod bit_vector;
/// Code generation interfaces used to produce execution kernels.
mod codegen;
/// Columnar in-memory data containers.
pub mod column;
/// Logical and physical data type definitions.
pub mod data_type;
/// Error handling modes and math/runtime errors.
pub mod error;
/// Expression evaluation entry points.
pub mod evaluate;
/// Expression AST and type inference utilities.
pub mod expr;
/// Operators used by expressions.
pub mod operator;
/// Parser for scheme-style expression syntax.
pub mod parser;
