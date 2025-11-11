/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

/// Primitive data types we support (extend as needed)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    BF16,
    F16,
    F32,
    F64,
    Bool,
    Utf8,
}

impl DataType {
    pub fn c_type(&self) -> &'static str {
        match self {
            DataType::I8 => "int8_t",
            DataType::I16 => "int16_t",
            DataType::I32 => "int32_t",
            DataType::I64 => "int64_t",
            DataType::U8 => "uint8_t",
            DataType::U16 => "uint16_t",
            DataType::U32 => "uint32_t",
            DataType::U64 => "uint64_t",
            DataType::BF16 => "bfloat16",
            DataType::F16 => "float16",
            DataType::F32 => "float",
            DataType::F64 => "double",
            DataType::Bool => "bool",
            DataType::Utf8 => "uint8_t",
        }
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64)
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(self, DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64)
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64)
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
        )
    }

    pub fn is_string(&self) -> bool {
        matches!(self, DataType::Utf8)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, DataType::Bool)
    }
}
