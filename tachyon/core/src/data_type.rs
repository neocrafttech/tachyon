/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bool,
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
    Str,
}

impl DataType {
    pub fn native_size(&self) -> usize {
        match self {
            DataType::Bool => std::mem::size_of::<bool>(),
            DataType::I8 => std::mem::size_of::<i8>(),
            DataType::I16 => std::mem::size_of::<i16>(),
            DataType::I32 => std::mem::size_of::<i32>(),
            DataType::I64 => std::mem::size_of::<i64>(),
            DataType::U8 => std::mem::size_of::<u8>(),
            DataType::U16 => std::mem::size_of::<u16>(),
            DataType::U32 => std::mem::size_of::<u32>(),
            DataType::U64 => std::mem::size_of::<u64>(),
            DataType::BF16 => std::mem::size_of::<bf16>(),
            DataType::F16 => std::mem::size_of::<f16>(),
            DataType::F32 => std::mem::size_of::<f32>(),
            DataType::F64 => std::mem::size_of::<f64>(),
            DataType::Str => std::mem::size_of::<u8>(),
        }
    }

    pub fn c_type(&self) -> &'static str {
        match self {
            DataType::Bool => "bool",
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
            DataType::Str => "uint8_t",
        }
    }

    pub fn kernel_type(&self) -> &'static str {
        match self {
            DataType::Bool => "Bool",
            DataType::I8 => "Int8",
            DataType::I16 => "Int16",
            DataType::I32 => "Int32",
            DataType::I64 => "Int64",
            DataType::U8 => "UInt8",
            DataType::U16 => "UInt16",
            DataType::U32 => "UInt32",
            DataType::U64 => "UInt64",
            DataType::BF16 => "BFloat16",
            DataType::F16 => "Float16",
            DataType::F32 => "Float32",
            DataType::F64 => "Float64",
            DataType::Str => "String",
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
        matches!(self, DataType::Str)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, DataType::Bool)
    }
}
