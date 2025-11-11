/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    BOOL,
    STR,
}

impl DataType {
    pub fn native_size(&self) -> usize {
        match self {
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
            DataType::BOOL => std::mem::size_of::<bool>(),
            DataType::STR => std::mem::size_of::<u8>(),
        }
    }

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
            DataType::BOOL => "bool",
            DataType::STR => "uint8_t",
        }
    }

    pub fn kernel_type(&self) -> &'static str {
        match self {
            DataType::I8 => "INT8",
            DataType::I16 => "INT16",
            DataType::I32 => "INT32",
            DataType::I64 => "INT64",
            DataType::U8 => "UINT8",
            DataType::U16 => "UINT16",
            DataType::U32 => "UINT32",
            DataType::U64 => "UINT64",
            DataType::BF16 => "BFLOAT16",
            DataType::F16 => "FLOAT16",
            DataType::F32 => "FLOAT32",
            DataType::F64 => "FLOAT64",
            DataType::BOOL => "BOOL",
            DataType::STR => "STRING",
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
        matches!(self, DataType::STR)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, DataType::BOOL)
    }
}
