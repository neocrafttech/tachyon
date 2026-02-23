/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use gpu::column as gpu_column;
use half::{bf16, f16};

use crate::bit_vector::{BitBlock, BitVector};
use crate::data_type::DataType;

/// Type-erased column storage interface.
pub trait Array: std::fmt::Debug + Send + Sync {
    /// Number of values in the array.
    fn len(&self) -> usize;
    /// Returns `true` when the array has no values.
    fn is_empty(&self) -> bool;
    /// Logical type of the underlying values.
    fn data_type(&self) -> DataType;
    /// Returns a `dyn Any` reference for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Debug)]
/// Generic vector-backed implementation of [`Array`].
pub struct VecArray<T> {
    /// Typed value buffer.
    pub data: Vec<T>,
    /// Logical data type for `data`.
    pub datatype: DataType,
}

impl<T: 'static + Send + Sync + Debug> Array for VecArray<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    fn data_type(&self) -> DataType {
        self.datatype
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone)]
/// A named column with values and optional null bitmap.
pub struct Column<B: BitBlock> {
    /// Column name.
    pub name: String,
    /// Logical value type.
    pub data_type: DataType,
    /// Type-erased values container.
    pub values: Arc<dyn Array>,
    /// Null bitmap where `1` indicates valid and `0` indicates null.
    pub null_bits: Option<BitVector<B>>,
}

macro_rules! from_gpu_column {
    ($name:expr, $type:ty, $data_type:expr, $column:expr) => {
        Self::new(
            $name,
            Arc::new(VecArray { data: $column.host_data::<$type>()?, datatype: $data_type }),
            $column.host_bitmap()?.map(|bitmap| BitVector::new(bitmap, $column.len())),
        )
    };
}

macro_rules! to_gpu_column {
    ($self:expr, $type:ty) => {
        gpu_column::Column::new(
            $self.data_as_slice::<$type>().ok_or("Failed to cast")?,
            $self.null_bits.as_ref().map(|bits| bits.as_slice()),
        )
    };
}

impl<B: BitBlock> Column<B> {
    /// Creates a new column from an [`Array`] implementation and optional null bitmap.
    pub fn new<T: Array + 'static>(
        name: &str, values: Arc<T>, null_bits: Option<BitVector<B>>,
    ) -> Self {
        Self { name: name.to_string(), data_type: values.data_type(), values, null_bits }
    }

    /// Number of rows in the column.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when this column has no rows.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns `true` when this column carries a null bitmap.
    pub fn have_null(&self) -> bool {
        self.null_bits.is_some()
    }

    /// Attempts to view the underlying values as a typed slice.
    pub fn data_as_slice<T: 'static>(&self) -> Option<&[T]> {
        self.values.as_any().downcast_ref::<VecArray<T>>().map(|a| a.data.as_slice())
    }

    /// Converts a GPU column into a host [`Column`].
    pub fn from_gpu_column(
        column: &gpu_column::Column, name: &str, data_type: DataType,
    ) -> Result<Self, Box<dyn Error>> {
        let col = match data_type {
            DataType::I8 => from_gpu_column!(name, i8, data_type, column),
            DataType::I16 => from_gpu_column!(name, i16, data_type, column),
            DataType::I32 => from_gpu_column!(name, i32, data_type, column),
            DataType::I64 => from_gpu_column!(name, i64, data_type, column),
            DataType::U8 => from_gpu_column!(name, u8, data_type, column),
            DataType::U16 => from_gpu_column!(name, u16, data_type, column),
            DataType::U32 => from_gpu_column!(name, u32, data_type, column),
            DataType::U64 => from_gpu_column!(name, u64, data_type, column),
            DataType::BF16 => from_gpu_column!(name, bf16, data_type, column),
            DataType::F16 => from_gpu_column!(name, f16, data_type, column),
            DataType::F32 => from_gpu_column!(name, f32, data_type, column),
            DataType::F64 => from_gpu_column!(name, f64, data_type, column),
            DataType::Bool => from_gpu_column!(name, bool, data_type, column),
            _ => todo!(),
        };
        Ok(col)
    }

    /// Converts this host column into its GPU representation.
    pub fn to_gpu_column(&self) -> Result<gpu_column::Column, Box<dyn Error>> {
        match self.data_type {
            DataType::I8 => to_gpu_column!(self, i8),
            DataType::I16 => to_gpu_column!(self, i16),
            DataType::I32 => to_gpu_column!(self, i32),
            DataType::I64 => to_gpu_column!(self, i64),
            DataType::U8 => to_gpu_column!(self, u8),
            DataType::U16 => to_gpu_column!(self, u16),
            DataType::U32 => to_gpu_column!(self, u32),
            DataType::U64 => to_gpu_column!(self, u64),
            DataType::BF16 => to_gpu_column!(self, bf16),
            DataType::F16 => to_gpu_column!(self, f16),
            DataType::F32 => to_gpu_column!(self, f32),
            DataType::F64 => to_gpu_column!(self, f64),
            DataType::Bool => to_gpu_column!(self, bool),
            _ => Err(format!("Unsupported data type: {:?}", self.data_type).into()),
        }
    }

    /// Returns the null bitmap if one is present.
    pub fn null_bits_as_slice(&self) -> Option<&BitVector<B>> {
        self.null_bits.as_ref()
    }
}
