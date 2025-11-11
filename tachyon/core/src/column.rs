/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use gpu::ffi::column as gpu_column;

use crate::data_type::DataType;
pub trait Array: std::fmt::Debug + Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn data_type(&self) -> DataType;
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Debug)]
pub struct VecArray<T> {
    pub data: Vec<T>,
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
pub struct Column {
    pub name: String,
    pub data_type: DataType,
    pub values: Arc<dyn Array>,
    pub null_bits: Option<Vec<u64>>,
}

impl Column {
    pub fn new<T: Array + 'static>(
        name: &str, values: Arc<T>, validity_bits: Option<Vec<u64>>,
    ) -> Self {
        Self {
            name: name.to_string(),
            data_type: values.data_type(),
            values,
            null_bits: validity_bits,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn have_null(&self) -> bool {
        self.null_bits.is_some()
    }

    pub fn data_as_slice<T: 'static>(&self) -> Option<&[T]> {
        self.values.as_any().downcast_ref::<VecArray<T>>().map(|a| a.data.as_slice())
    }

    pub fn from_gpu_column(
        column: &gpu_column::Column, name: &str, data_type: DataType,
    ) -> Result<Self, Box<dyn Error>> {
        let col = match data_type {
            DataType::I8 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<i8>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::I16 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<i16>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::I32 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<i32>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::I64 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<i64>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::U8 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<u8>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::U16 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<u16>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::U32 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<u32>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::U64 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<u64>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::F32 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<f32>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::F64 => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<f64>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            DataType::BOOL => Self::new(
                name,
                Arc::new(VecArray { data: column.host_data::<bool>()?, datatype: data_type }),
                column.host_bitmap()?,
            ),
            _ => todo!(),
        };
        Ok(col)
    }

    pub fn to_gpu_column(&self) -> Result<gpu_column::Column, Box<dyn Error>> {
        match self.data_type {
            DataType::I8 => gpu_column::Column::new(
                self.data_as_slice::<i8>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::I16 => gpu_column::Column::new(
                self.data_as_slice::<i16>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::I32 => gpu_column::Column::new(
                self.data_as_slice::<i32>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::I64 => gpu_column::Column::new(
                self.data_as_slice::<i64>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::U8 => gpu_column::Column::new(
                self.data_as_slice::<u8>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::U16 => gpu_column::Column::new(
                self.data_as_slice::<u16>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::U32 => gpu_column::Column::new(
                self.data_as_slice::<u32>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::U64 => gpu_column::Column::new(
                self.data_as_slice::<u64>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::F32 => gpu_column::Column::new(
                self.data_as_slice::<f32>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::F64 => gpu_column::Column::new(
                self.data_as_slice::<f64>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            DataType::BOOL => gpu_column::Column::new(
                self.data_as_slice::<bool>().ok_or("Failed to cast")?,
                self.null_bits_as_slice(),
            ),
            _ => Err(format!("Unsupported data type: {:?}", self.data_type).into()),
        }
    }

    pub fn null_bits_as_slice(&self) -> Option<&[u64]> {
        self.null_bits.as_deref()
    }
}
