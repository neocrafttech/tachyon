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
    /// Backing buffer for encoded UTF-8 bytes when `data_type` is [`DataType::Str`].
    pub string_buffer: Option<Arc<Vec<u8>>>,
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
        Self {
            name: name.to_string(),
            data_type: values.data_type(),
            values,
            string_buffer: None,
            null_bits,
        }
    }

    /// Creates a string column from pre-encoded string views and string buffer.
    pub fn new_string(
        name: &str, views: Arc<VecArray<gpu_column::StringView>>, string_buffer: Vec<u8>,
        null_bits: Option<BitVector<B>>,
    ) -> Self {
        Self {
            name: name.to_string(),
            data_type: DataType::Str,
            values: views,
            string_buffer: Some(Arc::new(string_buffer)),
            null_bits,
        }
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
            DataType::Str => {
                let views = column.host_data::<gpu_column::StringView>()?;
                let buffer = column.host_string_buffer()?.unwrap_or_default();
                Self::new_string(
                    name,
                    Arc::new(VecArray { data: views, datatype: data_type }),
                    buffer,
                    column.host_bitmap()?.map(|bitmap| BitVector::new(bitmap, column.len())),
                )
            }
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
            DataType::Str => {
                let views = self
                    .data_as_slice::<gpu_column::StringView>()
                    .ok_or("Failed to cast to StringView")?;
                let buffer = self
                    .string_buffer
                    .as_ref()
                    .ok_or("String column requires a separate string buffer")?;
                gpu_column::Column::new_string(
                    views,
                    buffer.as_slice(),
                    self.null_bits.as_ref().map(|bits| bits.as_slice()),
                )
            }
        }
    }

    /// Returns the null bitmap if one is present.
    pub fn null_bits_as_slice(&self) -> Option<&BitVector<B>> {
        self.null_bits.as_ref()
    }

    /// Returns encoded UTF-8 buffer for string columns.
    pub fn string_buffer_as_slice(&self) -> Option<&[u8]> {
        self.string_buffer.as_ref().map(|b| b.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use gpu::column::{
        STRING_INLINE_DATA_BYTES, STRING_INLINE_PREFIX_BYTES, STRING_INLINE_TOTAL_BYTES,
    };

    use super::*;

    fn pack_u32_prefix(bytes: &[u8]) -> u32 {
        let mut prefix = 0u32;
        let take = bytes.len().min(STRING_INLINE_PREFIX_BYTES);
        for (i, b) in bytes.iter().take(take).enumerate() {
            prefix |= (*b as u32) << (i * 8);
        }
        prefix
    }

    fn encode_string_view(
        s: &str, buffer: &mut Vec<u8>,
    ) -> Result<gpu_column::StringView, Box<dyn Error>> {
        let bytes = s.as_bytes();
        let size = u32::try_from(bytes.len()).map_err(|_| "String length exceeds u32")?;
        let prefix = pack_u32_prefix(bytes);
        if bytes.len() <= STRING_INLINE_TOTAL_BYTES {
            let mut payload = 0u64;
            for (i, b) in bytes
                .iter()
                .enumerate()
                .skip(STRING_INLINE_PREFIX_BYTES)
                .take(STRING_INLINE_DATA_BYTES)
            {
                payload |= (*b as u64) << ((i - STRING_INLINE_PREFIX_BYTES) * 8);
            }
            Ok(gpu_column::StringView { size, prefix, data: payload })
        } else {
            let offset =
                u64::try_from(buffer.len()).map_err(|_| "String buffer offset overflow")?;
            buffer.extend_from_slice(bytes);
            Ok(gpu_column::StringView { size, prefix, data: offset })
        }
    }

    fn decode_string_view(
        view: gpu_column::StringView, buffer: &[u8],
    ) -> Result<String, Box<dyn Error>> {
        let len = view.size as usize;
        if len <= STRING_INLINE_TOTAL_BYTES {
            let mut tmp = [0u8; STRING_INLINE_TOTAL_BYTES];
            for (i, out) in tmp.iter_mut().enumerate().take(STRING_INLINE_PREFIX_BYTES) {
                *out = ((view.prefix >> (i * 8)) & 0xFF) as u8;
            }
            for i in 0..STRING_INLINE_DATA_BYTES {
                tmp[STRING_INLINE_PREFIX_BYTES + i] = ((view.data >> (i * 8)) & 0xFF) as u8;
            }
            return Ok(std::str::from_utf8(&tmp[..len])?.to_string());
        }

        let offset = view.data as usize;
        let end = offset.checked_add(len).ok_or("StringView offset overflow while decoding")?;
        if end > buffer.len() {
            return Err("StringView points outside string buffer".into());
        }
        Ok(std::str::from_utf8(&buffer[offset..end])?.to_string())
    }

    #[test]
    fn test_string_view_inline_roundtrip() {
        let mut buf = Vec::new();
        let view = encode_string_view("hello", &mut buf).expect("encode");
        assert!(buf.is_empty());
        let out = decode_string_view(view, &buf).expect("decode");
        assert_eq!(out, "hello");
    }

    #[test]
    fn test_string_view_external_roundtrip() {
        let mut buf = Vec::new();
        let text = "this string is longer than twelve bytes";
        let view = encode_string_view(text, &mut buf).expect("encode");
        assert!(!buf.is_empty());
        let out = decode_string_view(view, &buf).expect("decode");
        assert_eq!(out, text);
    }

    #[test]
    fn test_string_view_inline_roundtrip_german_utf8() {
        let mut buf = Vec::new();
        let text = "straße";
        assert!(text.len() <= STRING_INLINE_TOTAL_BYTES);
        let view = encode_string_view(text, &mut buf).expect("encode");
        assert!(buf.is_empty());
        let out = decode_string_view(view, &buf).expect("decode");
        assert_eq!(out, text);
    }

    #[test]
    fn test_string_view_external_roundtrip_hindi_utf8() {
        let mut buf = Vec::new();
        let text = "नमस्ते दुनिया";
        assert!(text.len() > STRING_INLINE_TOTAL_BYTES);
        let view = encode_string_view(text, &mut buf).expect("encode");
        assert!(!buf.is_empty());
        let out = decode_string_view(view, &buf).expect("decode");
        assert_eq!(out, text);
    }

    #[test]
    fn test_string_view_roundtrip_mixed_languages() {
        let mut buf = Vec::new();
        let texts = ["Grüße", "नमस्ते", "München में स्वागत"];

        for text in texts {
            let view = encode_string_view(text, &mut buf).expect("encode");
            let out = decode_string_view(view, &buf).expect("decode");
            assert_eq!(out, text);
        }
    }
}
