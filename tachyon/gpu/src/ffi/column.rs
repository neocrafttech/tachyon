/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::error::Error;

use crate::column::{BitmapElement, GpuElement};
use crate::ffi::memory::gpu_memory::{GpuMemory, MemoryType};

pub use crate::column::StringView;

pub struct Column {
    data_memory: GpuMemory,
    validity_memory: Option<GpuMemory>,
    string_buffer_memory: Option<GpuMemory>,
    pub num_rows: usize,
}

impl Column {
    /// Uploads supported values and an optional unsigned validity bitmap.
    ///
    /// ```compile_fail
    /// use gpu::column::Column;
    /// let _ = Column::new(&[String::from("owned")], None::<&[u64]>);
    /// ```
    pub fn new<T, B>(data: &[T], null_bits: Option<&[B]>) -> Result<Self, Box<dyn Error>>
    where
        T: GpuElement,
        B: BitmapElement,
    {
        let memory_type = MemoryType::Device;
        let data_memory = memory_type
            .allocate_from_slice(data)
            .map_err(|e| format!("Failed to allocate device memory for data: {}", e))?;

        let validity_memory = if let Some(null_bits) = null_bits {
            let device_bitmap = memory_type.allocate_from_slice(null_bits).map_err(|e| {
                format!("Failed to allocate device memory for validity bitmap: {}", e)
            })?;
            Some(device_bitmap)
        } else {
            None
        };

        Ok(Column {
            data_memory,
            validity_memory,
            string_buffer_memory: None,
            num_rows: data.len(),
        })
    }

    pub fn new_string<B>(
        views: &[StringView], string_buffer: &[u8], null_bits: Option<&[B]>,
    ) -> Result<Self, Box<dyn Error>>
    where
        B: BitmapElement,
    {
        let memory_type = MemoryType::Device;
        let data_memory = memory_type
            .allocate_from_slice(views)
            .map_err(|e| format!("Failed to allocate device memory for string views: {}", e))?;
        let string_buffer_memory = if string_buffer.is_empty() {
            None
        } else {
            Some(memory_type.allocate_from_slice(string_buffer).map_err(|e| {
                format!("Failed to allocate device memory for string buffer: {}", e)
            })?)
        };

        let validity_memory = if let Some(null_bits) = null_bits {
            let device_bitmap = memory_type.allocate_from_slice(null_bits).map_err(|e| {
                format!("Failed to allocate device memory for validity bitmap: {}", e)
            })?;
            Some(device_bitmap)
        } else {
            None
        };

        Ok(Column { data_memory, validity_memory, string_buffer_memory, num_rows: views.len() })
    }

    /// Allocates output storage, zeroing values and validity bits so unwritten nulls are safe.
    pub fn new_uninitialized<B: BitmapElement>(
        data_len: usize, null_bits_len: usize, num_rows: usize,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(data_len > 0, "Cannot allocate zero-sized memory block.");
        let memory_type = MemoryType::Device;

        let data_memory = memory_type
            .allocate(data_len)
            .map_err(|e| format!("Failed to allocate device memory for data: {}", e))?;

        let validity_memory = if null_bits_len > 0 {
            let validity_memory =
                memory_type.allocate(null_bits_len * std::mem::size_of::<B>()).map_err(|e| {
                    format!("Failed to allocate device memory for validity bitmap: {}", e)
                })?;
            Some(validity_memory)
        } else {
            None
        };

        Ok(Column { data_memory, validity_memory, string_buffer_memory: None, num_rows })
    }

    /// Allocates zeroed string output storage, including descriptors and validity bits.
    pub fn new_uninitialized_string<B: BitmapElement>(
        num_rows: usize, string_buffer_size: usize, null_bits_len: usize,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(num_rows > 0, "Cannot allocate zero-row string column.");
        let memory_type = MemoryType::Device;
        let data_memory = memory_type
            .allocate(num_rows * std::mem::size_of::<StringView>())
            .map_err(|e| format!("Failed to allocate device memory for string views: {}", e))?;

        let string_buffer_memory = if string_buffer_size > 0 {
            Some(memory_type.allocate(string_buffer_size).map_err(|e| {
                format!("Failed to allocate device memory for string buffer: {}", e)
            })?)
        } else {
            None
        };

        let validity_memory = if null_bits_len > 0 {
            let validity_memory =
                memory_type.allocate(null_bits_len * std::mem::size_of::<B>()).map_err(|e| {
                    format!("Failed to allocate device memory for validity bitmap: {}", e)
                })?;
            Some(validity_memory)
        } else {
            None
        };

        Ok(Column { data_memory, validity_memory, string_buffer_memory, num_rows })
    }

    pub fn len(&self) -> usize {
        self.num_rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn as_ffi_column<B: Sized>(&self) -> ColumnFFI<B> {
        let data_ptr = self.data_memory.device_ptr();

        let validity_ptr =
            self.validity_memory.as_ref().map_or(std::ptr::null(), |vm| vm.device_ptr());
        let string_buffer_ptr =
            self.string_buffer_memory.as_ref().map_or(std::ptr::null_mut(), |vm| vm.device_ptr());
        let string_buffer_size = self.string_buffer_memory.as_ref().map_or(0, |vm| vm.len());

        ColumnFFI {
            data: data_ptr as *const std::os::raw::c_void,
            null_bits: validity_ptr as *const B,
            size: self.num_rows,
            string_buffer: string_buffer_ptr as *const u8,
            string_buffer_size,
        }
    }

    /// Downloads supported values, rejecting invalid representations (including bool bytes).
    /// The requested type may reinterpret the buffer, but every value is checked before use.
    ///
    /// ```compile_fail
    /// use gpu::column::Column;
    /// fn download_owned(column: &Column) {
    ///     let _ = column.host_data::<String>();
    /// }
    /// ```
    pub fn host_data<T: GpuElement>(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.data_memory
            .to_vec::<T>()
            .map_err(|e| format!("Failed to copy data from device: {}", e).into())
    }

    pub fn host_bitmap<B: BitmapElement>(&self) -> Result<Option<Vec<B>>, Box<dyn Error>> {
        self.validity_memory
            .as_ref()
            .map(|vm| {
                vm.to_vec::<B>()
                    .map_err(|e| format!("Failed to copy bit map from device: {}", e).into())
            })
            .transpose()
    }

    pub fn host_string_buffer(&self) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
        self.string_buffer_memory
            .as_ref()
            .map(|vm| {
                vm.to_vec::<u8>()
                    .map_err(|e| format!("Failed to copy string buffer from device: {}", e).into())
            })
            .transpose()
    }
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct ColumnFFI<B: Sized> {
    pub data: *const std::os::raw::c_void,
    pub null_bits: *const B,
    pub size: usize,
    pub string_buffer: *const u8,
    pub string_buffer_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_boolean_download() {
        let column = Column::new(&[2u8], None::<&[u64]>).unwrap();
        assert!(column.host_data::<bool>().is_err());
        assert_eq!(column.host_data::<u8>().unwrap(), [2]);
        assert!(column.host_data::<u16>().is_err());
    }

    #[test]
    fn unwritten_output_has_valid_zero_values() {
        let column = Column::new_uninitialized::<u64>(3, 1, 3).unwrap();
        assert_eq!(column.host_data::<bool>().unwrap(), [false; 3]);
        assert_eq!(column.host_bitmap::<u64>().unwrap().unwrap(), [0]);

        let strings = Column::new_uninitialized_string::<u32>(2, 32, 1).unwrap();
        assert_eq!(strings.host_data::<StringView>().unwrap(), [StringView::default(); 2]);
        assert_eq!(strings.host_string_buffer().unwrap().unwrap(), [0; 32]);
        assert_eq!(strings.host_bitmap::<u32>().unwrap().unwrap(), [0]);
    }

    #[test]
    fn boolean_and_bitmap_roundtrip() {
        let column = Column::new(&[false, true, false], Some(&[0b110u8][..])).unwrap();
        assert_eq!(column.host_data::<bool>().unwrap(), [false, true, false]);
        assert_eq!(column.host_bitmap::<u8>().unwrap().unwrap(), [0b110]);
    }
}
