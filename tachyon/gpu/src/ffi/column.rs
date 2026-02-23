/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::error::Error;

use crate::ffi::memory::gpu_memory::{GpuMemory, MemoryType};

/// Compact string descriptor stored in the main column stream.
///
/// Memory layout (16 bytes total):
/// - 4 bytes: string byte length
/// - 4 bytes: prefix (first 4 bytes)
/// - 8 bytes: inline continuation bytes or offset into string buffer stream
pub const STRING_INLINE_PREFIX_BYTES: usize = 4;
pub const STRING_INLINE_DATA_BYTES: usize = 8;
pub const STRING_INLINE_TOTAL_BYTES: usize = STRING_INLINE_PREFIX_BYTES + STRING_INLINE_DATA_BYTES;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StringView {
    pub size: u32,
    pub prefix: u32,
    pub data: u64,
}

pub struct Column {
    data_memory: GpuMemory,
    validity_memory: Option<GpuMemory>,
    string_buffer_memory: Option<GpuMemory>,
    pub num_rows: usize,
}

impl Column {
    pub fn new<T, B>(data: &[T], null_bits: Option<&[B]>) -> Result<Self, Box<dyn Error>>
    where
        T: Sized,
        B: Sized,
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
        B: Sized,
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

    pub fn new_uninitialized<B: Sized>(
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

    pub fn new_uninitialized_string<B: Sized>(
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

    pub fn host_data<T: Sized>(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.data_memory
            .to_vec::<T>()
            .map_err(|e| format!("Failed to copy data from device: {}", e).into())
    }

    pub fn host_bitmap<B: Sized>(&self) -> Result<Option<Vec<B>>, Box<dyn Error>> {
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
