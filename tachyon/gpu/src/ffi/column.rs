/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::error::Error;

use crate::ffi::memory::gpu_memory::{GpuMemory, MemoryType};
pub struct Column {
    data_memory: GpuMemory,
    validity_memory: Option<GpuMemory>,
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

        Ok(Column { data_memory, validity_memory, num_rows: data.len() })
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

        Ok(Column { data_memory, validity_memory, num_rows })
    }

    pub fn len(&self) -> usize {
        self.num_rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_ffi_column<B: Sized>(&self) -> ColumnFFI<B> {
        let data_ptr = self.data_memory.device_ptr();

        let validity_ptr =
            self.validity_memory.as_ref().map_or(std::ptr::null(), |vm| vm.device_ptr());

        ColumnFFI {
            data: data_ptr as *const std::os::raw::c_void,
            null_bits: validity_ptr as *const B,
            size: self.data_memory.len(),
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
}

#[repr(C)]
#[derive(Debug)]
pub struct ColumnFFI<B: Sized> {
    pub data: *const std::os::raw::c_void,
    pub null_bits: *const B,
    pub size: usize,
}
