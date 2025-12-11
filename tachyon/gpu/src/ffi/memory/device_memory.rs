/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;

use crate::ffi::cuda_error::CudaResult;
use crate::ffi::cuda_runtime::{MemcpyKind, cuda_rt};
use crate::ffi::memory::cuda_memory::CudaMemory;

/// Device-only memory with lazy pinned staging buffer
pub struct DeviceMemory {
    device_ptr: *mut c_void,
    len: usize,
    _phantom: PhantomData<c_void>,
}

impl DeviceMemory {
    pub fn new(len: usize) -> CudaResult<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();

        cuda_rt::cuda_malloc(&mut ptr as *mut *mut c_void, len)?;

        Ok(DeviceMemory { device_ptr: ptr, len, _phantom: PhantomData })
    }

    fn copy_from_host<T: Sized>(&mut self, host_data: &[T]) -> CudaResult<()> {
        let size = std::mem::size_of_val(host_data);
        if size > self.len {
            panic!("Host data size exceeds device memory capacity");
        }

        cuda_rt::cuda_memcpy(
            self.device_ptr,
            host_data.as_ptr() as *const c_void,
            size,
            MemcpyKind::HostToDevice,
        )
    }

    fn copy_to_host<T: Sized>(&self, host_data: &mut [MaybeUninit<T>]) -> CudaResult<()> {
        let size_bytes = host_data.len() * std::mem::size_of::<T>();
        if size_bytes < self.len {
            panic!("Host buffer too small for device data");
        }

        cuda_rt::cuda_memcpy(
            host_data.as_mut_ptr() as *mut c_void,
            self.device_ptr as *const c_void,
            size_bytes,
            MemcpyKind::DeviceToHost,
        )
    }
}

impl CudaMemory for DeviceMemory {
    fn device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }

    fn len(&self) -> usize {
        self.len
    }

    fn from_slice<T: Sized>(host_data: &[T]) -> CudaResult<Self> {
        let mut device_mem = Self::new(std::mem::size_of_val(host_data))?;
        device_mem.copy_from_host(host_data)?;
        Ok(device_mem)
    }

    fn to_vec<T: Sized>(&self) -> CudaResult<Vec<T>> {
        let count = self.len() / std::mem::size_of::<T>();

        let mut host_vec: Vec<MaybeUninit<T>> = {
            let mut v = Vec::with_capacity(count);
            // SAFETY: We are setting the length to the capacity we just allocated.
            // The elements are still uninitialized, but they are wrapped in MaybeUninit.
            unsafe { v.set_len(count) };
            v
        };

        self.copy_to_host(host_vec.as_mut_slice())?;

        // SAFETY: We assume `copy_to_host` successfully initialized all elements to valid `T` values.
        let host_vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(host_vec) };

        Ok(host_vec)
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        println!("Dropping DeviceMemory of type with size {}", self.len);
        let _ = cuda_rt::cuda_free(self.device_ptr);
    }
}
