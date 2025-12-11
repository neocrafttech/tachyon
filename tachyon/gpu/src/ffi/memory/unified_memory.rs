/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use crate::ffi::cuda_error::CudaResult;
use crate::ffi::cuda_runtime::{CU_MEM_ATTACH_GLOBAL, cuda_rt};
use crate::ffi::memory::cuda_memory::CudaMemory;

/// Unified/Managed memory - host_ptr and device_ptr return the same pointer
pub struct UnifiedMemory {
    ptr: *mut c_void,
    len: usize, // Storing the size in bytes is critical for cudaFree
    _phantom: PhantomData<c_void>,
}

impl UnifiedMemory {
    pub fn new(len_bytes: usize) -> CudaResult<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();
        cuda_rt::cuda_malloc_managed(
            &mut ptr as *mut *mut c_void,
            len_bytes,
            CU_MEM_ATTACH_GLOBAL,
        )?;

        Ok(UnifiedMemory { ptr, len: len_bytes, _phantom: PhantomData })
    }

    /// Direct access as slice (since managed memory is host-accessible)
    /// SAFETY: Caller must ensure that the memory has been initialized by the GPU/CPU.
    pub fn as_slice<T>(&self) -> &[T] {
        let count = self.len / std::mem::size_of::<T>();
        // SAFETY: The memory was allocated for len_bytes and is accessible.
        // The user must ensure the memory contains valid T values.
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, count) }
    }

    /// Direct mutable access as slice
    /// SAFETY: Caller must ensure that subsequent GPU operations don't write to this memory
    /// while the slice is held mutably.
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        let count = self.len / std::mem::size_of::<T>();
        // SAFETY: The memory was allocated for len_bytes and is accessible.
        // The user must ensure the memory contains valid T values.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, count) }
    }
}

impl CudaMemory for UnifiedMemory {
    fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }

    fn from_slice<T: Sized>(_host_data: &[T]) -> CudaResult<Self> {
        unimplemented!()
    }
    fn to_vec<T: Sized>(&self) -> CudaResult<Vec<T>> {
        unimplemented!()
    }
}

impl Drop for UnifiedMemory {
    fn drop(&mut self) {
        eprintln!("Dropping ManagedMemory with size {} bytes", self.len);
        let _ = cuda_rt::cuda_free(self.ptr);
    }
}
