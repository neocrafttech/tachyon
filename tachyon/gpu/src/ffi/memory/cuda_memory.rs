/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::ffi::c_void;

use crate::ffi::cuda_error::CudaResult;

/// Trait for CUDA memory types
pub trait CudaMemory {
    /// Get the device pointer (always returns device memory pointer)
    fn device_ptr(&self) -> *mut c_void;

    /// Get the number of elements
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create from host data (allocate + copy)
    fn from_slice<T: Sized>(host_data: &[T]) -> CudaResult<Self>
    where
        Self: Sized;

    /// Copy all data to a new Vec on host
    fn to_vec<T: Sized>(&self) -> CudaResult<Vec<T>>;
}
