/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

// FFI bindings
pub type CudaError = i32;
pub type CudaMemcpyKind = u32;

pub const CU_MEM_ATTACH_GLOBAL: u32 = 0x01;
pub const CUDA_SUCCESS: CudaError = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: CudaMemcpyKind = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: CudaMemcpyKind = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(devPtr: *mut c_void) -> CudaError;
    pub fn cudaFreeHost(ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpy(
        dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind,
    ) -> CudaError;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind,
        stream: *mut c_void,
    ) -> CudaError;
    pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: u32) -> CudaError;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> CudaError;
}

/// Trait for CUDA memory types
pub trait CudaMemory {
    /// Get the device pointer (always returns device memory pointer)
    fn device_ptr(&self) -> *mut c_void;

    /// Get the host pointer (returns valid host-accessible pointer)
    /// For DeviceMemory: creates staging buffer and copies data
    /// For ManagedMemory: returns same pointer
    fn host_ptr(&mut self) -> Result<*mut c_void, CudaError>;

    /// Synchronize host pointer with device (copy device -> host)
    fn sync_host(&mut self) -> Result<(), CudaError>;

    /// Synchronize device pointer with host (copy host -> device)
    fn sync_device(&mut self) -> Result<(), CudaError>;

    /// Get the number of elements
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy data from host slice to device
    fn copy_from_host<T: Sized>(&mut self, host_data: &[T]) -> Result<(), CudaError>;

    /// Copy data from device to host slice
    fn copy_to_host<T: Sized>(&self, host_data: &mut [T]) -> Result<(), CudaError>;

    /// Create from host data (allocate + copy)
    fn from_slice<T: Sized>(host_data: &[T]) -> Result<Self, CudaError>
    where
        Self: Sized;

    /// Copy all data to a new Vec on host
    fn to_vec<T: std::clone::Clone + Sized>(&self) -> Result<Vec<T>, CudaError> {
        let mut host_vec =
            vec![unsafe { std::mem::zeroed() }; self.len() / std::mem::size_of::<T>()];
        self.copy_to_host(&mut host_vec)?;
        Ok(host_vec)
    }
}

/// Device-only memory with lazy pinned staging buffer
pub struct DeviceMemory {
    device_ptr: *mut c_void,
    len: usize,
    // Lazy-allocated pinned host staging buffer
    staging_buffer: RefCell<Option<*mut c_void>>,
    _phantom: PhantomData<c_void>,
}

impl DeviceMemory {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let err = cudaMalloc(&mut ptr as *mut *mut c_void, len);
            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(DeviceMemory {
            device_ptr: ptr,
            len,
            staging_buffer: RefCell::new(None),
            _phantom: PhantomData,
        })
    }

    /// Allocate staging buffer if not already allocated
    fn ensure_staging_buffer(&self) -> Result<*mut c_void, CudaError> {
        let mut staging = self.staging_buffer.borrow_mut();

        if let Some(ptr) = *staging {
            return Ok(ptr);
        }

        // Allocate pinned host memory
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let err = cudaMallocHost(&mut ptr as *mut *mut c_void, self.len);
            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        *staging = Some(ptr);
        Ok(ptr)
    }

    /// Free staging buffer if allocated
    fn free_staging_buffer(&self) {
        let mut staging = self.staging_buffer.borrow_mut();

        if let Some(ptr) = *staging {
            unsafe {
                cudaFreeHost(ptr);
            }
            *staging = None;
        }
    }
}

impl CudaMemory for DeviceMemory {
    fn device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }

    fn host_ptr(&mut self) -> Result<*mut c_void, CudaError> {
        // Ensure staging buffer exists
        let host_ptr = self.ensure_staging_buffer()?;

        // Synchronize: copy device -> host
        self.sync_host()?;

        Ok(host_ptr)
    }

    fn sync_host(&mut self) -> Result<(), CudaError> {
        let host_ptr = self.ensure_staging_buffer()?;

        unsafe {
            let err = cudaMemcpyAsync(
                host_ptr,
                self.device_ptr,
                self.len,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                ptr::null_mut(), // default stream
            );

            if err != CUDA_SUCCESS {
                return Err(err);
            }

            // Synchronize to ensure copy is complete
            Err(cudaStreamSynchronize(ptr::null_mut()))
        }
    }

    fn sync_device(&mut self) -> Result<(), CudaError> {
        // Only sync if staging buffer exists
        let staging = self.staging_buffer.borrow();
        if let Some(host_ptr) = *staging {
            unsafe {
                let err = cudaMemcpyAsync(
                    self.device_ptr,
                    host_ptr,
                    self.len,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );

                if err != CUDA_SUCCESS {
                    return Err(err);
                }

                Err(cudaStreamSynchronize(ptr::null_mut()))
            }
        } else {
            // No staging buffer, nothing to sync
            Ok(())
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn copy_from_host<T: Sized>(&mut self, host_data: &[T]) -> Result<(), CudaError> {
        let size = std::mem::size_of_val(host_data);
        if size > self.len {
            panic!("Host data size exceeds device memory capacity");
        }

        unsafe {
            let err = cudaMemcpy(
                self.device_ptr,
                host_data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(())
    }

    fn copy_to_host<T: Sized>(&self, host_data: &mut [T]) -> Result<(), CudaError> {
        let size = std::mem::size_of_val(host_data);
        if size < self.len {
            panic!("Host buffer too small for device data");
        }

        unsafe {
            let err = cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.device_ptr as *const c_void,
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );

            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(())
    }

    fn from_slice<T: Sized>(host_data: &[T]) -> Result<Self, CudaError> {
        let mut device_mem = Self::new(std::mem::size_of_val(host_data))?;
        device_mem.copy_from_host(host_data)?;
        Ok(device_mem)
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        println!("Dropping DeviceMemory of type with size {}", self.len);
        // Free staging buffer first
        self.free_staging_buffer();

        // Free device memory
        unsafe {
            cudaFree(self.device_ptr);
        }
    }
}

/// Managed/Unified memory - host_ptr returns same pointer
pub struct ManagedMemory {
    ptr: *mut c_void,
    len: usize,
    _phantom: PhantomData<c_void>,
}

impl ManagedMemory {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let err = cudaMallocManaged(&mut ptr as *mut *mut c_void, len, CU_MEM_ATTACH_GLOBAL);
            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(ManagedMemory { ptr, len, _phantom: PhantomData })
    }

    /// Direct access as slice (since managed memory is host-accessible)
    pub fn as_slice<T: Sized>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr as *const T, self.len / std::mem::size_of::<T>())
        }
    }

    /// Direct mutable access as slice
    pub fn as_slice_mut<T: Sized>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len / std::mem::size_of::<T>())
        }
    }
}

impl CudaMemory for ManagedMemory {
    fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn host_ptr(&mut self) -> Result<*mut c_void, CudaError> {
        // Same pointer for managed memory
        Ok(self.ptr)
    }

    fn sync_host(&mut self) -> Result<(), CudaError> {
        // Managed memory handles this automatically
        // But we can add a device synchronization to ensure completion
        unsafe { Err(cudaStreamSynchronize(ptr::null_mut())) }
    }

    fn sync_device(&mut self) -> Result<(), CudaError> {
        // Managed memory handles this automatically
        unsafe { Err(cudaStreamSynchronize(ptr::null_mut())) }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn copy_from_host<T: Sized>(&mut self, host_data: &[T]) -> Result<(), CudaError> {
        let size = std::mem::size_of_val(host_data);
        if size > self.len {
            panic!("Host data size exceeds memory capacity");
        }

        unsafe {
            let err = cudaMemcpy(
                self.ptr,
                host_data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(())
    }

    fn copy_to_host<T: Sized>(&self, host_data: &mut [T]) -> Result<(), CudaError> {
        let size = std::mem::size_of_val(host_data);
        if size < self.len {
            panic!("Host buffer too small");
        }

        let size = self.len * std::mem::size_of::<T>();

        unsafe {
            let err = cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );

            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(())
    }

    fn from_slice<T: Sized>(host_data: &[T]) -> Result<Self, CudaError> {
        let mut managed_mem = Self::new(std::mem::size_of_val(host_data))?;
        managed_mem.copy_from_host(host_data)?;
        Ok(managed_mem)
    }
}

impl Drop for ManagedMemory {
    fn drop(&mut self) {
        println!("Dropping ManagedMemory with size {}", self.len);
        unsafe {
            cudaFree(self.ptr);
        }
    }
}
