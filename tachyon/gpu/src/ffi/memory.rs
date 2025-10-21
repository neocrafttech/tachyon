use std::clone::Clone;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use std::cell::RefCell;

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
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
    ) -> CudaError;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
        stream: *mut c_void,
    ) -> CudaError;
    pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: u32) -> CudaError;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> CudaError;
}

/// Trait for CUDA memory types
pub trait CudaMemory<T: std::clone::Clone + Sized> {
    /// Get the device pointer (always returns device memory pointer)
    fn device_ptr(&self) -> *mut T;

    /// Get the host pointer (returns valid host-accessible pointer)
    /// For DeviceMemory: creates staging buffer and copies data
    /// For ManagedMemory: returns same pointer
    fn host_ptr(&mut self) -> Result<*mut T, CudaError>;

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
    fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), CudaError>;

    /// Copy data from device to host slice
    fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), CudaError>;

    /// Create from host data (allocate + copy)
    fn from_slice(host_data: &[T]) -> Result<Self, CudaError>
    where
        Self: Sized;

    /// Copy all data to a new Vec on host
    fn to_vec(&self) -> Result<Vec<T>, CudaError> {
        let mut host_vec = vec![unsafe { std::mem::zeroed() }; self.len()];
        self.copy_to_host(&mut host_vec)?;
        Ok(host_vec)
    }
}

/// Device-only memory with lazy pinned staging buffer
pub struct DeviceMemory<T> {
    device_ptr: *mut T,
    len: usize,
    // Lazy-allocated pinned host staging buffer
    staging_buffer: RefCell<Option<*mut T>>,
    _phantom: PhantomData<T>,
}

impl<T> DeviceMemory<T> {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();

        unsafe {
            let err = cudaMalloc(&mut ptr as *mut *mut c_void, size);
            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(DeviceMemory {
            device_ptr: ptr as *mut T,
            len,
            staging_buffer: RefCell::new(None),
            _phantom: PhantomData,
        })
    }

    /// Allocate staging buffer if not already allocated
    fn ensure_staging_buffer(&self) -> Result<*mut T, CudaError> {
        let mut staging = self.staging_buffer.borrow_mut();

        if let Some(ptr) = *staging {
            return Ok(ptr);
        }

        // Allocate pinned host memory
        let mut ptr: *mut c_void = ptr::null_mut();
        let size = self.len * std::mem::size_of::<T>();

        unsafe {
            let err = cudaMallocHost(&mut ptr as *mut *mut c_void, size);
            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        let typed_ptr = ptr as *mut T;
        *staging = Some(typed_ptr);
        Ok(typed_ptr)
    }

    /// Free staging buffer if allocated
    fn free_staging_buffer(&self) {
        let mut staging = self.staging_buffer.borrow_mut();

        if let Some(ptr) = *staging {
            unsafe {
                cudaFreeHost(ptr as *mut c_void);
            }
            *staging = None;
        }
    }
}

impl<T: std::clone::Clone> CudaMemory<T> for DeviceMemory<T> {
    fn device_ptr(&self) -> *mut T {
        self.device_ptr
    }

    fn host_ptr(&mut self) -> Result<*mut T, CudaError> {
        // Ensure staging buffer exists
        let host_ptr = self.ensure_staging_buffer()?;

        // Synchronize: copy device -> host
        self.sync_host()?;

        Ok(host_ptr)
    }

    fn sync_host(&mut self) -> Result<(), CudaError> {
        let host_ptr = self.ensure_staging_buffer()?;
        let size = self.len * std::mem::size_of::<T>();

        unsafe {
            let err = cudaMemcpyAsync(
                host_ptr as *mut c_void,
                self.device_ptr as *const c_void,
                size,
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
            let size = self.len * std::mem::size_of::<T>();

            unsafe {
                let err = cudaMemcpyAsync(
                    self.device_ptr as *mut c_void,
                    host_ptr as *const c_void,
                    size,
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

    fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), CudaError> {
        if host_data.len() > self.len {
            panic!("Host data size exceeds device memory capacity");
        }

        let size = std::mem::size_of_val(host_data);

        unsafe {
            let err = cudaMemcpy(
                self.device_ptr as *mut c_void,
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

    fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), CudaError> {
        if host_data.len() < self.len {
            panic!("Host buffer too small for device data");
        }

        let size = self.len * std::mem::size_of::<T>();

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

    fn from_slice(host_data: &[T]) -> Result<Self, CudaError> {
        let mut device_mem = Self::new(host_data.len())?;
        device_mem.copy_from_host(host_data)?;
        Ok(device_mem)
    }
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        println!(
            "Dropping DeviceMemory of type {} with size {}",
            std::any::type_name::<T>(),
            self.len
        );
        // Free staging buffer first
        self.free_staging_buffer();

        // Free device memory
        unsafe {
            cudaFree(self.device_ptr as *mut c_void);
        }
    }
}

unsafe impl<T: Send> Send for DeviceMemory<T> {}
unsafe impl<T: Sync> Sync for DeviceMemory<T> {}

/// Managed/Unified memory - host_ptr returns same pointer
pub struct ManagedMemory<T> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T> ManagedMemory<T> {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();

        unsafe {
            let err = cudaMallocManaged(&mut ptr as *mut *mut c_void, size, CU_MEM_ATTACH_GLOBAL);
            if err != CUDA_SUCCESS {
                return Err(err);
            }
        }

        Ok(ManagedMemory {
            ptr: ptr as *mut T,
            len,
            _phantom: PhantomData,
        })
    }

    /// Direct access as slice (since managed memory is host-accessible)
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Direct mutable access as slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Clone> CudaMemory<T> for ManagedMemory<T> {
    fn device_ptr(&self) -> *mut T {
        self.ptr
    }

    fn host_ptr(&mut self) -> Result<*mut T, CudaError> {
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

    fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), CudaError> {
        if host_data.len() > self.len {
            panic!("Host data size exceeds memory capacity");
        }

        let size = std::mem::size_of_val(host_data);

        unsafe {
            let err = cudaMemcpy(
                self.ptr as *mut c_void,
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

    fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), CudaError> {
        if host_data.len() < self.len {
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

    fn from_slice(host_data: &[T]) -> Result<Self, CudaError> {
        let mut managed_mem = Self::new(host_data.len())?;
        managed_mem.copy_from_host(host_data)?;
        Ok(managed_mem)
    }
}

impl<T> Drop for ManagedMemory<T> {
    fn drop(&mut self) {
        println!(
            "Dropping ManagedMemory of type {} with size {}",
            std::any::type_name::<T>(),
            self.len
        );
        unsafe {
            cudaFree(self.ptr as *mut c_void);
        }
    }
}

unsafe impl<T: Send> Send for ManagedMemory<T> {}
unsafe impl<T: Sync> Sync for ManagedMemory<T> {}

// Add missing constants
