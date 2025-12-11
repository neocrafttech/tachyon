use std::ffi::c_void;

use crate::ffi::cuda_error::CudaResult;
use crate::ffi::memory::cuda_memory::CudaMemory;
use crate::ffi::memory::device_memory::DeviceMemory;

pub enum GpuMemory {
    Device(DeviceMemory),
    // Unified(UnifiedMemory),
}

impl GpuMemory {
    pub fn device_ptr(&self) -> *mut c_void {
        match self {
            GpuMemory::Device(mem) => mem.device_ptr(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            GpuMemory::Device(mem) => mem.len(),
        }
    }

    pub fn to_vec<T: Sized>(&self) -> CudaResult<Vec<T>> {
        match self {
            GpuMemory::Device(mem) => mem.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryType {
    Device,
    // Unified,
}

impl MemoryType {
    pub fn allocate_from_slice<T: Sized>(&self, data: &[T]) -> CudaResult<GpuMemory> {
        match self {
            MemoryType::Device => {
                let mem = DeviceMemory::from_slice(data)?;
                Ok(GpuMemory::Device(mem))
            } // MemoryType::Unified => { ... }
        }
    }

    pub fn allocate(&self, size: usize) -> CudaResult<GpuMemory> {
        match self {
            MemoryType::Device => {
                let mem = DeviceMemory::new(size)?;
                Ok(GpuMemory::Device(mem))
            } // MemoryType::Unified => { ... }
        }
    }
}
