/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use tracing::debug;

use crate::ffi::cuda_error::{CudaError, CudaResult};
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

        let memory = DeviceMemory { device_ptr: ptr, len, _phantom: PhantomData };
        // Null rows and unwritten outputs must have initialized bytes before any download.
        // Keep the allocation owned so a failed memset also frees it.
        if len > 0 {
            cuda_rt::cuda_memset(memory.device_ptr, 0, len)?;
        }
        Ok(memory)
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

    fn to_vec<T: bytemuck::CheckedBitPattern>(&self) -> CudaResult<Vec<T>> {
        check_element_size::<T>(self.len)?;
        let mut bytes = vec![0u8; self.len];
        if !bytes.is_empty() {
            cuda_rt::cuda_memcpy(
                bytes.as_mut_ptr().cast(),
                self.device_ptr.cast_const(),
                self.len,
                MemcpyKind::DeviceToHost,
            )?;
        }
        decode_values(&bytes)
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        debug!("Dropping DeviceMemory of type with size {}", self.len);
        let _ = cuda_rt::cuda_free(self.device_ptr);
    }
}

fn check_element_size<T>(len: usize) -> CudaResult<()> {
    let size = std::mem::size_of::<T>();
    if size == 0 || !len.is_multiple_of(size) {
        return Err(CudaError::Other(
            "Device byte length is not a multiple of a nonzero element size".into(),
        ));
    }
    Ok(())
}

// Decode bytes rather than assuming that a successful memcpy creates valid Rust values.
fn decode_values<T: bytemuck::CheckedBitPattern>(bytes: &[u8]) -> CudaResult<Vec<T>> {
    check_element_size::<T>(bytes.len())?;
    bytes
        .chunks_exact(std::mem::size_of::<T>())
        .map(|chunk| {
            bytemuck::checked::try_pod_read_unaligned(chunk)
                .map_err(|e| CudaError::Other(format!("Invalid device value: {e}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_boolean_bytes() {
        assert!(decode_values::<bool>(&[2]).is_err());
        assert!(decode_values::<bool>(&[0, 1, 255]).is_err());
        assert_eq!(decode_values::<bool>(&[0, 1, 0]).unwrap(), [false, true, false]);
    }

    #[test]
    fn rejects_partial_and_zero_sized_elements() {
        assert!(decode_values::<u32>(&[1, 2, 3]).is_err());
        assert!(decode_values::<()>(&[]).is_err());
        assert!(decode_values::<u64>(&[]).unwrap().is_empty());
    }

    #[test]
    fn decodes_supported_values_without_alignment_requirements() {
        let bytes = [0, 1, 2, 3, 4];
        assert_eq!(decode_values::<u32>(&bytes[1..]).unwrap(), [u32::from_ne_bytes([1, 2, 3, 4])]);
        assert_eq!(
            decode_values::<crate::column::StringView>(&[0; 16]).unwrap(),
            [crate::column::StringView::default()]
        );
        assert_eq!(decode_values::<half::f16>(&[0; 2]).unwrap(), [half::f16::ZERO]);
        assert_eq!(decode_values::<half::bf16>(&[0; 2]).unwrap(), [half::bf16::ZERO]);
    }
}
