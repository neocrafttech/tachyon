/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
use std::ffi::c_void;
use std::ptr;

use crate::ffi::cuda_error::{CUresult, CudaErrorType, CudaResult, CudaResultExt};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
}

pub const CU_MEM_ATTACH_GLOBAL: u32 = 0x01;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: u32) -> CUresult;

    pub fn cuDeviceGet(device: *mut i32, ordinal: i32) -> CUresult;

    pub fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> CUresult;

    pub fn cuCtxCreate_v2(pctx: *mut *mut std::ffi::c_void, flags: u32, dev: i32) -> CUresult;

    pub fn cuModuleLoadData(
        module: *mut *mut std::ffi::c_void, image: *const std::ffi::c_void,
    ) -> CUresult;

    pub fn cuModuleGetFunction(
        hfunc: *mut *mut std::ffi::c_void, hmod: *mut std::ffi::c_void, name: *const i8,
    ) -> CUresult;

    pub fn cuLaunchKernel(
        f: *const std::ffi::c_void, grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32, shared_mem_bytes: u32,
        h_stream: *mut std::ffi::c_void, kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CUresult;

    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cudaGetLastError() -> CUresult;
}

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> CudaErrorType;

    pub fn cudaFree(devPtr: *mut c_void) -> CudaErrorType;

    pub fn cudaMemcpy(
        dst: *mut c_void, src: *const c_void, count: usize, kind: MemcpyKind,
    ) -> CudaErrorType;

    pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: u32) -> CudaErrorType;
}

pub mod cuda {
    use super::*;
    #[inline]
    pub fn init_cuda() -> CudaResult<()> {
        unsafe { cuInit(0) }.check_with_context("cuInit")
    }

    #[inline]
    pub fn get_device(ordinal: i32) -> CudaResult<i32> {
        let mut device = 0;
        unsafe { cuDeviceGet(&mut device, ordinal) }.check_with_context("cuDeviceGet")?;
        Ok(device)
    }

    #[inline]
    pub fn get_arch_flag(device: i32) -> CudaResult<String> {
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            cuDeviceGetAttribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
        }
        .check_with_context("cuDeviceGetAttribute")?;
        unsafe {
            cuDeviceGetAttribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
        }
        .check_with_context("cuDeviceGetAttribute")?;
        println!("Device compute capability: {}.{}", major, minor);
        let arch_flag = format!("--gpu-architecture=sm_{}{}", major, minor);
        Ok(arch_flag)
    }

    #[inline]
    pub async fn launch_kernel(
        kernel: *const std::ffi::c_void, grid_size: u32, block_size: u32,
        args: *mut *mut std::ffi::c_void,
    ) -> CudaResult<()> {
        unsafe {
            cuLaunchKernel(
                kernel,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                ptr::null_mut(),
                args,
                ptr::null_mut(),
            )
        }
        .check_with_context("cuLaunchKernel")?;
        Ok(())
    }

    #[inline]
    pub fn synchronize() -> CudaResult<()> {
        unsafe { cuCtxSynchronize() }.check_with_context("cuCtxSynchronize")
    }

    #[inline]
    pub fn create_context(device: i32) -> CudaResult<()> {
        let mut context: *mut std::ffi::c_void = ptr::null_mut();
        unsafe { cuCtxCreate_v2(&mut context, 0, device) }.check_with_context("cuCtxCreate_v2")
    }

    #[inline]
    pub fn module_load_data(module: &mut *mut c_void, cubin: &[u8]) -> CudaResult<()> {
        unsafe { cuModuleLoadData(module as *mut *mut c_void, cubin.as_ptr() as *const c_void) }
            .check_with_context("cuModuleLoadData")
    }

    #[inline]
    pub fn module_get_function(
        function: &mut *mut c_void, module: *mut std::ffi::c_void, name: &str,
    ) -> CudaResult<()> {
        unsafe {
            cuModuleGetFunction(
                function as *mut *mut c_void,
                module,
                name.as_ptr() as *const std::ffi::c_char,
            )
        }
        .check_with_context("cuModuleGetFunction")
    }

    #[inline]
    pub fn last_error() -> CudaResult<()> {
        unsafe { cudaGetLastError() }.check_with_context("cuGetLastError")
    }
}

pub mod cuda_rt {
    use super::*;
    #[inline]
    pub fn cuda_malloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaResult<()> {
        unsafe { cudaMalloc(dev_ptr, size) }.check_with_context("cudaMalloc")
    }

    #[inline]
    pub fn cuda_free(dev_ptr: *mut c_void) -> CudaResult<()> {
        unsafe { cudaFree(dev_ptr) }.check_with_context("cudaFree")
    }

    #[inline]
    pub fn cuda_memcpy(
        dst: *mut c_void, src: *const c_void, count: usize, kind: MemcpyKind,
    ) -> CudaResult<()> {
        unsafe { cudaMemcpy(dst, src, count, kind) }.check_with_context("cudaMemcpy")
    }

    #[inline]
    pub fn cuda_malloc_managed(
        dev_ptr: *mut *mut c_void, size: usize, flags: u32,
    ) -> CudaResult<()> {
        unsafe { cudaMallocManaged(dev_ptr, size, flags) }.check_with_context("cudaMallocManaged")
    }
}
