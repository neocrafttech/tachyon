/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::ffi::{CString, c_void};
use std::ptr;

use crate::ffi::cuda_error::{
    CUDA_SUCCESS, CudaResult, CudaResultExt, NvrtcResult, nvrtcGetErrorString,
};

#[link(name = "nvrtc")]
unsafe extern "C" {
    pub fn nvrtcCreateProgram(
        prog: *mut *mut std::ffi::c_void, src: *const i8, name: *const i8, num_headers: i32,
        headers: *const *const i8, include_names: *const *const i8,
    ) -> NvrtcResult;

    pub fn nvrtcCompileProgram(
        prog: *mut std::ffi::c_void, num_options: i32, options: *const *const i8,
    ) -> NvrtcResult;

    pub fn nvrtcGetCUBINSize(prog: *mut std::ffi::c_void, cubin_size: *mut usize) -> NvrtcResult;

    pub fn nvrtcGetCUBIN(prog: *mut std::ffi::c_void, cubin: *mut i8) -> NvrtcResult;

    pub fn nvrtcDestroyProgram(prog: *mut *mut std::ffi::c_void) -> NvrtcResult;

    pub fn nvrtcGetProgramLogSize(prog: *mut std::ffi::c_void, log_size: *mut usize)
    -> NvrtcResult;

    pub fn nvrtcGetProgramLog(prog: *mut std::ffi::c_void, log: *mut i8) -> NvrtcResult;
}

pub fn log_nvrtc(result: NvrtcResult, msg: &str, prog: *mut std::ffi::c_void) {
    if result != CUDA_SUCCESS as u16 {
        unsafe {
            let err_str = std::ffi::CStr::from_ptr(nvrtcGetErrorString(result));

            // Try to get compilation log
            let mut log_size: usize = 0;
            if nvrtcGetProgramLogSize(prog, &mut log_size) == CUDA_SUCCESS as u16 && log_size > 1 {
                let mut log = vec![0u8; log_size];
                if nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut i8) == CUDA_SUCCESS as u16 {
                    let log_str = String::from_utf8_lossy(&log);
                    let error = format!("{}: {}", msg, err_str.to_string_lossy());
                    println!("Error: {}", error);
                    println!("Log:\n{}", log_str);
                }
            }
        }
    }
}

pub mod nvrtc_wrap {
    use super::*;

    pub fn create_program(
        prog: &mut *mut c_void, kernel_name: &str, kernel_source: &str,
    ) -> CudaResult<()> {
        let name = CString::new(kernel_name).unwrap();
        let src = CString::new(kernel_source).unwrap();
        unsafe {
            nvrtcCreateProgram(
                prog as *mut *mut c_void,
                src.as_ptr(),
                name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        }
        .check_with_context("nvrtcCreateProgram")
    }

    pub fn compile_program(prog: *mut c_void, c_options: &[CString]) -> CudaResult<()> {
        let option_ptrs: Vec<*const i8> = c_options.iter().map(|s| s.as_ptr()).collect();
        let result =
            unsafe { nvrtcCompileProgram(prog, option_ptrs.len() as i32, option_ptrs.as_ptr()) };

        log_nvrtc(result, "Compile Program", prog);

        result.check_with_context("nvrtcCompileProgram")
    }

    pub fn get_cubin_size(prog: *mut c_void) -> CudaResult<usize> {
        let mut cubin_size: usize = 0;
        unsafe { nvrtcGetCUBINSize(prog, &mut cubin_size) }
            .check_with_context("nvrtcGetCUBINSize")?;
        Ok(cubin_size)
    }

    pub fn get_cubin(prog: *mut c_void) -> CudaResult<Vec<u8>> {
        let cubin_size = nvrtc_wrap::get_cubin_size(prog)?;
        let mut cubin = vec![0u8; cubin_size];
        unsafe { nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut i8) }
            .check_with_context("nvrtcGetCUBIN")?;
        Ok(cubin)
    }

    pub fn destroy_program(mut prog: *mut c_void) -> CudaResult<()> {
        unsafe { nvrtcDestroyProgram(&mut prog) }.check_with_context("nvrtcDestroyProgram")
    }
}
