/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NvrtcResult {
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    Compilation = 6,
    InternalError = 11,
    Unknown = 999,
}

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

    pub fn nvrtcGetErrorString(result: NvrtcResult) -> *const i8;

    pub fn nvrtcGetProgramLogSize(prog: *mut std::ffi::c_void, log_size: *mut usize)
    -> NvrtcResult;

    pub fn nvrtcGetProgramLog(prog: *mut std::ffi::c_void, log: *mut i8) -> NvrtcResult;
}

pub fn check_nvrtc(
    result: NvrtcResult, msg: &str, prog: *mut std::ffi::c_void,
) -> Result<(), String> {
    if result != NvrtcResult::Success {
        unsafe {
            let err_str = std::ffi::CStr::from_ptr(nvrtcGetErrorString(result));

            // Try to get compilation log
            let mut log_size: usize = 0;
            if nvrtcGetProgramLogSize(prog, &mut log_size) == NvrtcResult::Success && log_size > 1 {
                let mut log = vec![0u8; log_size];
                if nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut i8) == NvrtcResult::Success {
                    let log_str = String::from_utf8_lossy(&log);
                    let error = format!("{}: {}", msg, err_str.to_string_lossy());
                    println!("Error: {}", error);
                    println!("Log:\n{}", log_str);
                    return Err(error);
                }
            }

            return Err(format!("{}: {}", msg, err_str.to_string_lossy()));
        }
    }
    Ok(())
}
