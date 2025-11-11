/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::ffi::CString;
use std::ptr;

use indoc::formatdoc;

use crate::ffi::column::{Column, ColumnFFI};
use crate::ffi::cuda_runtime::*;
use crate::ffi::memory::*;
use crate::ffi::nvrtc::*;

#[inline(always)]
fn init_cuda() -> Result<(), String> {
    println!("Initializing CUDA...");
    unsafe {
        check_cuda(cuInit(0), "Failed to initialize CUDA")?;
    }
    Ok(())
}

#[inline(always)]
fn get_device() -> Result<i32, String> {
    let mut device: i32 = 0;
    unsafe {
        check_cuda(cuDeviceGet(&mut device, 0), "Failed to get device")?;
    }
    Ok(device)
}

#[inline(always)]
fn get_arch_flag(device: i32) -> Result<String, String> {
    // Get compute capability
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    unsafe {
        check_cuda(
            cuDeviceGetAttribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
            "Failed to get compute capability major",
        )?;
        check_cuda(
            cuDeviceGetAttribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
            "Failed to get compute capability minor",
        )?;
    }
    println!("Device compute capability: {}.{}", major, minor);
    let arch_flag = format!("--gpu-architecture=sm_{}{}", major, minor);
    Ok(arch_flag)
}

#[inline(always)]
fn compose_kernel_source(code: &str) -> Result<(String, String), String> {
    let kernel_name = "add_vectors_123".to_string();

    let kernel_source = formatdoc! {r#"
        #include "types.cuh"
        #include "column.cuh"
        #include "context.cuh"
        #include "math.cuh"
        extern "C" __global__ void {kernel_name}(Context* ctx, Column* input, Column* output, size_t num_rows) {{
            size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (row_idx >= num_rows) return;

            {code}
        }}
    "#};
    Ok((kernel_source, kernel_name))
}

fn build_or_load_kernel(
    kernel_source: &str, kernel_name: &str, device: i32,
) -> Result<*const std::ffi::c_void, String> {
    let src = CString::new(kernel_source).unwrap();
    let name = CString::new(kernel_name).unwrap();

    let mut prog: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        check_nvrtc(
            nvrtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, ptr::null(), ptr::null()),
            "Failed to create NVRTC program",
            prog,
        )?;
    }

    println!("Compiling kernel to CUBIN (fat binary)...");

    let arch_flag = get_arch_flag(device)?;

    let kernel_dir = "/home/manish/tachyon/tachyon/gpu/src/ffi/kernels";
    let include_dir = format!("-I{}", kernel_dir);
    let options = [arch_flag.as_str(), "--std=c++20", &include_dir, "-I/usr/local/cuda/include"];

    let c_options: Vec<CString> = options.iter().map(|s| CString::new(*s).unwrap()).collect();
    let option_ptrs: Vec<*const i8> = c_options.iter().map(|s| s.as_ptr()).collect();

    unsafe {
        check_nvrtc(
            nvrtcCompileProgram(prog, option_ptrs.len() as i32, option_ptrs.as_ptr()),
            "Failed to compile kernel",
            prog,
        )?;
    }

    println!("Getting CUBIN...");

    // Get CUBIN (compiled binary)
    let mut cubin_size: usize = 0;
    unsafe {
        check_nvrtc(nvrtcGetCUBINSize(prog, &mut cubin_size), "Failed to get CUBIN size", prog)?;
    }

    println!("CUBIN size: {} bytes", cubin_size);

    let mut cubin = vec![0u8; cubin_size];
    unsafe {
        check_nvrtc(
            nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut i8),
            "Failed to get CUBIN",
            prog,
        )?;
        nvrtcDestroyProgram(&mut prog);
    }

    // Create context
    let mut context: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        check_cuda(cuCtxCreate_v2(&mut context, 0, device), "Failed to create context")?;
    }

    println!("Loading CUBIN module...");

    // Load module from CUBIN
    let mut module: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        check_cuda(
            cuModuleLoadData(&mut module, cubin.as_ptr() as *const std::ffi::c_void),
            "Failed to load CUBIN module",
        )?;
    }

    // Get kernel function
    let mut kernel: *mut std::ffi::c_void = ptr::null_mut();
    let kernel_name = CString::new(kernel_name).unwrap();

    unsafe {
        check_cuda(
            cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr()),
            "Failed to get kernel function",
        )?;
    }
    Ok(kernel)
}

fn launch_kernel(
    kernel: *const std::ffi::c_void, context_ptr: u64, input_ptr: u64, output_ptr: u64, size: usize,
) -> Result<(), String> {
    let mut args = [
        &context_ptr as *const u64 as *mut std::ffi::c_void,
        &input_ptr as *const u64 as *mut std::ffi::c_void,
        &output_ptr as *const u64 as *mut std::ffi::c_void,
        &size as *const usize as *mut std::ffi::c_void,
    ];
    let block_size = 256;
    let grid_size = size.div_ceil(block_size);
    println!("Launching kernel with grid size {} and block size {}", grid_size, block_size);
    unsafe {
        check_cuda(
            cuLaunchKernel(
                kernel,
                grid_size as u32,
                1,
                1,
                block_size as u32,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "Failed to launch kernel",
        )?;

        check_cuda(cuCtxSynchronize(), "Failed to synchronize")?;
    }
    Ok(())
}

pub fn launch(code: &str, input: &[Column], output: &[Column]) -> Result<(), String> {
    init_cuda()?;
    let device = get_device()?;
    let (kernel_source, kernel_name) = compose_kernel_source(code)?;
    println!("{:#}", kernel_source);
    let kernel = build_or_load_kernel(&kernel_source, &kernel_name, device)?;

    let input_ffi: Vec<ColumnFFI> = input.iter().map(|col| col.as_ffi_column()).collect();
    let output_ffi: Vec<ColumnFFI> = output.iter().map(|col| col.as_ffi_column()).collect();

    let host_ctx = ContextFFI { error_code: 0 };
    let device_ctx = DeviceMemory::from_slice(&[host_ctx])
        .map_err(|e| format!("Failed to allocate device memory for context: {}", e))?; //TODO: Make single value init

    let dm_input = DeviceMemory::from_slice(&input_ffi)
        .map_err(|e| format!("Failed to allocate device memory for input: {}", e))?;

    let dm_output = DeviceMemory::from_slice(&output_ffi)
        .map_err(|e| format!("Failed to allocate device memory for output: {}", e))?;

    if !input.is_empty() {
        launch_kernel(
            kernel,
            device_ctx.device_ptr() as u64,
            dm_input.device_ptr() as u64,
            dm_output.device_ptr() as u64,
            input[0].num_rows,
        )?;
    }
    let error = unsafe { cudaGetLastError() };
    check_cuda(error, "Failed to get last error")?;

    let host_ctx = device_ctx.to_vec::<ContextFFI>().unwrap();

    if host_ctx[0].error_code != 0 {
        return Err(format!("CUDA error: {}", host_ctx[0].error_code));
    }

    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ContextFFI {
    pub error_code: u32,
}
