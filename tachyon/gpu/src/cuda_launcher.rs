/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::ffi::CString;
use std::ptr;

use indoc::formatdoc;
use sha2::{Digest, Sha256};

use crate::error::GpuResult;
use crate::ffi::column::{Column, ColumnFFI};
use crate::ffi::cuda_error::{CudaResult, CudaResultExt, KernelErrorCode};
use crate::ffi::cuda_runtime::*;
use crate::ffi::memory::cuda_memory::CudaMemory;
use crate::ffi::memory::device_memory::DeviceMemory;
use crate::ffi::nvrtc::*;
use crate::kernel_cache::get_or_compile_kernel;

#[inline(always)]
fn compose_kernel_source(kernel_name: &str, code: &str) -> String {
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
    kernel_source
}

pub fn kernel_name(code: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(code.as_bytes());
    let result = hasher.finalize();

    format!("kernel_{:x}", result)
}

fn build_or_load_kernel(
    kernel_name: &str, kernel_source: &str, device: i32,
) -> GpuResult<*const std::ffi::c_void> {
    let cache_key = format!("{}-{}", kernel_name, device);
    let cubin =
        get_or_compile_kernel(&cache_key, || build_kernel(kernel_name, kernel_source, device))?;

    cuda::create_context(device)?;
    println!("Loading CUBIN module...");

    let mut module: *mut std::ffi::c_void = ptr::null_mut();
    cuda::module_load_data(&mut module, &cubin)?;

    let mut kernel: *mut std::ffi::c_void = ptr::null_mut();
    cuda::module_get_function(&mut kernel, module, kernel_name)?;
    Ok(kernel)
}

fn build_kernel(kernel_name: &str, kernel_source: &str, device: i32) -> CudaResult<Vec<u8>> {
    let mut prog: *mut std::ffi::c_void = ptr::null_mut();
    nvrtc_wrap::create_program(&mut prog, kernel_name, kernel_source)?;
    println!("Compiling kernel to CUBIN (fat binary)...");
    let arch_flag = cuda::get_arch_flag(device)?;

    let kernel_dir = get_kernel_path("src/ffi/kernels");
    let include_dir = format!("-I{}", kernel_dir.display());
    println!("Kernel directory: {}", include_dir);
    let options = [arch_flag.as_str(), "--std=c++20", &include_dir, "-I/usr/local/cuda/include"];
    let c_options: Vec<CString> = options.iter().map(|s| CString::new(*s).unwrap()).collect();
    nvrtc_wrap::compile_program(prog, &c_options)?;
    println!("Getting CUBIN...");
    let cubin = nvrtc_wrap::get_cubin(prog)?;
    nvrtc_wrap::destroy_program(prog)?;
    Ok(cubin)
}

fn get_kernel_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("LIB_ROOT")).join(relative)
}

async fn launch_kernel(
    kernel: *const std::ffi::c_void, context_ptr: u64, input_ptr: u64, output_ptr: u64, size: usize,
) -> GpuResult<()> {
    let mut args = [
        &context_ptr as *const u64 as *mut std::ffi::c_void,
        &input_ptr as *const u64 as *mut std::ffi::c_void,
        &output_ptr as *const u64 as *mut std::ffi::c_void,
        &size as *const usize as *mut std::ffi::c_void,
    ];
    let block_size = 256;
    let grid_size = size.div_ceil(block_size);
    println!("Launching kernel with grid size {} and block size {}", grid_size, block_size);

    cuda::launch_kernel(kernel, grid_size as u32, block_size as u32, args.as_mut_ptr()).await?;
    cuda::synchronize()?;
    Ok(())
}

pub async fn launch(code: &str, input: &[Column], output: &[Column]) -> GpuResult<()> {
    cuda::init_cuda()?;
    let device = cuda::get_device(0)?;

    let kernel_name = kernel_name(code);
    let kernel_source = compose_kernel_source(&kernel_name, code);
    println!("{:#}", kernel_source);
    let kernel = build_or_load_kernel(&kernel_name, &kernel_source, device)?;

    let input_ffi: Vec<ColumnFFI> = input.iter().map(|col| col.as_ffi_column()).collect();
    let output_ffi: Vec<ColumnFFI> = output.iter().map(|col| col.as_ffi_column()).collect();

    let host_ctx = ContextFFI { error_code: 0 };
    let device_ctx = DeviceMemory::from_slice(&[host_ctx])?;
    let dm_input = DeviceMemory::from_slice(&input_ffi)?;
    let dm_output = DeviceMemory::from_slice(&output_ffi)?;
    if !input.is_empty() {
        launch_kernel(
            kernel,
            device_ctx.device_ptr() as u64,
            dm_input.device_ptr() as u64,
            dm_output.device_ptr() as u64,
            input[0].num_rows,
        )
        .await?;
    }
    cuda::last_error()?;

    let host_ctx = device_ctx.to_vec::<ContextFFI>().unwrap();

    (host_ctx[0].error_code as KernelErrorCode).check_with_context("Kernel Error")?;

    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ContextFFI {
    pub error_code: u32,
}
