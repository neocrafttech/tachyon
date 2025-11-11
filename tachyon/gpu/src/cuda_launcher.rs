/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::ffi::CString;
use std::ptr;

use crate::cuda_type::CudaType;
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
fn compose_kernel_source<T: CudaType>() -> Result<(String, String), String> {
    let type_name = T::cuda_type_name();
    let kernel_name = format!("add_vectors_{}", type_name.replace(" ", "_"));

    let kernel_source = format!(
        r#"
        typedef char int8_t;
        typedef unsigned char uint8_t;
        typedef short int16_t;
        typedef unsigned short uint16_t;
        typedef int int32_t;
        typedef unsigned int uint32_t;
        typedef long long int64_t;
        typedef unsigned long long uint64_t;
        extern "C" __global__ void {kernel_name}(
        {type_name}* a,
        {type_name}* b,
        {type_name}* c,
        int n){{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {{
                c[idx] = a[idx] + b[idx];
            }}
        }}
        "#
    );

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
    let arch_options =
        [CString::new(arch_flag.as_str()).unwrap(), CString::new("--std=c++20").unwrap()];
    let arch_ptrs: Vec<*const i8> = arch_options.iter().map(|s| s.as_ptr()).collect();

    unsafe {
        check_nvrtc(
            nvrtcCompileProgram(prog, arch_ptrs.len() as i32, arch_ptrs.as_ptr()),
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
    kernel: *const std::ffi::c_void, (d_a, d_b, d_c, n): (u64, u64, u64, usize),
) -> Result<(), String> {
    let mut args = [
        &d_a as *const u64 as *mut std::ffi::c_void,
        &d_b as *const u64 as *mut std::ffi::c_void,
        &d_c as *const u64 as *mut std::ffi::c_void,
        &n as *const usize as *mut std::ffi::c_void,
    ];
    let block_size = 256;
    let grid_size = n.div_ceil(block_size);
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

pub fn launch<T>(input: &[&Vec<T>]) -> Result<Vec<Vec<T>>, String>
where
    T: CudaType,
{
    init_cuda()?;
    let device = get_device()?;
    let (kernel_source, kernel_name) = compose_kernel_source::<T>()?;
    let kernel = build_or_load_kernel(&kernel_source, &kernel_name, device)?;

    let h_a = &input[0];
    let h_b = &input[1];
    let n = h_a.len();

    let dm_a = DeviceMemory::from_slice(h_a)
        .map_err(|e| format!("Failed to allocate device memory for input A: {}", e))?;
    let dm_b = DeviceMemory::from_slice(h_b)
        .map_err(|e| format!("Failed to allocate device memory for input B: {}", e))?;
    let dm_c = DeviceMemory::<T>::new(n)
        .map_err(|e| format!("Failed to allocate device memory for output C: {}", e))?;

    launch_kernel(
        kernel,
        (dm_a.device_ptr() as u64, dm_b.device_ptr() as u64, dm_c.device_ptr() as u64, n),
    )?;

    let error = unsafe { cudaGetLastError() };
    check_cuda(error, "Failed to get last error")?;

    let host_vec = dm_c.to_vec().map_err(|e| format!("Failed to copy data from device: {}", e))?;

    Ok(vec![host_vec])
}
