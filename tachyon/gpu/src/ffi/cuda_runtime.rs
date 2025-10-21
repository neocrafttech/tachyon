#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: u32) -> i32;

    pub fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;

    pub fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> i32;

    pub fn cuCtxCreate_v2(pctx: *mut *mut std::ffi::c_void, flags: u32, dev: i32) -> i32;

    pub fn cuModuleLoadData(
        module: *mut *mut std::ffi::c_void,
        image: *const std::ffi::c_void,
    ) -> i32;

    pub fn cuModuleGetFunction(
        hfunc: *mut *mut std::ffi::c_void,
        hmod: *mut std::ffi::c_void,
        name: *const i8,
    ) -> i32;

    pub fn cuLaunchKernel(
        f: *const std::ffi::c_void,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        h_stream: *mut std::ffi::c_void,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> i32;

    pub fn cuCtxSynchronize() -> i32;

    pub fn cudaGetLastError() -> i32;
}

pub const CUDA_SUCCESS: i32 = 0;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

pub fn check_cuda(result: i32, msg: &str) -> Result<(), String> {
    if result != CUDA_SUCCESS {
        return Err(format!("{}: error code {}", msg, result));
    }
    Ok(())
}
