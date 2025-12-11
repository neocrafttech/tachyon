use std::ffi::CStr;
use std::fmt;

use thiserror::Error;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuGetErrorString(error: CUresult, pStr: *mut *const i8) -> CUresult;

    pub fn cuGetErrorName(error: CUresult, pStr: *mut *const i8) -> CUresult;
}

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaGetErrorString(error: CudaErrorType) -> *const i8;

    pub fn cudaGetErrorName(error: CudaErrorType) -> *const i8;
}

#[link(name = "nvrtc")]
unsafe extern "C" {
    pub fn nvrtcGetErrorString(result: NvrtcResult) -> *const i8;
}

#[link(name = "tachyon_kernel_lib")]
unsafe extern "C" {
    pub fn kernelGetErrorString(result: KernelErrorCode) -> *const i8;
}

pub const CUDA_SUCCESS: i32 = 0;
pub type KernelErrorCode = i16;
pub type NvrtcResult = u16;
pub type CUresult = u32;
pub type CudaErrorType = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelError(pub KernelErrorCode);

impl KernelError {
    pub fn is_success(&self) -> bool {
        self.0 == CUDA_SUCCESS as i16
    }

    pub fn error_string(&self) -> String {
        unsafe {
            let result = kernelGetErrorString(self.0);

            if result.is_null() {
                return format!("Unknown CUresult: {}", self.0);
            }

            CStr::from_ptr(result).to_str().unwrap_or("Invalid UTF-8").to_string()
        }
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.error_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvrtcError(pub NvrtcResult);

impl NvrtcError {
    pub fn is_success(&self) -> bool {
        self.0 == CUDA_SUCCESS as u16
    }

    pub fn error_string(&self) -> String {
        unsafe {
            let result = nvrtcGetErrorString(self.0);

            if result.is_null() {
                return format!("Unknown CUresult: {}", self.0);
            }

            CStr::from_ptr(result).to_str().unwrap_or("Invalid UTF-8").to_string()
        }
    }
}

impl fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.error_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDriverError(pub CUresult);

impl CudaDriverError {
    pub fn is_success(&self) -> bool {
        self.0 == CUDA_SUCCESS as u32
    }

    pub fn error_string(&self) -> String {
        unsafe {
            let mut ptr: *const i8 = std::ptr::null();
            let result = cuGetErrorString(self.0, &mut ptr);

            if result != CUDA_SUCCESS as u32 || ptr.is_null() {
                return format!("Unknown CUresult: {}", self.0);
            }

            CStr::from_ptr(ptr).to_str().unwrap_or("Invalid UTF-8").to_string()
        }
    }

    pub fn error_name(&self) -> String {
        unsafe {
            let mut ptr: *const i8 = std::ptr::null();
            let result = cuGetErrorName(self.0, &mut ptr);

            if result != CUDA_SUCCESS as u32 || ptr.is_null() {
                return format!("UNKNOWN_ERROR_{}", self.0);
            }

            CStr::from_ptr(ptr).to_str().unwrap_or("INVALID_UTF8").to_string()
        }
    }
}

impl fmt::Display for CudaDriverError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.error_name(), self.error_string())
    }
}

impl std::error::Error for CudaDriverError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaRuntimeError(pub CudaErrorType);

impl CudaRuntimeError {
    pub fn is_success(&self) -> bool {
        self.0 == CUDA_SUCCESS
    }

    pub fn error_string(&self) -> String {
        unsafe {
            let ptr = cudaGetErrorString(self.0);
            if ptr.is_null() {
                return format!("Unknown CudaErrorType: {}", self.0);
            }

            CStr::from_ptr(ptr).to_str().unwrap_or("Invalid UTF-8").to_string()
        }
    }

    pub fn error_name(&self) -> String {
        unsafe {
            let ptr = cudaGetErrorName(self.0);
            if ptr.is_null() {
                return format!("UNKNOWN_ERROR_{}", self.0);
            }

            CStr::from_ptr(ptr).to_str().unwrap_or("INVALID_UTF8").to_string()
        }
    }
}

impl fmt::Display for CudaRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.error_name(), self.error_string())
    }
}

impl std::error::Error for CudaRuntimeError {}

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("{context}: {error}")]
    Nvrtc { error: NvrtcError, context: String },

    #[error("{context}: {error}")]
    Driver { error: CudaDriverError, context: String },

    #[error("{context}: {error}")]
    Runtime { error: CudaRuntimeError, context: String },

    #[error("{context}: {error}")]
    Kernel { error: KernelError, context: String },

    #[error("Error: {0}")]
    Other(String),
}

impl CudaError {
    pub fn nvrtc_with_context(error: NvrtcResult, context: impl Into<String>) -> Self {
        CudaError::Nvrtc { error: NvrtcError(error), context: context.into() }
    }

    pub fn driver_with_context(error: CUresult, context: impl Into<String>) -> Self {
        CudaError::Driver { error: CudaDriverError(error), context: context.into() }
    }

    pub fn runtime_with_context(error: CudaErrorType, context: impl Into<String>) -> Self {
        CudaError::Runtime { error: CudaRuntimeError(error), context: context.into() }
    }

    pub fn kernel_with_context(error: KernelErrorCode, context: impl Into<String>) -> Self {
        CudaError::Kernel { error: KernelError(error), context: context.into() }
    }
}

pub type CudaResult<T> = Result<T, CudaError>;

pub trait CudaResultExt {
    fn check_with_context(self, context: &str) -> CudaResult<()>;
}

impl CudaResultExt for NvrtcResult {
    fn check_with_context(self, context: &str) -> CudaResult<()> {
        if self == CUDA_SUCCESS as u16 {
            Ok(())
        } else {
            Err(CudaError::nvrtc_with_context(self, context))
        }
    }
}

impl CudaResultExt for CUresult {
    fn check_with_context(self, context: &str) -> CudaResult<()> {
        if self == CUDA_SUCCESS as u32 {
            Ok(())
        } else {
            Err(CudaError::driver_with_context(self, context))
        }
    }
}

impl CudaResultExt for CudaErrorType {
    fn check_with_context(self, context: &str) -> CudaResult<()> {
        if self == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::runtime_with_context(self, context))
        }
    }
}

impl CudaResultExt for KernelErrorCode {
    fn check_with_context(self, context: &str) -> CudaResult<()> {
        if self == CUDA_SUCCESS as i16 {
            Ok(())
        } else {
            Err(CudaError::kernel_with_context(self, context))
        }
    }
}
