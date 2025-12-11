/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use thiserror::Error;

use crate::ffi::cuda_error::CudaError;
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Math error: {0}")]
    Math(String),
}

pub type GpuResult<T> = Result<T, GpuError>;
