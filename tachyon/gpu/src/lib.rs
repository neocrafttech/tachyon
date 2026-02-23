//! GPU runtime and CUDA FFI helpers for Tachyon compute execution.

/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

/// Public GPU column facade.
pub mod column;
mod cuda_launcher;
/// GPU-facing error types.
pub mod error;
/// Low-level CUDA and column FFI bindings.
mod ffi;
pub(crate) mod kernel_cache; //TODO Hide it

pub use cuda_launcher::{launch, launch_aggregate};
