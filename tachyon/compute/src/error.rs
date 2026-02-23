/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#[derive(Clone, Copy, Debug, PartialEq)]
/// Error semantics used during expression evaluation.
pub enum ErrorMode {
    /// ANSI-like semantics (for example, overflow returns an error).
    Ansi,
    /// Tachyon-native semantics.
    Tachyon,
}

#[repr(C)]
#[derive(Debug, thiserror::Error)]
/// Arithmetic/runtime error emitted by generated kernels.
pub enum MathError {
    #[error("Add Overflow")]
    AddOverflow,
}
