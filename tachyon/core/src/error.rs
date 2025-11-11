/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorMode {
    Ansi,
    Tachyon,
}

#[repr(C)]
#[derive(Debug, thiserror::Error)]
pub enum MathError {
    #[error("Add Overflow")]
    AddOverflow,
}
