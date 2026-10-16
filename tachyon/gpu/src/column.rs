/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

/// Compact string descriptor stored in the main column stream.
///
/// Memory layout (16 bytes total):
/// - 4 bytes: string byte length
/// - 4 bytes: prefix (first 4 bytes)
/// - 8 bytes: inline continuation bytes or offset into string buffer stream
pub const STRING_INLINE_PREFIX_BYTES: usize = 4;
pub const STRING_INLINE_DATA_BYTES: usize = 8;
pub const STRING_INLINE_TOTAL_BYTES: usize = STRING_INLINE_PREFIX_BYTES + STRING_INLINE_DATA_BYTES;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StringView {
    pub size: u32,
    pub prefix: u32,
    pub data: u64,
}

#[cfg(feature = "gpu")]
pub use crate::ffi::column::Column;
