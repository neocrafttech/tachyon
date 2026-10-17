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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StringView {
    pub size: u32,
    pub prefix: u32,
    pub data: u64,
}

mod sealed {
    pub trait Element {}
    pub trait Bitmap {}
}

/// A supported GPU value with no padding or ownership and a checked byte representation.
///
/// This trait is sealed: arbitrary Rust types cannot be transferred as column values.
///
/// ```compile_fail
/// use gpu::column::GpuElement;
/// fn require_element<T: GpuElement>() {}
/// require_element::<String>();
/// ```
pub trait GpuElement: sealed::Element + bytemuck::NoUninit + bytemuck::CheckedBitPattern {}

macro_rules! elements {
    ($($ty:ty),* $(,)?) => { $(
        impl sealed::Element for $ty {}
        impl GpuElement for $ty {}
    )* };
}

elements!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, half::f16, half::bf16, bool, StringView);

/// An unsigned integer block supported by GPU validity bitmaps.
///
/// ```compile_fail
/// use gpu::column::BitmapElement;
/// fn require_bitmap<T: BitmapElement>() {}
/// require_bitmap::<bool>();
/// ```
pub trait BitmapElement: GpuElement + sealed::Bitmap {}

macro_rules! bitmap_elements {
    ($($ty:ty),* $(,)?) => { $(
        impl sealed::Bitmap for $ty {}
        impl BitmapElement for $ty {}
    )* };
}

bitmap_elements!(u8, u16, u32, u64);

#[cfg(feature = "gpu")]
pub use crate::ffi::column::Column;
