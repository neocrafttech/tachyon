/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;

pub trait CudaType: Copy + Debug + Default + 'static {
    fn cuda_type_name() -> &'static str;
}

impl CudaType for i8 {
    fn cuda_type_name() -> &'static str {
        "int8_t"
    }
}

impl CudaType for u8 {
    fn cuda_type_name() -> &'static str {
        "uint8_t"
    }
}

impl CudaType for i16 {
    fn cuda_type_name() -> &'static str {
        "int16_t"
    }
}

impl CudaType for u16 {
    fn cuda_type_name() -> &'static str {
        "uint16_t"
    }
}

impl CudaType for i32 {
    fn cuda_type_name() -> &'static str {
        "int32_t"
    }
}

impl CudaType for u32 {
    fn cuda_type_name() -> &'static str {
        "uint32_t"
    }
}

impl CudaType for i64 {
    fn cuda_type_name() -> &'static str {
        "int64_t"
    }
}

impl CudaType for u64 {
    fn cuda_type_name() -> &'static str {
        "uint64_t"
    }
}

impl CudaType for f32 {
    fn cuda_type_name() -> &'static str {
        "float"
    }
}

impl CudaType for f64 {
    fn cuda_type_name() -> &'static str {
        "double"
    }
}
