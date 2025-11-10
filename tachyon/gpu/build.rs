/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

fn main() {
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
}
