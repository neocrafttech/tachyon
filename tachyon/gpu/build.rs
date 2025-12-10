/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

fn main() {
    let header_dir = "src/ffi/kernels";
    let cpp_file = format!("{}/error.cpp", header_dir);
    let header_file = format!("{}/error.h", header_dir);

    println!("cargo:rerun-if-changed={}", cpp_file);
    println!("cargo:rerun-if-changed={}", header_file);
    println!("cargo:rerun-if-changed=build.rs");

    cc::Build::new()
        .file(cpp_file)
        .cpp(true)
        .flag_if_supported("-std=c++20")
        .compile("tachyon_kernel_lib");

    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-env=LIB_ROOT={}", std::env::var("CARGO_MANIFEST_DIR").unwrap());
}
