fn main() {
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
}
