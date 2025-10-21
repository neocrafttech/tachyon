use crate::nvrtc_wrapper::compile_cuda_file_to_fatbin;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum KernelCacheError {
    #[error("NVRTC compile failed: {0}")]
    Nvrtc(#[from] crate::nvrtc_wrapper::NvrtcError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Compute a SHA256 hash of the kernel source to detect changes.
fn compute_source_hash<P: AsRef<Path>>(path: P) -> Result<String, KernelCacheError> {
    let src = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&src);
    Ok(format!("{:x}", hasher.finalize()))
}

fn get_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("NVRTC_CACHE_DIR") {
        return PathBuf::from(dir);
    }

    if let Ok(home) = env::var("HOME") {
        return PathBuf::from(home).join(".nvrtc_cache");
    }

    //Window case
    if let Ok(home) = env::var("USERPROFILE") {
        return PathBuf::from(home).join(".nvrtc_cache");
    }

    //Current directory as fallback
    PathBuf::from(".nvrtc_cache")
}

/// Given a CUDA file and target arch, either load from cache or compile anew.
pub fn load_or_compile_kernel<P: AsRef<Path>>(
    path: P,
    arch: &str,
) -> Result<PathBuf, KernelCacheError> {
    let path = path.as_ref();
    let hash = compute_source_hash(path)?;

    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    let fatbin_name = format!(
        "{}_{}_{}.fatbin",
        path.file_stem().unwrap().to_string_lossy(),
        arch,
        &hash[..16] // shorten hash
    );
    let fatbin_path = cache_dir.join(fatbin_name);

    if fatbin_path.exists() {
        println!("Loaded cached fatbin: {}", fatbin_path.display());
        return Ok(fatbin_path);
    }

    println!("Compiling kernel {} for {}", path.display(), arch);

    let fatbin = compile_cuda_file_to_fatbin(path, arch)?;
    println!("fatbin {:?}", fatbin_path);
    let mut file = fs::File::create(&fatbin_path)?;
    file.write_all(fatbin.as_bytes())?;
    println!("Saved fatbin to {}", fatbin_path.display());

    Ok(fatbin_path)
}
