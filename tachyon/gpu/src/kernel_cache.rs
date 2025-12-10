/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::ffi::cuda_error::{CudaError, CudaResult};

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("I/O error: {0}")]
    IoError(io::Error),
}

impl From<io::Error> for CacheError {
    fn from(err: io::Error) -> Self {
        CacheError::IoError(err)
    }
}

static KERNEL_CACHE: OnceLock<Mutex<KernelCache>> = OnceLock::new();

pub struct KernelCache {
    cache_dir: PathBuf,
    memory_cache: HashMap<String, Vec<u8>>,
}

impl KernelCache {
    pub fn global() -> &'static Mutex<KernelCache> {
        KERNEL_CACHE.get_or_init(|| {
            let cache =
                KernelCache::new(Self::default_dir()).expect("Failed to initialize kernel cache");
            Mutex::new(cache)
        })
    }

    fn new<P: AsRef<Path>>(cache_dir: P) -> io::Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }

        Ok(Self { cache_dir, memory_cache: HashMap::new() })
    }

    fn default_dir() -> PathBuf {
        let mut path = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"));
        path.push("tachyon");
        path.push("kernels");
        path
    }

    fn cache_path(&self, kernel_name: &str) -> PathBuf {
        let safe_name = kernel_name.replace(['/', '\\', ':'], "_");
        self.cache_dir.join(format!("{}.bin", safe_name))
    }

    fn exists(&self, kernel_name: &str) -> bool {
        self.memory_cache.contains_key(kernel_name) || self.cache_path(kernel_name).exists()
    }

    fn load(&mut self, kernel_name: &str) -> Result<&[u8], CacheError> {
        if !self.memory_cache.contains_key(kernel_name) {
            let path = self.cache_path(kernel_name);
            let mut file = fs::File::open(&path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            self.memory_cache.insert(kernel_name.to_string(), buffer);
        }

        Ok(&self.memory_cache[kernel_name])
    }

    fn save(&mut self, kernel_name: &str, kernel_data: &[u8]) -> Result<(), CacheError> {
        let path = self.cache_path(kernel_name);
        let mut file = fs::File::create(&path)?;
        file.write_all(kernel_data)?;

        self.memory_cache.insert(kernel_name.to_string(), kernel_data.to_vec());

        Ok(())
    }

    pub fn get_or_compile<F>(&mut self, kernel_name: &str, compile_fn: F) -> CudaResult<&[u8]>
    where
        F: FnOnce() -> CudaResult<Vec<u8>>,
    {
        if self.exists(kernel_name) {
            return self
                .load(kernel_name)
                .map_err(|e| CudaError::Other(format!("Failed to load kernel: {:?}", e)));
        }

        println!("Compiling kernel: {}", kernel_name);
        let kernel_data = compile_fn()?;

        self.save(kernel_name, &kernel_data)
            .map_err(|e| CudaError::Other(format!("Failed to save kernel: {:?}", e)))?;

        self.load(kernel_name)
            .map_err(|e| CudaError::Other(format!("Failed to load kernel: {:?}", e)))
    }

    fn clear_memory_cache(&mut self) {
        self.memory_cache.clear();
    }

    fn clear_disk_cache(&self) -> io::Result<()> {
        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("bin") {
                    fs::remove_file(entry.path())?;
                }
            }
        }
        Ok(())
    }

    pub fn clear_all(&mut self) -> io::Result<()> {
        self.clear_memory_cache();
        self.clear_disk_cache()
    }
}

pub fn get_or_compile_kernel<F>(kernel_name: &str, compile_fn: F) -> CudaResult<Vec<u8>>
where
    F: FnOnce() -> CudaResult<Vec<u8>>,
{
    let mut cache = KernelCache::global().lock().unwrap();
    let data = cache.get_or_compile(kernel_name, compile_fn)?;
    Ok(data.to_vec())
}

#[allow(dead_code)]
pub fn clear_kernel_cache() -> io::Result<()> {
    let mut cache = KernelCache::global().lock().unwrap();
    cache.clear_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_cache() {
        let temp_dir = std::env::temp_dir().join("test_kernel_cache");
        let mut cache = KernelCache::new(&temp_dir).unwrap();

        let mut compile_count = 0;

        let kernel1 = cache
            .get_or_compile("test_kernel", || {
                compile_count += 1;
                Ok(vec![1, 2, 3, 4])
            })
            .unwrap();
        assert_eq!(compile_count, 1);
        assert_eq!(kernel1, vec![1, 2, 3, 4]);

        let kernel2 = cache
            .get_or_compile("test_kernel", || {
                compile_count += 1;
                Ok(vec![5, 6, 7, 8])
            })
            .unwrap();
        assert_eq!(compile_count, 1); // Still 1, not compiled again
        assert_eq!(kernel2, vec![1, 2, 3, 4]); // Same as first

        cache.clear_all().unwrap();
        fs::remove_dir(&temp_dir).ok();
    }
}
