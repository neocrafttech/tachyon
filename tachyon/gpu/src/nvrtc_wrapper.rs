use crate::ffi::nvrtc::*;
use std::ffi::CString;
use std::ffi::c_char;
use std::fs;
use std::io;
use std::path::Path;
use std::string::FromUtf8Error;

#[derive(thiserror::Error, Debug)]
pub enum NvrtcError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("UTF8 conversion error: {0}")]
    Utf8(#[from] FromUtf8Error),

    #[error("NVRTC compilation error: {0}")]
    Nvrtc(String),
}

pub fn compile_cuda_file_to_fatbin<P: AsRef<Path>>(
    path: P,
    arch: &str,
) -> Result<String, NvrtcError> {
    let src = fs::read_to_string(path.as_ref())?;
    let src_c = CString::new(src).unwrap();
    let name_c = CString::new(
        path.as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("kernel.cu"),
    )
    .unwrap();

    let arch_flag = format!("--gpu-architecture={}", arch);
    let opts = vec![
        CString::new(arch_flag).unwrap(),
        CString::new("--fatbin").unwrap(),
        CString::new("--use_fast_math").unwrap(),
    ];
    let opt_ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();

    unsafe {
        let mut prog: NvrtcProgram = std::ptr::null_mut();
        nvrtcCreateProgram(
            &mut prog,
            src_c.as_ptr(),
            name_c.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );

        let res = nvrtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
        if res != NvrtcResult::NvrtcSuccess {
            return Err(NvrtcError::Nvrtc("compile failed".into()));
        }

        let mut size: usize = 0;
        nvrtcGetCUBINSize(prog, &mut size);
        let mut buf = vec![0u8; size];
        nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut c_char);
        nvrtcDestroyProgram(&mut prog);

        Ok(String::from_utf8(buf)?)
    }
}
