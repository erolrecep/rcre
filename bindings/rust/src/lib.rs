//! Safe Rust FFI Bindings for RCRE (Recep's Code Run Everywhere)
//! C23 High-Performance Computer Vision and Machine Learning Engine

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uchar, c_uint};

#[repr(C)]
#[derive(Debug)]
pub struct RCRE_Arena {
    pub buffer: *mut u8,
    pub capacity: usize,
    pub offset: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RCRE_Tensor {
    pub data: *mut std::ffi::c_void,
    pub shape: [usize; 4],
    pub strides: [isize; 4],
    pub ndim: u8,
    pub dtype: u8,
    pub layout: u8,
}

extern "C" {
    pub fn rcre_arena_create(capacity: usize) -> *mut RCRE_Arena;
    pub fn rcre_arena_reset(arena: *mut RCRE_Arena);
    pub fn rcre_arena_destroy(arena: *mut RCRE_Arena);

    pub fn rcre_tensor_create(
        shape: *const usize,
        ndim: u8,
        dtype: u8,
        layout: u8,
        arena: *mut RCRE_Arena,
    ) -> RCRE_Tensor;

    pub fn rcre_cv_rgb_to_gray(src: *const RCRE_Tensor, dst: *mut RCRE_Tensor) -> c_int;
    pub fn rcre_hal_init();
    pub fn rcre_cpu_get_brand() -> *const c_char;
}

/// Safe RAII wrapper around RCRE linear scratchpad arena.
pub struct Arena {
    raw: *mut RCRE_Arena,
}

impl Arena {
    pub fn new(capacity_bytes: usize) -> Result<Self, &'static str> {
        let ptr = unsafe { rcre_arena_create(capacity_bytes) };
        if ptr.is_null() {
            Err("Failed to create RCRE_Arena")
        } else {
            Ok(Self { raw: ptr })
        }
    }

    pub fn reset(&mut self) {
        unsafe { rcre_arena_reset(self.raw) };
    }

    pub fn capacity(&self) -> usize {
        unsafe { (*self.raw).capacity }
    }

    pub fn offset(&self) -> usize {
        unsafe { (*self.raw).offset }
    }

    pub fn as_raw(&mut self) -> *mut RCRE_Arena {
        self.raw
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { rcre_arena_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Returns the detected host CPU brand string.
pub fn cpu_brand() -> &'static str {
    unsafe {
        let ptr = rcre_cpu_get_brand();
        if ptr.is_null() {
            "Unknown"
        } else {
            CStr::from_ptr(ptr).to_str().unwrap_or("Unknown")
        }
    }
}
