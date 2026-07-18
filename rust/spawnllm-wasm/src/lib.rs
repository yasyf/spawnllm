use std::alloc::{Layout, alloc, dealloc};
use std::slice;

fn layout(len: usize) -> Layout {
    Layout::from_size_align(len.max(1), 1).unwrap()
}

#[unsafe(no_mangle)]
pub extern "C" fn sl_alloc(len: u32) -> u32 {
    unsafe { alloc(layout(len as usize)) as u32 }
}

/// # Safety
///
/// `ptr` must be a live buffer from [`sl_alloc`] (or unpacked from an
/// [`sl_call`] response) and `len` the exact length it was allocated with;
/// any other pair, or a second free, is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_free(ptr: u32, len: u32) {
    unsafe { dealloc(ptr as *mut u8, layout(len as usize)) };
}

/// Dispatches one request and returns the response buffer packed as
/// `(ptr << 32) | len`.
///
/// # Safety
///
/// `ptr` must point at `len` initialized bytes of valid UTF-8 in the module's
/// linear memory (the host always sends JSON; anything else traps), and the
/// returned buffer must be released with [`sl_free`] using the unpacked
/// pointer and length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_call(ptr: u32, len: u32) -> u64 {
    let request = unsafe { slice::from_raw_parts(ptr as *const u8, len as usize) };
    let response = spawnllm_core::dispatch(std::str::from_utf8(request).unwrap());
    let bytes = response.as_bytes();
    let out = sl_alloc(bytes.len() as u32);
    unsafe { slice::from_raw_parts_mut(out as *mut u8, bytes.len()) }.copy_from_slice(bytes);
    ((out as u64) << 32) | bytes.len() as u64
}
