use std::alloc::{Layout, alloc, dealloc};
use std::slice;

fn layout(len: usize) -> Layout {
    Layout::from_size_align(len.max(1), 1).unwrap()
}

#[unsafe(no_mangle)]
pub extern "C" fn sl_alloc(len: u32) -> u32 {
    unsafe { alloc(layout(len as usize)) as u32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_free(ptr: u32, len: u32) {
    unsafe { dealloc(ptr as *mut u8, layout(len as usize)) };
}

// Returns the response buffer as `(ptr << 32) | len` in one u64; host frees it via `sl_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_call(ptr: u32, len: u32) -> u64 {
    let request = unsafe { slice::from_raw_parts(ptr as *const u8, len as usize) };
    let response = spawnllm_core::dispatch(std::str::from_utf8(request).unwrap());
    let bytes = response.as_bytes();
    let out = sl_alloc(bytes.len() as u32);
    unsafe { slice::from_raw_parts_mut(out as *mut u8, bytes.len()) }.copy_from_slice(bytes);
    ((out as u64) << 32) | bytes.len() as u64
}
