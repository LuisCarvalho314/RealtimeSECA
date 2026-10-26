//! Requested allocator bytes, excluding pre-existing input allocation; separate untimed pass.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
pub struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            record(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        System.dealloc(p, layout);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = System.realloc(p, layout, size);
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            record(size);
        }
        new
    }
}
fn record(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
    PEAK.fetch_max(LIVE.fetch_add(size, Relaxed) + size, Relaxed);
}
pub fn start() -> usize {
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    before
}
pub fn finish(before: usize) -> serde_json::Value {
    let allocations = CALLS.load(Relaxed);
    let requested_bytes = BYTES.load(Relaxed);
    let peak_additional_live_bytes = PEAK.load(Relaxed).saturating_sub(before);
    serde_json::json!({"allocations":allocations,"requested_bytes":requested_bytes,
        "peak_additional_live_bytes":peak_additional_live_bytes})
}
