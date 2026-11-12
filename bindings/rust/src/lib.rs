//! Rust bindings to montlok, the CPU runtime for quantitative trading models.
//!
//! The same C++ core as the Python package (`import montlok`) and the C++
//! library: model families `sequential` (linear, MLP, GRU, LSTM),
//! `tree_ensemble` (LightGBM, XGBoost, scikit-learn) and plugin families, plus
//! the streaming microstructure feature engine. Model files are written by
//! `montlok.export` in Python.
//!
//! ```no_run
//! let model = montlok::Model::open("gru.montlok")?;
//! let info = model.info();
//! let window = vec![0.0; info.input_len()];
//! let prediction = model.predict(&window, 1)?;
//! let mut live = model.stream(1)?;
//! let now = live.step(&vec![0.0; info.features])?;
//! # Ok::<(), montlok::Error>(())
//! ```

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

/// Raw declarations of include/montlok/montlok.h.
pub mod sys {
    use std::ffi::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct montlok_model {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct montlok_stream {
        _private: [u8; 0],
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct montlok_model_info {
        pub features: u32,
        pub sequence: u32,
        pub outputs: u32,
        pub flags: u32,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy, PartialEq)]
    pub struct montlok_micro_event {
        pub receive_ns: i64,
        pub kind: i32,
        pub side: i32,
        pub a: f64,
        pub b: f64,
        pub c: f64,
        pub d: f64,
    }

    pub const MONTLOK_FLOAT32: c_int = 1;
    pub const MONTLOK_FLOAT64: c_int = 2;
    pub const MONTLOK_MODEL_STREAMABLE: u32 = 1;
    pub const MONTLOK_MODEL_PREPROCESS: u32 = 2;

    extern "C" {
        pub fn montlok_abi_version() -> u32;
        pub fn montlok_version() -> *const c_char;
        pub fn montlok_last_error() -> *const c_char;
        pub fn montlok_set_threads(threads: c_int);
        pub fn montlok_threads() -> c_int;
        pub fn montlok_set_latency_threads(threads: c_int) -> c_int;
        pub fn montlok_latency_threads() -> c_int;
        pub fn montlok_families() -> *const c_char;
        pub fn montlok_model_open(path: *const c_char, model: *mut *mut montlok_model) -> c_int;
        pub fn montlok_model_open_memory(data: *const c_void, size: usize, model: *mut *mut montlok_model) -> c_int;
        pub fn montlok_model_close(model: *mut montlok_model);
        pub fn montlok_model_get_info(model: *const montlok_model, info: *mut montlok_model_info) -> c_int;
        pub fn montlok_model_family(model: *const montlok_model) -> *const c_char;
        pub fn montlok_model_spec(model: *const montlok_model) -> *const c_char;
        pub fn montlok_model_predict(
            model: *const montlok_model,
            dtype: c_int,
            x: *const c_void,
            batch: usize,
            out: *mut f64,
        ) -> c_int;
        pub fn montlok_stream_open(model: *const montlok_model, batch: usize, stream: *mut *mut montlok_stream)
            -> c_int;
        pub fn montlok_stream_step(stream: *mut montlok_stream, dtype: c_int, x: *const c_void, out: *mut f64)
            -> c_int;
        pub fn montlok_stream_reset(stream: *mut montlok_stream) -> c_int;
        pub fn montlok_stream_close(stream: *mut montlok_stream);
        pub fn montlok_plugin_load(path: *const c_char) -> c_int;
        pub fn montlok_micro_schema() -> *const c_char;
        pub fn montlok_micro_event_size() -> usize;
        pub fn montlok_micro_create(capacity: usize) -> *mut c_void;
        pub fn montlok_micro_destroy(engine: *mut c_void);
        pub fn montlok_micro_reset(engine: *mut c_void);
        pub fn montlok_micro_update(engine: *mut c_void, event: *const montlok_micro_event) -> c_int;
        pub fn montlok_micro_snapshot(engine: *mut c_void, out: *mut f64, count: usize) -> c_int;
    }
}

/// A failed call: the montlok status code and its message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub status: i32,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "montlok error {}: {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// MONTLOK_ERROR_INVALID_ARGUMENT
pub const INVALID_ARGUMENT: i32 = 1;
/// MONTLOK_ERROR_SHAPE
pub const SHAPE: i32 = 5;

fn text(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

fn check(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error { status, message: text(unsafe { sys::montlok_last_error() }) })
    }
}

fn path_arg(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| Error { status: INVALID_ARGUMENT, message: "path contains a NUL byte".into() })
}

pub fn version() -> String {
    text(unsafe { sys::montlok_version() })
}

/// Worker threads for large batches; 0 selects one per hardware thread.
pub fn set_threads(threads: usize) {
    unsafe { sys::montlok_set_threads(threads.min(c_int::MAX as usize) as c_int) }
}

/// Threads that compute a single prediction together with the calling thread:
/// single rows of a tree ensemble divide their trees among the calling thread
/// and up to `threads` more. Each spins on its core while it exists; 0 stops
/// them.
pub fn set_latency_threads(threads: usize) -> Result<()> {
    check(unsafe { sys::montlok_set_latency_threads(threads.min(c_int::MAX as usize) as c_int) })
}

pub fn latency_threads() -> usize {
    unsafe { sys::montlok_latency_threads() }.max(0) as usize
}

pub fn families() -> Vec<String> {
    text(unsafe { sys::montlok_families() }).split(',').filter(|s| !s.is_empty()).map(str::to_owned).collect()
}

/// Loads a plugin library and registers its model families for the process.
pub fn load_plugin(path: impl AsRef<Path>) -> Result<()> {
    let path = path_arg(path.as_ref())?;
    check(unsafe { sys::montlok_plugin_load(path.as_ptr()) })
}

/// Input and output dimensions of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    pub features: usize,
    /// Time steps per sample; 0 for tabular models.
    pub sequence: usize,
    pub outputs: usize,
    pub streamable: bool,
    pub preprocess: bool,
}

impl ModelInfo {
    /// Values per sample: max(sequence, 1) * features.
    pub fn input_len(&self) -> usize {
        self.sequence.max(1) * self.features
    }
}

/// Input element types accepted by predict and step.
pub trait Element: Copy {
    #[doc(hidden)]
    const DTYPE: c_int;
}

impl Element for f32 {
    const DTYPE: c_int = sys::MONTLOK_FLOAT32;
}

impl Element for f64 {
    const DTYPE: c_int = sys::MONTLOK_FLOAT64;
}

/// A loaded model. Immutable and safe to share between threads.
pub struct Model {
    raw: NonNull<sys::montlok_model>,
    info: ModelInfo,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_arg(path.as_ref())?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::montlok_model_open(path.as_ptr(), &mut raw) })?;
        Self::wrap(raw)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::montlok_model_open_memory(bytes.as_ptr().cast::<c_void>(), bytes.len(), &mut raw) })?;
        Self::wrap(raw)
    }

    fn wrap(raw: *mut sys::montlok_model) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or(Error { status: 7, message: "null model".into() })?;
        let mut info = sys::montlok_model_info::default();
        let status = unsafe { sys::montlok_model_get_info(raw.as_ptr(), &mut info) };
        let model = Model {
            raw,
            info: ModelInfo {
                features: info.features as usize,
                sequence: info.sequence as usize,
                outputs: info.outputs as usize,
                streamable: info.flags & sys::MONTLOK_MODEL_STREAMABLE != 0,
                preprocess: info.flags & sys::MONTLOK_MODEL_PREPROCESS != 0,
            },
        };
        check(status)?;
        Ok(model)
    }

    pub fn info(&self) -> ModelInfo {
        self.info
    }

    pub fn family(&self) -> String {
        text(unsafe { sys::montlok_model_family(self.raw.as_ptr()) })
    }

    /// The model spec (schema montlok.model/1) as JSON text.
    pub fn spec_json(&self) -> String {
        text(unsafe { sys::montlok_model_spec(self.raw.as_ptr()) })
    }

    /// `x` holds `batch * info().input_len()` values; returns `batch * outputs`.
    pub fn predict<T: Element>(&self, x: &[T], batch: usize) -> Result<Vec<f64>> {
        let mut out = vec![0.0; batch * self.info.outputs];
        self.predict_into(x, batch, &mut out)?;
        Ok(out)
    }

    pub fn predict_into<T: Element>(&self, x: &[T], batch: usize, out: &mut [f64]) -> Result<()> {
        if x.len() != batch * self.info.input_len() || out.len() != batch * self.info.outputs {
            return Err(Error { status: SHAPE, message: "input or output length does not match batch".into() });
        }
        check(unsafe {
            sys::montlok_model_predict(self.raw.as_ptr(), T::DTYPE, x.as_ptr().cast::<c_void>(), batch, out.as_mut_ptr())
        })
    }

    /// `batch` independent recurrent states starting at zero.
    pub fn stream(&self, batch: usize) -> Result<Stream> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::montlok_stream_open(self.raw.as_ptr(), batch, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or(Error { status: 7, message: "null stream".into() })?;
        Ok(Stream { raw, batch, info: self.info })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::montlok_model_close(self.raw.as_ptr()) }
    }
}

/// Recurrent state carried from one time step to the next. It keeps its
/// model alive on its own, so it may outlive the `Model` value.
pub struct Stream {
    raw: NonNull<sys::montlok_stream>,
    batch: usize,
    info: ModelInfo,
}

unsafe impl Send for Stream {}

impl Stream {
    /// One time step: `x` holds `batch * features` values; returns `batch * outputs`.
    pub fn step<T: Element>(&mut self, x: &[T]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.batch * self.info.outputs];
        self.step_into(x, &mut out)?;
        Ok(out)
    }

    pub fn step_into<T: Element>(&mut self, x: &[T], out: &mut [f64]) -> Result<()> {
        if x.len() != self.batch * self.info.features || out.len() != self.batch * self.info.outputs {
            return Err(Error { status: SHAPE, message: "input or output length does not match batch".into() });
        }
        check(unsafe {
            sys::montlok_stream_step(self.raw.as_ptr(), T::DTYPE, x.as_ptr().cast::<c_void>(), out.as_mut_ptr())
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        check(unsafe { sys::montlok_stream_reset(self.raw.as_ptr()) })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::montlok_stream_close(self.raw.as_ptr()) }
    }
}

/// Streaming L1/trade microstructure features (schema xstock-observable-l1-v2-32).
pub mod micro {
    use super::{sys, text, Error, Result};
    use std::ffi::c_void;
    use std::ptr::NonNull;

    pub use super::sys::montlok_micro_event as Event;

    pub const FEATURES: usize = 32;
    pub const REFERENCE: i32 = 0;
    pub const BOOK: i32 = 1;
    pub const TRADE: i32 = 2;

    pub fn schema() -> String {
        text(unsafe { sys::montlok_micro_schema() })
    }

    pub struct FeatureEngine {
        raw: NonNull<c_void>,
    }

    unsafe impl Send for FeatureEngine {}

    impl FeatureEngine {
        pub fn new(capacity: usize) -> Result<Self> {
            let raw = unsafe { sys::montlok_micro_create(capacity) };
            NonNull::new(raw)
                .map(|raw| FeatureEngine { raw })
                .ok_or(Error { status: 1, message: "capacity must be at least 2".into() })
        }

        /// A rejected event (non-monotonic time, non-finite values, overflow)
        /// leaves the engine faulted until `reset`.
        pub fn update(&mut self, event: &Event) -> Result<()> {
            match unsafe { sys::montlok_micro_update(self.raw.as_ptr(), event) } {
                0 => Ok(()),
                _ => Err(Error { status: 1, message: "event rejected; reset required".into() }),
            }
        }

        /// The 32 features, or None while warming up or after a fault.
        pub fn snapshot(&mut self) -> Option<[f64; FEATURES]> {
            let mut out = [0.0; FEATURES];
            match unsafe { sys::montlok_micro_snapshot(self.raw.as_ptr(), out.as_mut_ptr(), FEATURES) } {
                1 => Some(out),
                _ => None,
            }
        }

        pub fn reset(&mut self) {
            unsafe { sys::montlok_micro_reset(self.raw.as_ptr()) }
        }
    }

    impl Drop for FeatureEngine {
        fn drop(&mut self) {
            unsafe { sys::montlok_micro_destroy(self.raw.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod layout {
    #[test]
    fn micro_event_matches_the_c_layout() {
        assert_eq!(std::mem::size_of::<super::sys::montlok_micro_event>(), unsafe {
            super::sys::montlok_micro_event_size()
        });
        assert_eq!(unsafe { super::sys::montlok_abi_version() }, 1);
    }
}
