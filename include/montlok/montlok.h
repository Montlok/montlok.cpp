/*
 * montlok.h - stable C ABI of the montlok CPU runtime.
 *
 * Copyright 2026 Montlok. Apache-2.0.
 *
 * Used by the Python package (montlok._core), the Rust crate (bindings/rust),
 * C++ clients through the RAII wrapper in montlok.hpp, and plugin model
 * families.
 *
 * Models are safetensors files whose __metadata__["montlok"] entry holds the
 * model spec (schema "montlok.model/1"). A loaded model is immutable:
 * montlok_model_predict may be called from any number of threads at once. A
 * stream carries recurrent state between calls and belongs to one thread at a
 * time.
 *
 * Every function that can fail returns a montlok_status; the message of the
 * last failure on the calling thread is montlok_last_error().
 */
#ifndef MONTLOK_H
#define MONTLOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(MONTLOK_SHARED)
#if defined(MONTLOK_BUILDING)
#define MONTLOK_API __declspec(dllexport)
#else
#define MONTLOK_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define MONTLOK_API __attribute__((visibility("default")))
#else
#define MONTLOK_API
#endif

#define MONTLOK_ABI_VERSION 1u

typedef enum montlok_status {
    MONTLOK_OK = 0,
    MONTLOK_ERROR_INVALID_ARGUMENT = 1,
    MONTLOK_ERROR_IO = 2,
    MONTLOK_ERROR_FORMAT = 3,
    MONTLOK_ERROR_UNSUPPORTED = 4,
    MONTLOK_ERROR_SHAPE = 5,
    MONTLOK_ERROR_STATE = 6,
    MONTLOK_ERROR_INTERNAL = 7
} montlok_status;

typedef enum montlok_dtype {
    MONTLOK_FLOAT32 = 1,
    MONTLOK_FLOAT64 = 2
} montlok_dtype;

typedef struct montlok_model montlok_model;
typedef struct montlok_stream montlok_stream;

/* The model accepts the stream of feature vectors as soon as each one is
 * complete (recurrent families). */
#define MONTLOK_MODEL_STREAMABLE 1u
/* The model standardizes and clips raw inputs itself (float64). */
#define MONTLOK_MODEL_PREPROCESS 2u

typedef struct montlok_model_info {
    uint32_t features; /* values per time step */
    uint32_t sequence; /* time steps per sample; 0 for tabular models */
    uint32_t outputs;  /* values per sample */
    uint32_t flags;    /* MONTLOK_MODEL_* */
} montlok_model_info;

MONTLOK_API uint32_t montlok_abi_version(void);
MONTLOK_API const char* montlok_version(void);
MONTLOK_API const char* montlok_last_error(void);

/* Worker threads for large batches; 0 selects one per hardware thread. */
MONTLOK_API void montlok_set_threads(int threads);
MONTLOK_API int montlok_threads(void);

/* Latency threads compute a single prediction together with the calling
 * thread: single rows of a tree ensemble divide their trees among the calling
 * thread and up to `threads` more. Each latency thread spins on its core
 * while it exists; 0, the default, stops them. */
MONTLOK_API int montlok_set_latency_threads(int threads);
MONTLOK_API int montlok_latency_threads(void);

/* Comma-separated names of the registered model families. The pointer stays
 * valid until the next call of this function on the same thread. */
MONTLOK_API const char* montlok_families(void);

MONTLOK_API int montlok_model_open(const char* path, montlok_model** model);
MONTLOK_API int montlok_model_open_memory(const void* data, size_t size, montlok_model** model);
MONTLOK_API void montlok_model_close(montlok_model* model);
MONTLOK_API int montlok_model_get_info(const montlok_model* model, montlok_model_info* info);
/* Both strings live as long as the model. */
MONTLOK_API const char* montlok_model_family(const montlok_model* model);
MONTLOK_API const char* montlok_model_spec(const montlok_model* model);

/* x holds batch * max(sequence, 1) * features values of `dtype`, row-major;
 * out receives batch * outputs doubles. */
MONTLOK_API int montlok_model_predict(const montlok_model* model, montlok_dtype dtype, const void* x, size_t batch,
                                      double* out);

/* A stream of `batch` independent recurrent states, all starting at zero.
 * step consumes one time step per state: x holds batch * features values and
 * out receives batch * outputs doubles. */
MONTLOK_API int montlok_stream_open(const montlok_model* model, size_t batch, montlok_stream** stream);
MONTLOK_API int montlok_stream_step(montlok_stream* stream, montlok_dtype dtype, const void* x, double* out);
MONTLOK_API int montlok_stream_reset(montlok_stream* stream);
MONTLOK_API void montlok_stream_close(montlok_stream* stream);

/* ------------------------------------------------------------------------ */
/* Plugin model families                                                     */
/* ------------------------------------------------------------------------ */

#define MONTLOK_PLUGIN_ABI_VERSION 1u

typedef struct montlok_archive montlok_archive;

typedef enum montlok_tensor_dtype {
    MONTLOK_TENSOR_F64 = 1,
    MONTLOK_TENSOR_F32 = 2,
    MONTLOK_TENSOR_I64 = 3,
    MONTLOK_TENSOR_I32 = 4,
    MONTLOK_TENSOR_U8 = 5,
    MONTLOK_TENSOR_BOOL = 6
} montlok_tensor_dtype;

/* A tensor of the model file. The data stays valid until the family's
 * destroy callback returns; it is aligned to its element size when the file
 * was written by montlok.export. */
typedef struct montlok_tensor_view {
    int32_t dtype; /* montlok_tensor_dtype */
    int32_t ndim;
    const int64_t* shape;
    const void* data;
    size_t nbytes;
} montlok_tensor_view;

/* Dimensions of a model: what the spec declares (host->dims) and what a
 * family reports for a model it created (which must match). */
typedef struct montlok_family_shape_v1 {
    uint32_t features;
    uint32_t sequence;
    uint32_t outputs;
    uint32_t streamable; /* nonzero: stream callbacks are implemented */
} montlok_family_shape_v1;

/* Services the runtime offers to family callbacks. Each returns 0 on
 * success. */
typedef struct montlok_host_v1 {
    uint32_t abi_version;
    int (*tensor)(const montlok_archive* archive, const char* name, montlok_tensor_view* out);
    /* JSON text of the spec's "config" object, and of the whole spec. */
    const char* (*config_json)(const montlok_archive* archive);
    const char* (*spec_json)(const montlok_archive* archive);
    /* The spec's inputs.features, inputs.sequence and outputs.size
     * (streamable is left 0). */
    int (*dims)(const montlok_archive* archive, montlok_family_shape_v1* out);
} montlok_host_v1;

/* A model family. Every callback that can fail returns 0 on success and
 * otherwise writes a message into `error` (capacity `error_size`). Inputs are
 * already preprocessed float64: batch * max(sequence, 1) * features values for
 * predict, batch * features for stream_step. predict must be safe to call
 * concurrently on one model. */
typedef struct montlok_family_v1 {
    uint32_t abi_version; /* MONTLOK_PLUGIN_ABI_VERSION */
    const char* name;
    void* (*create)(const montlok_host_v1* host, const montlok_archive* archive, montlok_family_shape_v1* shape,
                    char* error, size_t error_size);
    void (*destroy)(void* model);
    int (*predict)(void* model, const double* x, size_t batch, double* out, char* error, size_t error_size);
    /* Optional (null when the family is not streamable). */
    void* (*stream_create)(void* model, size_t batch, char* error, size_t error_size);
    void (*stream_destroy)(void* stream);
    int (*stream_step)(void* model, void* stream, const double* x, double* out, char* error, size_t error_size);
    void (*stream_reset)(void* stream);
} montlok_family_v1;

/* What a plugin's montlok_plugin_init fills in. The family array must stay
 * valid for the life of the process. */
typedef struct montlok_plugin_v1 {
    uint32_t abi_version;
    size_t family_count;
    const montlok_family_v1* families;
} montlok_plugin_v1;

/* Signature of the function a plugin library exports as
 * "montlok_plugin_init". */
typedef int (*montlok_plugin_init_fn)(const montlok_host_v1* host, montlok_plugin_v1* plugin);

/* Registers a family from inside the process (the struct is copied). A name
 * that is already registered is rejected. */
MONTLOK_API int montlok_register_family(const montlok_family_v1* family);
/* Loads a shared library and registers the families it provides. */
MONTLOK_API int montlok_plugin_load(const char* path);

/* ------------------------------------------------------------------------ */
/* Streaming L1 / trade microstructure features (microstructure/)           */
/* ------------------------------------------------------------------------ */

/* kind 0 reference (a: raw price, b: fair value, c: uncertainty), kind 1 L1
 * book (a: bid, b: ask, c: bid size, d: ask size), kind 2 execution (a: price,
 * b: quantity, side: +1 buy / -1 sell aggressor). receive_ns is one monotonic
 * collector clock. */
typedef struct montlok_micro_event {
    int64_t receive_ns;
    int32_t kind;
    int32_t side;
    double a, b, c, d;
} montlok_micro_event;

MONTLOK_API const char* montlok_micro_schema(void);
MONTLOK_API size_t montlok_micro_event_size(void);
MONTLOK_API void* montlok_micro_create(size_t capacity);
MONTLOK_API void montlok_micro_destroy(void* engine);
MONTLOK_API void montlok_micro_reset(void* engine);
/* 0 success, -1 invalid event or capacity overflow (reset required). */
MONTLOK_API int montlok_micro_update(void* engine, const montlok_micro_event* event);
/* 1 ready, 0 warming up or faulted, -1 bad arguments; count must be 32. */
MONTLOK_API int montlok_micro_snapshot(void* engine, double* out, size_t count);
MONTLOK_API int montlok_micro_linear(const double* x, const double* w, const double* bias, size_t heads, double* out);

#ifdef __cplusplus
}
#endif

#endif /* MONTLOK_H */
