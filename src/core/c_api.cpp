// Copyright 2026 Montlok. Apache-2.0.
// extern "C" implementation of include/montlok/montlok.h. Each entry point
// turns exceptions into a status code and the thread's last error.
#include <string>

#include "montlok/montlok.h"
#include "plugin.hpp"
#include "runtime.hpp"

struct montlok_model {
    std::shared_ptr<const montlok::core::Model> impl;
};

struct montlok_stream {
    std::unique_ptr<montlok::core::Stream> impl;
};

namespace {

using montlok::core::Error;
using montlok::core::Status;

thread_local std::string t_error;
thread_local std::string t_families;

void set_error(const char* message) noexcept {
    try {
        t_error = message;
    } catch (...) {
    }
}

template <typename F>
int guard(F&& f) noexcept {
    try {
        f();
        return MONTLOK_OK;
    } catch (const Error& e) {
        set_error(e.what());
        return static_cast<int>(e.status());
    } catch (const std::bad_alloc&) {
        set_error("out of memory");
        return MONTLOK_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        set_error(e.what());
        return MONTLOK_ERROR_INTERNAL;
    } catch (...) {
        set_error("unknown error");
        return MONTLOK_ERROR_INTERNAL;
    }
}

montlok::core::InputType input_type(montlok_dtype dtype) {
    if (dtype == MONTLOK_FLOAT32) return montlok::core::InputType::F32;
    if (dtype == MONTLOK_FLOAT64) return montlok::core::InputType::F64;
    throw Error(Status::InvalidArgument, "dtype must be MONTLOK_FLOAT32 or MONTLOK_FLOAT64");
}

void require(bool ok, const char* what) {
    if (!ok) throw Error(Status::InvalidArgument, what);
}

}  // namespace

extern "C" {

uint32_t montlok_abi_version(void) { return MONTLOK_ABI_VERSION; }

const char* montlok_version(void) { return montlok::core::version(); }

const char* montlok_last_error(void) { return t_error.c_str(); }

void montlok_set_threads(int threads) { montlok::core::set_threads(threads); }

int montlok_threads(void) { return montlok::core::threads(); }

int montlok_set_latency_threads(int threads) {
    return guard([&] { montlok::core::set_latency_threads(threads); });
}

int montlok_latency_threads(void) { return montlok::core::latency_threads(); }

const char* montlok_families(void) {
    try {
        t_families.clear();
        for (const auto& name : montlok::core::Registry::instance().names()) {
            if (!t_families.empty()) t_families += ',';
            t_families += name;
        }
    } catch (...) {
        t_families.clear();
    }
    return t_families.c_str();
}

int montlok_model_open(const char* path, montlok_model** model) {
    if (model) *model = nullptr;
    return guard([&] {
        require(path && model, "null argument");
        auto handle = std::make_unique<montlok_model>();
        handle->impl = montlok::core::Model::open(path);
        *model = handle.release();
    });
}

int montlok_model_open_memory(const void* data, size_t size, montlok_model** model) {
    if (model) *model = nullptr;
    return guard([&] {
        require(model != nullptr && (data != nullptr || size == 0), "null argument");
        auto handle = std::make_unique<montlok_model>();
        handle->impl = montlok::core::Model::from_bytes(data, size);
        *model = handle.release();
    });
}

void montlok_model_close(montlok_model* model) { delete model; }

int montlok_model_get_info(const montlok_model* model, montlok_model_info* info) {
    return guard([&] {
        require(model && info, "null argument");
        const auto& shape = model->impl->shape();
        info->features = static_cast<uint32_t>(shape.features);
        info->sequence = static_cast<uint32_t>(shape.sequence);
        info->outputs = static_cast<uint32_t>(shape.outputs);
        info->flags = (shape.streamable ? MONTLOK_MODEL_STREAMABLE : 0u) |
                      (model->impl->has_preprocess() ? MONTLOK_MODEL_PREPROCESS : 0u);
    });
}

const char* montlok_model_family(const montlok_model* model) {
    return model ? model->impl->family().c_str() : "";
}

const char* montlok_model_spec(const montlok_model* model) { return model ? model->impl->spec_text().c_str() : ""; }

int montlok_model_predict(const montlok_model* model, montlok_dtype dtype, const void* x, size_t batch, double* out) {
    return guard([&] {
        require(model != nullptr, "null model");
        model->impl->predict(input_type(dtype), x, batch, out);
    });
}

int montlok_stream_open(const montlok_model* model, size_t batch, montlok_stream** stream) {
    if (stream) *stream = nullptr;
    return guard([&] {
        require(model && stream, "null argument");
        auto handle = std::make_unique<montlok_stream>();
        handle->impl = model->impl->stream(batch);
        *stream = handle.release();
    });
}

int montlok_stream_step(montlok_stream* stream, montlok_dtype dtype, const void* x, double* out) {
    return guard([&] {
        require(stream != nullptr, "null stream");
        stream->impl->step(input_type(dtype), x, out);
    });
}

int montlok_stream_reset(montlok_stream* stream) {
    return guard([&] {
        require(stream != nullptr, "null stream");
        stream->impl->reset();
    });
}

void montlok_stream_close(montlok_stream* stream) { delete stream; }

int montlok_register_family(const montlok_family_v1* family) {
    return guard([&] { montlok::core::register_family_v1(family, "registered"); });
}

int montlok_plugin_load(const char* path) {
    return guard([&] {
        require(path != nullptr, "null path");
        montlok::core::load_plugin(path);
    });
}

}  // extern "C"
