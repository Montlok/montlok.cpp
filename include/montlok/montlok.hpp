// montlok.hpp - header-only C++17 wrapper over the montlok C ABI.
//
// Copyright 2026 Montlok. Apache-2.0.
//
//   montlok::Model model = montlok::Model::open("gru.montlok");
//   std::vector<double> y = model.predict(x, batch);   // x: batch * steps * features
//   montlok::Stream live = model.stream();
//   live.step(features_now, y.data());
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "montlok.h"

namespace montlok {

class Error : public std::runtime_error {
public:
    Error(int status, const std::string& message) : std::runtime_error(message), status_(status) {}
    int status() const noexcept { return status_; }

private:
    int status_;
};

inline void check(int status) {
    if (status != MONTLOK_OK) throw Error(status, montlok_last_error());
}

inline std::string version() { return montlok_version(); }
inline void set_threads(int threads) { montlok_set_threads(threads); }
inline int threads() { return montlok_threads(); }
inline void set_latency_threads(int threads) { check(montlok_set_latency_threads(threads)); }
inline int latency_threads() { return montlok_latency_threads(); }
inline void load_plugin(const std::string& path) { check(montlok_plugin_load(path.c_str())); }

struct ModelInfo {
    uint32_t features = 0;
    uint32_t sequence = 0;
    uint32_t outputs = 0;
    bool streamable = false;
    bool preprocess = false;

    size_t steps() const noexcept { return sequence == 0 ? 1 : sequence; }
    size_t input_size() const noexcept { return steps() * features; }
};

class Stream {
public:
    void step(const float* x, double* out) { check(montlok_stream_step(handle_.get(), MONTLOK_FLOAT32, x, out)); }
    void step(const double* x, double* out) { check(montlok_stream_step(handle_.get(), MONTLOK_FLOAT64, x, out)); }
    void reset() { check(montlok_stream_reset(handle_.get())); }
    size_t batch() const noexcept { return batch_; }
    montlok_stream* get() const noexcept { return handle_.get(); }

private:
    friend class Model;
    Stream(montlok_stream* stream, size_t batch) : handle_(stream, &montlok_stream_close), batch_(batch) {}
    std::unique_ptr<montlok_stream, void (*)(montlok_stream*)> handle_;
    size_t batch_;
};

class Model {
public:
    static Model open(const std::string& path) {
        montlok_model* model = nullptr;
        check(montlok_model_open(path.c_str(), &model));
        return Model(model);
    }

    static Model from_memory(const void* data, size_t size) {
        montlok_model* model = nullptr;
        check(montlok_model_open_memory(data, size, &model));
        return Model(model);
    }

    ModelInfo info() const {
        montlok_model_info raw{};
        check(montlok_model_get_info(get(), &raw));
        ModelInfo info;
        info.features = raw.features;
        info.sequence = raw.sequence;
        info.outputs = raw.outputs;
        info.streamable = (raw.flags & MONTLOK_MODEL_STREAMABLE) != 0;
        info.preprocess = (raw.flags & MONTLOK_MODEL_PREPROCESS) != 0;
        return info;
    }

    std::string family() const { return montlok_model_family(get()); }
    std::string spec() const { return montlok_model_spec(get()); }

    void predict(const float* x, size_t batch, double* out) const {
        check(montlok_model_predict(get(), MONTLOK_FLOAT32, x, batch, out));
    }
    void predict(const double* x, size_t batch, double* out) const {
        check(montlok_model_predict(get(), MONTLOK_FLOAT64, x, batch, out));
    }

    template <typename T>
    std::vector<double> predict(const std::vector<T>& x, size_t batch) const {
        const ModelInfo i = info();
        if (x.size() != batch * i.input_size()) throw Error(MONTLOK_ERROR_SHAPE, "input size does not match batch");
        std::vector<double> out(batch * i.outputs);
        predict(x.data(), batch, out.data());
        return out;
    }

    Stream stream(size_t batch = 1) const {
        montlok_stream* stream = nullptr;
        check(montlok_stream_open(get(), batch, &stream));
        return Stream(stream, batch);
    }

    montlok_model* get() const noexcept { return handle_.get(); }

private:
    explicit Model(montlok_model* model) : handle_(model, &montlok_model_close) {}
    std::shared_ptr<montlok_model> handle_;
};

}  // namespace montlok
