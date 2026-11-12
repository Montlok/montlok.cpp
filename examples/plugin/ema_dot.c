/*
 * ema_dot.c - a montlok model family as a plugin, in C.
 *
 * The model keeps an exponential moving average of an affine signal:
 *
 *     s_t = alpha * s_{t-1} + (1 - alpha) * (w . x_t + bias),   s_0 = 0
 *
 * predict() runs the window of `sequence` steps; a stream carries s from
 * call to call, so after the same steps it returns the same value.
 *
 * Tensors: ema.weight F64[features], ema.alpha F64[1], ema.bias F64[1], read
 * through the host table passed to montlok_plugin_init.
 *
 *     cc -O2 -shared -fPIC -I<montlok.cpp>/include ema_dot.c -o libema_dot.so
 *     python -c "import montlok; montlok.load_plugin('./libema_dot.so')"
 *
 * Copyright 2026 Montlok. Apache-2.0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "montlok/montlok.h"

typedef struct {
    montlok_family_shape_v1 shape;
    double alpha, bias;
    double* weight;
} ema_model;

typedef struct {
    size_t batch;
    double state[];
} ema_stream;

static void set_error(char* error, size_t size, const char* message) {
    if (error && size) snprintf(error, size, "%s", message);
}

static int scalar(const montlok_host_v1* host, const montlok_archive* archive, const char* name, double* out) {
    montlok_tensor_view view;
    if (host->tensor(archive, name, &view) != 0 || view.dtype != MONTLOK_TENSOR_F64 || view.nbytes != sizeof(double)) {
        return -1;
    }
    memcpy(out, view.data, sizeof(double));
    return 0;
}

static void* ema_create(const montlok_host_v1* host, const montlok_archive* archive, montlok_family_shape_v1* shape,
                        char* error, size_t error_size) {
    montlok_family_shape_v1 dims;
    montlok_tensor_view weight;
    if (host->dims(archive, &dims) != 0 || dims.outputs != 1) {
        set_error(error, error_size, "ema_dot has exactly one output");
        return NULL;
    }
    if (host->tensor(archive, "ema.weight", &weight) != 0 || weight.dtype != MONTLOK_TENSOR_F64 || weight.ndim != 1 ||
        (uint32_t)weight.shape[0] != dims.features) {
        set_error(error, error_size, "needs F64 tensor ema.weight[features]");
        return NULL;
    }
    ema_model* model = calloc(1, sizeof(*model));
    if (!model) return NULL;
    model->weight = malloc(weight.nbytes);
    if (!model->weight || scalar(host, archive, "ema.alpha", &model->alpha) != 0 ||
        scalar(host, archive, "ema.bias", &model->bias) != 0 || !(model->alpha >= 0 && model->alpha < 1)) {
        set_error(error, error_size, "needs ema.alpha[1] in [0, 1) and ema.bias[1]");
        free(model->weight);
        free(model);
        return NULL;
    }
    memcpy(model->weight, weight.data, weight.nbytes);
    model->shape = dims;
    model->shape.streamable = 1;
    *shape = model->shape;
    return model;
}

static void ema_destroy(void* handle) {
    ema_model* model = handle;
    free(model->weight);
    free(model);
}

static double ema_update(const ema_model* model, double state, const double* x) {
    double signal = model->bias;
    for (uint32_t i = 0; i < model->shape.features; ++i) signal += model->weight[i] * x[i];
    return model->alpha * state + (1.0 - model->alpha) * signal;
}

static int ema_predict(void* handle, const double* x, size_t batch, double* out, char* error, size_t error_size) {
    (void)error;
    (void)error_size;
    const ema_model* model = handle;
    const size_t steps = model->shape.sequence ? model->shape.sequence : 1;
    for (size_t b = 0; b < batch; ++b) {
        double state = 0.0;
        for (size_t t = 0; t < steps; ++t) state = ema_update(model, state, x + (b * steps + t) * model->shape.features);
        out[b] = state;
    }
    return 0;
}

static void* ema_stream_create(void* handle, size_t batch, char* error, size_t error_size) {
    (void)handle;
    ema_stream* stream = calloc(1, sizeof(*stream) + batch * sizeof(double));
    if (!stream) set_error(error, error_size, "out of memory");
    else stream->batch = batch;
    return stream;
}

static void ema_stream_destroy(void* stream) { free(stream); }

static int ema_stream_step(void* handle, void* opaque, const double* x, double* out, char* error, size_t error_size) {
    (void)error;
    (void)error_size;
    const ema_model* model = handle;
    ema_stream* stream = opaque;
    for (size_t b = 0; b < stream->batch; ++b) {
        stream->state[b] = ema_update(model, stream->state[b], x + b * model->shape.features);
        out[b] = stream->state[b];
    }
    return 0;
}

static void ema_stream_reset(void* opaque) {
    ema_stream* stream = opaque;
    memset(stream->state, 0, stream->batch * sizeof(double));
}

static const montlok_family_v1 kFamilies[] = {{
    MONTLOK_PLUGIN_ABI_VERSION, "ema_dot", ema_create, ema_destroy, ema_predict,
    ema_stream_create, ema_stream_destroy, ema_stream_step, ema_stream_reset,
}};

#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
int montlok_plugin_init(const montlok_host_v1* host, montlok_plugin_v1* plugin) {
    if (!host || host->abi_version < MONTLOK_PLUGIN_ABI_VERSION) return 1;
    plugin->abi_version = MONTLOK_PLUGIN_ABI_VERSION;
    plugin->family_count = sizeof(kFamilies) / sizeof(kFamilies[0]);
    plugin->families = kFamilies;
    return 0;
}
