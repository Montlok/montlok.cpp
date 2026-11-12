// Copyright 2026 Montlok. Apache-2.0.
// Plugin model families: the C vtable of montlok.h, registered in-process
// (montlok_register_family) or loaded from a shared library that exports
// montlok_plugin_init. Families read their model through montlok_host_v1.
#include "plugin.hpp"

#include <deque>
#include <mutex>

#include "runtime.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

struct montlok_archive {
    const montlok::core::Archive* archive = nullptr;
    std::string config_json;
    montlok_family_shape_v1 dims{};
};

namespace montlok::core {
namespace {

int32_t tensor_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::F64: return MONTLOK_TENSOR_F64;
        case DType::F32: return MONTLOK_TENSOR_F32;
        case DType::I64: return MONTLOK_TENSOR_I64;
        case DType::I32: return MONTLOK_TENSOR_I32;
        case DType::U8: return MONTLOK_TENSOR_U8;
        case DType::Bool: return MONTLOK_TENSOR_BOOL;
    }
    return 0;
}

int host_tensor(const montlok_archive* archive, const char* name, montlok_tensor_view* out) {
    if (!archive || !archive->archive || !name || !out) return 1;
    const Tensor* t = archive->archive->find(name);
    if (!t) return 1;
    out->dtype = tensor_dtype(t->dtype);
    out->ndim = static_cast<int32_t>(t->shape.size());
    out->shape = t->shape.data();
    out->data = t->data;
    out->nbytes = t->nbytes;
    return 0;
}

const char* host_config(const montlok_archive* archive) { return archive ? archive->config_json.c_str() : "{}"; }

const char* host_spec(const montlok_archive* archive) {
    return archive && archive->archive ? archive->archive->spec_text().c_str() : "{}";
}

int host_dims(const montlok_archive* archive, montlok_family_shape_v1* out) {
    if (!archive || !out) return 1;
    *out = archive->dims;
    return 0;
}

const montlok_host_v1 kHost = {MONTLOK_PLUGIN_ABI_VERSION, &host_tensor, &host_config, &host_spec, &host_dims};

struct StoredFamily {
    montlok_family_v1 vtable;
    std::string name;
    std::string origin;
};

std::mutex g_mutex;
std::deque<StoredFamily> g_families;  // stable addresses for the life of the process

std::string message(const char* family, const char* buffer, const char* fallback) {
    return std::string(family) + ": " + (buffer && buffer[0] ? buffer : fallback);
}

class PluginStream final : public FamilyStream {
public:
    PluginStream(const montlok_family_v1& family, void* model, void* stream)
        : family_(family), model_(model), stream_(stream) {}
    ~PluginStream() override { family_.stream_destroy(stream_); }

    void step(const double* x, double* out) override {
        char error[512] = {0};
        if (family_.stream_step(model_, stream_, x, out, error, sizeof(error)) != 0) {
            fail(Status::Internal, message(family_.name, error, "stream step failed"));
        }
    }
    void reset() override { family_.stream_reset(stream_); }

private:
    const montlok_family_v1& family_;
    void* model_;
    void* stream_;
};

class PluginModel final : public FamilyModel {
public:
    PluginModel(const montlok_family_v1& family, const BuildContext& ctx) : family_(family), archive_(ctx.archive) {
        view_.archive = archive_.get();
        view_.config_json = ctx.config.dump();
        view_.dims.features = static_cast<uint32_t>(ctx.features);
        view_.dims.sequence = static_cast<uint32_t>(ctx.sequence);
        view_.dims.outputs = static_cast<uint32_t>(json_size(*ctx.spec.find("outputs"), "size", 0));
        montlok_family_shape_v1 shape{};
        char error[512] = {0};
        model_ = family_.create(&kHost, &view_, &shape, error, sizeof(error));
        if (!model_) fail(Status::Format, message(family_.name, error, "model creation failed"));
        shape_ = Shape{shape.features, shape.sequence, shape.outputs,
                       shape.streamable != 0 && family_.stream_create != nullptr};
    }
    PluginModel(const PluginModel&) = delete;
    PluginModel& operator=(const PluginModel&) = delete;
    ~PluginModel() override { family_.destroy(model_); }

    Shape shape() const override { return shape_; }

    void predict(const double* x, size_t batch, double* out) const override {
        char error[512] = {0};
        if (family_.predict(model_, x, batch, out, error, sizeof(error)) != 0) {
            fail(Status::Internal, message(family_.name, error, "predict failed"));
        }
    }

    std::unique_ptr<FamilyStream> stream(size_t batch) const override {
        if (!family_.stream_create) fail(Status::Unsupported, std::string(family_.name) + " is not streamable");
        char error[512] = {0};
        void* stream = family_.stream_create(model_, batch, error, sizeof(error));
        if (!stream) fail(Status::Internal, message(family_.name, error, "stream creation failed"));
        return std::make_unique<PluginStream>(family_, model_, stream);
    }

private:
    const montlok_family_v1& family_;
    std::shared_ptr<const Archive> archive_;
    montlok_archive view_;
    void* model_ = nullptr;
    Shape shape_;
};

}  // namespace

void register_family_v1(const montlok_family_v1* family, const std::string& origin) {
    if (!family) fail(Status::InvalidArgument, "null family");
    if (family->abi_version != MONTLOK_PLUGIN_ABI_VERSION) {
        fail(Status::Unsupported, "family ABI version " + std::to_string(family->abi_version) + " (runtime supports " +
                                      std::to_string(MONTLOK_PLUGIN_ABI_VERSION) + ")");
    }
    if (!family->name || !family->create || !family->destroy || !family->predict) {
        fail(Status::InvalidArgument, "family needs name, create, destroy and predict");
    }
    if (family->stream_create && (!family->stream_destroy || !family->stream_step || !family->stream_reset)) {
        fail(Status::InvalidArgument, std::string(family->name) + ": stream callbacks must be given together");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_families.push_back(StoredFamily{*family, family->name, origin});
    StoredFamily& stored = g_families.back();
    stored.vtable.name = stored.name.c_str();
    const montlok_family_v1* vtable = &stored.vtable;
    try {
        Registry::instance().add(
            stored.name,
            [vtable](const BuildContext& ctx) -> std::unique_ptr<FamilyModel> {
                return std::make_unique<PluginModel>(*vtable, ctx);
            },
            origin);
    } catch (...) {
        g_families.pop_back();
        throw;
    }
}

void load_plugin(const std::string& path) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(path.c_str());
    if (!handle) fail(Status::Io, "cannot load plugin " + path);
    auto init = reinterpret_cast<montlok_plugin_init_fn>(GetProcAddress(handle, "montlok_plugin_init"));
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* why = dlerror();
        fail(Status::Io, "cannot load plugin " + path + ": " + (why ? why : "unknown error"));
    }
    auto init = reinterpret_cast<montlok_plugin_init_fn>(dlsym(handle, "montlok_plugin_init"));
#endif
    // The library stays loaded: registered families point into it.
    if (!init) fail(Status::Format, path + " does not export montlok_plugin_init");
    montlok_plugin_v1 plugin{};
    if (init(&kHost, &plugin) != 0) fail(Status::Internal, path + ": montlok_plugin_init failed");
    if (plugin.abi_version != MONTLOK_PLUGIN_ABI_VERSION) {
        fail(Status::Unsupported, path + " was built for plugin ABI " + std::to_string(plugin.abi_version));
    }
    if (plugin.family_count && !plugin.families) fail(Status::Format, path + " lists families but gives none");
    for (size_t i = 0; i < plugin.family_count; ++i) register_family_v1(&plugin.families[i], path);
}

}  // namespace montlok::core
