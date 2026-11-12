// Copyright 2026 Montlok. Apache-2.0.
// Internal C++ interfaces of the montlok runtime. Consumers use the C ABI
// (include/montlok/montlok.h) or its C++ wrapper (montlok.hpp); this header is
// for the families compiled into the library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"

namespace montlok::core {

// Mirrors montlok_status.
enum class Status : int {
    Ok = 0,
    InvalidArgument = 1,
    Io = 2,
    Format = 3,
    Unsupported = 4,
    Shape = 5,
    State = 6,
    Internal = 7,
};

class Error : public std::runtime_error {
public:
    Error(Status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    Status status() const noexcept { return status_; }

private:
    Status status_;
};

[[noreturn]] inline void fail(Status status, const std::string& message) { throw Error(status, message); }

const char* version() noexcept;

// ---------------------------------------------------------------------------
// Model files: safetensors with the spec in __metadata__["montlok"].
// ---------------------------------------------------------------------------

enum class DType : uint8_t { F64 = 1, F32 = 2, I64 = 3, I32 = 4, U8 = 5, Bool = 6 };

size_t dtype_size(DType dtype) noexcept;
const char* dtype_name(DType dtype) noexcept;

struct Tensor {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    const uint8_t* data;  // into the archive's buffer
    size_t nbytes;

    size_t numel() const noexcept;
    std::string describe() const;
};

class Archive {
public:
    static std::shared_ptr<const Archive> from_file(const std::string& path);
    static std::shared_ptr<const Archive> from_bytes(const void* data, size_t size);

    const json::Value& spec() const noexcept { return spec_; }
    const std::string& spec_text() const noexcept { return spec_text_; }
    const std::vector<Tensor>& tensors() const noexcept { return tensors_; }

    const Tensor* find(std::string_view name) const noexcept;
    const Tensor& tensor(std::string_view name) const;

    // Typed copies. `shape` entries of -1 accept any extent; the copy converts
    // between floating types (f32 -> f64 exactly) and between integer types.
    std::vector<double> f64(std::string_view name, const std::vector<int64_t>& shape) const;
    std::vector<int32_t> i32(std::string_view name, const std::vector<int64_t>& shape) const;
    std::vector<uint8_t> u8(std::string_view name, const std::vector<int64_t>& shape) const;

private:
    struct AlignedDelete {
        void operator()(uint8_t* p) const noexcept { ::operator delete[](p, std::align_val_t(64)); }
    };

    void allocate(size_t size);
    void parse();
    const Tensor& checked(std::string_view name, const std::vector<int64_t>& shape) const;

    std::unique_ptr<uint8_t[], AlignedDelete> storage_;
    size_t size_ = 0;
    std::vector<Tensor> tensors_;
    json::Value spec_;
    std::string spec_text_;
};

// ---------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------

struct Shape {
    size_t features = 0;
    size_t sequence = 0;  // 0: tabular
    size_t outputs = 0;
    bool streamable = false;

    size_t steps() const noexcept { return sequence == 0 ? 1 : sequence; }
};

class FamilyStream {
public:
    virtual ~FamilyStream() = default;
    // x: [batch, features] preprocessed; out: [batch, outputs].
    virtual void step(const double* x, double* out) = 0;
    virtual void reset() = 0;
};

class FamilyModel {
public:
    virtual ~FamilyModel() = default;
    virtual Shape shape() const = 0;
    // x: [batch, steps, features] preprocessed; out: [batch, outputs].
    // Must be safe to call concurrently.
    virtual void predict(const double* x, size_t batch, double* out) const = 0;
    virtual std::unique_ptr<FamilyStream> stream(size_t batch) const;
    // Rough multiply-add count per sample; steers batch threading.
    virtual double cost_per_sample() const { return 0; }
};

// What a family factory receives.
struct BuildContext {
    std::shared_ptr<const Archive> archive;
    const json::Value& spec;    // whole spec
    const json::Value& config;  // spec["config"] (an object, possibly empty)
    size_t features;            // declared input features
    size_t sequence;            // declared sequence length (0: tabular)
};

using FamilyFactory = std::function<std::unique_ptr<FamilyModel>(const BuildContext&)>;

class Registry {
public:
    static Registry& instance();
    // Throws Status::InvalidArgument if the name is taken.
    void add(const std::string& name, FamilyFactory factory, const std::string& origin);
    FamilyFactory find(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    Registry();
    struct Entry {
        FamilyFactory factory;
        std::string origin;
    };
    mutable std::mutex mutex_;
    std::map<std::string, Entry> families_;
};

// Built-in families register themselves through these, called once by
// Registry's constructor so that static builds link them in.
void register_sequential(Registry& registry);
void register_tree_ensemble(Registry& registry);

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

void set_threads(int threads) noexcept;
int threads() noexcept;
// Runs fn(begin, end) over [0, n) in chunks of at least `grain`, on the pool
// when that is worth it, otherwise inline. Re-entrant: a nested or concurrent
// call runs inline.
void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn);

// Latency threads wait for work in a spin loop, so one sample can be split
// across cores within a fraction of a microsecond. Each keeps its core busy
// while it exists; 0 stops them.
void set_latency_threads(int threads);
int latency_threads() noexcept;

// Runs task(context, part, parts) for every part in [0, parts): part 0 on the
// calling thread, the others on latency threads. parts <= max_parts; it is 1
// when latency threads are off or busy with another call. Tasks must not throw.
using PartTask = void (*)(const void* context, size_t part, size_t parts);
void run_parts(size_t max_parts, PartTask task, const void* context) noexcept;

template <typename F>
void run_parts(size_t max_parts, const F& f) noexcept {
    run_parts(max_parts, [](const void* c, size_t part, size_t parts) { (*static_cast<const F*>(c))(part, parts); }, &f);
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

enum class InputType : uint8_t { F32 = 1, F64 = 2 };

class Stream;

class Model : public std::enable_shared_from_this<Model> {
public:
    static std::shared_ptr<const Model> open(const std::string& path);
    static std::shared_ptr<const Model> from_bytes(const void* data, size_t size);

    const Shape& shape() const noexcept { return shape_; }
    const std::string& family() const noexcept { return family_; }
    const std::string& spec_text() const noexcept { return archive_->spec_text(); }
    bool has_preprocess() const noexcept { return standardize_ || clip_; }

    void predict(InputType type, const void* x, size_t batch, double* out) const;
    std::unique_ptr<Stream> stream(size_t batch) const;

    // Raw [rows, features] values -> standardized/clipped float64. With no
    // preprocessing this only widens the values.
    void preprocess(InputType type, const void* x, size_t rows, double* out) const;

private:
    explicit Model(std::shared_ptr<const Archive> archive);
    static std::shared_ptr<const Model> build(std::shared_ptr<const Archive> archive);

    std::shared_ptr<const Archive> archive_;
    std::string family_;
    Shape shape_;
    bool standardize_ = false;
    bool clip_ = false;
    std::vector<double> mean_, scale_;
    double clip_lo_ = 0, clip_hi_ = 0;
    std::unique_ptr<FamilyModel> impl_;
};

class Stream {
public:
    Stream(std::shared_ptr<const Model> model, std::unique_ptr<FamilyStream> impl, size_t batch);
    void step(InputType type, const void* x, double* out);
    void reset();
    size_t batch() const noexcept { return batch_; }

private:
    std::shared_ptr<const Model> model_;
    std::unique_ptr<FamilyStream> impl_;
    size_t batch_;
    std::vector<double> scratch_;
};

// Shared helpers for families.
size_t json_size(const json::Value& object, std::string_view key, size_t fallback);
double json_double(const json::Value& object, std::string_view key, double fallback);
std::string json_string(const json::Value& object, std::string_view key, const std::string& fallback);

}  // namespace montlok::core
