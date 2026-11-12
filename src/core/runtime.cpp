// Copyright 2026 Montlok. Apache-2.0.
// Family registry, worker threads, and the Model/Stream wrappers that apply
// input preprocessing, chunking and batch threading around every family.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <thread>

#include "runtime.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#include <unistd.h>
#define MONTLOK_HAVE_GETPID 1
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#if defined(__linux__)
#include <sched.h>
#include <cstdlib>
#include <string>
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define MONTLOK_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define MONTLOK_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define MONTLOK_SPIN_PAUSE() ((void)0)
#endif

namespace montlok::core {

const char* version() noexcept { return "0.2.0"; }

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

size_t json_size(const json::Value& object, std::string_view key, size_t fallback) {
    const json::Value* v = object.find(key);
    if (!v || v->is_null()) return fallback;
    int64_t n = 0;
    try {
        n = v->as_int();
    } catch (const std::exception&) {
        fail(Status::Format, "\"" + std::string(key) + "\" must be a non-negative integer");
    }
    if (n < 0) fail(Status::Format, "\"" + std::string(key) + "\" must be a non-negative integer");
    return static_cast<size_t>(n);
}

double json_double(const json::Value& object, std::string_view key, double fallback) {
    const json::Value* v = object.find(key);
    if (!v || v->is_null()) return fallback;
    if (!v->is_number()) fail(Status::Format, "\"" + std::string(key) + "\" must be a number");
    return v->as_number();
}

std::string json_string(const json::Value& object, std::string_view key, const std::string& fallback) {
    const json::Value* v = object.find(key);
    if (!v || v->is_null()) return fallback;
    if (!v->is_string()) fail(Status::Format, "\"" + std::string(key) + "\" must be a string");
    return v->as_string();
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

std::unique_ptr<FamilyStream> FamilyModel::stream(size_t) const {
    fail(Status::Unsupported, "this model family does not support streaming");
}

Registry::Registry() {
    register_sequential(*this);
    register_tree_ensemble(*this);
}

Registry& Registry::instance() {
    static Registry registry;
    return registry;
}

void Registry::add(const std::string& name, FamilyFactory factory, const std::string& origin) {
    if (name.empty() || name.size() > 128) fail(Status::InvalidArgument, "family name must be 1-128 characters");
    std::lock_guard<std::mutex> lock(mutex_);
    if (families_.count(name)) fail(Status::InvalidArgument, "model family already registered: " + name);
    families_.emplace(name, Entry{std::move(factory), origin});
}

FamilyFactory Registry::find(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = families_.find(name);
    return it == families_.end() ? FamilyFactory() : it->second.factory;
}

std::vector<std::string> Registry::names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto& entry : families_) out.push_back(entry.first);
    return out;
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

namespace {

std::atomic<int> g_requested_threads{0};
thread_local bool t_in_parallel = false;

int resolved_threads() noexcept {
    int n = g_requested_threads.load(std::memory_order_relaxed);
    if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
    return std::clamp(n, 1, 256);
}

class Pool {
public:
    explicit Pool(int workers) {
        for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { loop(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& t : threads_) t.join();
    }

    int workers() const noexcept { return static_cast<int>(threads_.size()); }

    void run(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            n_ = n;
            grain_ = grain;
            next_.store(0, std::memory_order_relaxed);
            active_ = static_cast<int>(threads_.size());
            error_ = nullptr;
            ++generation_;
        }
        wake_.notify_all();
        work();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return active_ == 0; });
        fn_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    void work() {
        t_in_parallel = true;
        try {
            while (true) {
                const size_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
                if (begin >= n_) break;
                (*fn_)(begin, std::min(n_, begin + grain_));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
            next_.store(n_, std::memory_order_relaxed);
        }
        t_in_parallel = false;
    }

    void loop() {
        size_t seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            work();
            std::lock_guard<std::mutex> lock(mutex_);
            if (--active_ == 0) done_.notify_one();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(size_t, size_t)>* fn_ = nullptr;
    size_t n_ = 0, grain_ = 1;
    std::atomic<size_t> next_{0};
    size_t generation_ = 0;
    int active_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

std::mutex g_pool_mutex;
Pool* g_pool = nullptr;  // leaked at exit: the workers end with the process
#ifdef MONTLOK_HAVE_GETPID
pid_t g_pool_pid = 0;
#endif

Pool& pool_for(int threads) {
#ifdef MONTLOK_HAVE_GETPID
    // After fork() the child has the pool object but none of its threads.
    if (g_pool && g_pool_pid != getpid()) g_pool = nullptr;
#endif
    if (g_pool && g_pool->workers() != threads - 1) {
        delete g_pool;
        g_pool = nullptr;
    }
    if (!g_pool) {
        g_pool = new Pool(threads - 1);
#ifdef MONTLOK_HAVE_GETPID
        g_pool_pid = getpid();
#endif
    }
    return *g_pool;
}

}  // namespace

void set_threads(int threads) noexcept { g_requested_threads.store(std::max(threads, 0), std::memory_order_relaxed); }

int threads() noexcept { return resolved_threads(); }

void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn) {
    if (n == 0) return;
    grain = std::max<size_t>(grain, 1);
    const int t = resolved_threads();
    if (t <= 1 || n <= grain || t_in_parallel) {
        fn(0, n);
        return;
    }
    std::unique_lock<std::mutex> lock(g_pool_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        fn(0, n);
        return;
    }
    pool_for(t).run(n, grain, fn);
}

// ---------------------------------------------------------------------------
// Latency threads
// ---------------------------------------------------------------------------

namespace {

// Each worker spins on its own mailbox line. A call writes the task into the
// mailboxes of the workers it needs, publishes a word holding a new epoch and
// the part count in each, runs part 0 and waits until each of those workers
// has stored the word back in `done`. Only the workers with a part are woken.
// `busy_` admits one call or one reconfiguration at a time.
class Team {
public:
    Team() {
#ifdef MONTLOK_HAVE_GETPID
        pthread_atfork(nullptr, nullptr, [] { team_after_fork(); });
#endif
    }

    void configure(int workers) {
        while (busy_.exchange(true, std::memory_order_acquire)) MONTLOK_SPIN_PAUSE();
        if (forked_) {
            // A forked child has the thread objects but none of the threads.
            new std::vector<std::thread>(std::move(threads_));
            threads_.clear();
            forked_ = false;
        }
        stop_.store(true, std::memory_order_relaxed);
        for (auto& thread : threads_) thread.join();
        threads_.clear();
        stop_.store(false, std::memory_order_relaxed);
        workers_.store(0, std::memory_order_relaxed);
        const auto count = static_cast<size_t>(workers);
        boxes_.reset(count ? new Mailbox[count] : nullptr);
        try {
            for (size_t i = 0; i < count; ++i) threads_.emplace_back([this, i] { work(i); });
        } catch (...) {
            stop_.store(true, std::memory_order_relaxed);
            for (auto& thread : threads_) thread.join();
            threads_.clear();
            stop_.store(false, std::memory_order_relaxed);
            busy_.store(false, std::memory_order_release);
            throw;
        }
        workers_.store(workers, std::memory_order_relaxed);
        busy_.store(false, std::memory_order_release);
    }

    int workers() const noexcept { return workers_.load(std::memory_order_relaxed); }

    bool run(size_t max_parts, PartTask task, const void* context) noexcept {
        if (max_parts < 2 || workers_.load(std::memory_order_relaxed) == 0) return false;
        if (busy_.exchange(true, std::memory_order_acquire)) return false;
        const auto workers = static_cast<size_t>(workers_.load(std::memory_order_relaxed));
        if (workers == 0) {
            busy_.store(false, std::memory_order_release);
            return false;
        }
        const size_t parts = std::min(max_parts, workers + 1);
        const uint64_t word = ++epoch_ << kPartBits | parts;
        for (size_t i = 0; i + 1 < parts; ++i) {
            boxes_[i].task = task;
            boxes_[i].context = context;
        }
        std::atomic_thread_fence(std::memory_order_release);
        for (size_t i = 0; i + 1 < parts; ++i) boxes_[i].word.store(word, std::memory_order_relaxed);
        task(context, 0, parts);
        // One pass loads every mailbox, so the cache misses overlap.
        bool done = false;
        while (!done) {
            done = true;
            for (size_t i = 0; i + 1 < parts; ++i) done &= boxes_[i].done.load(std::memory_order_acquire) == word;
            if (!done) MONTLOK_SPIN_PAUSE();
        }
        busy_.store(false, std::memory_order_release);
        return true;
    }

    // In a forked child: no worker threads, nothing in flight.
    void after_fork() noexcept {
        forked_ = true;
        workers_.store(0, std::memory_order_relaxed);
        busy_.store(false, std::memory_order_relaxed);
    }

private:
    static constexpr unsigned kPartBits = 16;

    struct alignas(64) Mailbox {
        std::atomic<uint64_t> word{0};  // epoch << kPartBits | parts
        PartTask task = nullptr;
        const void* context = nullptr;
        std::atomic<uint64_t> done{0};
    };

    static void team_after_fork() noexcept;

    // MONTLOK_LATENCY_CPUS (Linux): CPUs for the latency threads, such as
    // "2-5,8"; thread i runs on the i-th, cycling through the list.
    static void place(size_t index) noexcept {
        (void)index;
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__) && defined(CPU_SET)
        const char* text = std::getenv("MONTLOK_LATENCY_CPUS");
        if (!text) return;
        std::vector<int> cpus;
        for (const char* p = text; *p;) {
            char* end = nullptr;
            const long first = std::strtol(p, &end, 10);
            if (end == p || first < 0 || first >= CPU_SETSIZE) return;
            long last = first;
            p = end;
            if (*p == '-') {
                last = std::strtol(p + 1, &end, 10);
                if (end == p + 1 || last < first || last >= CPU_SETSIZE) return;
                p = end;
            }
            for (long cpu = first; cpu <= last; ++cpu) cpus.push_back(static_cast<int>(cpu));
            if (*p == ',') ++p;
            else if (*p) return;
        }
        if (cpus.empty()) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[index % cpus.size()], &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
    }

    void work(size_t index) {
        place(index);
        Mailbox& box = boxes_[index];
        uint64_t seen = 0;  // a new mailbox; the first call may come before this thread runs
        while (true) {
            const uint64_t word = box.word.load(std::memory_order_acquire);
            if (word == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                MONTLOK_SPIN_PAUSE();
                continue;
            }
            seen = word;
            box.task(box.context, index + 1, word & ((uint64_t{1} << kPartBits) - 1));
            box.done.store(word, std::memory_order_release);
        }
    }

    alignas(64) std::atomic<bool> busy_{false};
    std::atomic<int> workers_{0};
    std::atomic<bool> stop_{false};
    uint64_t epoch_ = 0;
    bool forked_ = false;
    std::unique_ptr<Mailbox[]> boxes_;
    std::vector<std::thread> threads_;
};

Team& team() {
    static Team* const instance = new Team;  // leaked at exit, like the pool
    return *instance;
}

void Team::team_after_fork() noexcept { team().after_fork(); }

}  // namespace

void set_latency_threads(int threads) {
    if (threads < 0 || threads > 256) fail(Status::InvalidArgument, "latency threads must be in [0, 256]");
    team().configure(threads);
}

int latency_threads() noexcept { return team().workers(); }

void run_parts(size_t max_parts, PartTask task, const void* context) noexcept {
    if (!team().run(max_parts, task, context)) task(context, 0, 1);
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

namespace {

constexpr size_t kMaxExtent = size_t(1) << 24;
constexpr size_t kChunkValues = 1 << 16;          // 512 KiB of float64 per chunk
constexpr double kParallelCost = 4.0e5;           // multiply-adds before threading pays off
constexpr double kGrainCost = 1.0e5;

const json::Value& empty_object() {
    static const json::Value value = json::Value::parse("{}");
    return value;
}

}  // namespace

Model::Model(std::shared_ptr<const Archive> archive) : archive_(std::move(archive)) {
    const json::Value& spec = archive_->spec();
    family_ = json_string(spec, "family", "");
    if (family_.empty()) fail(Status::Format, "model spec has no \"family\"");

    const json::Value* inputs = spec.find("inputs");
    const json::Value* outputs = spec.find("outputs");
    if (!inputs || !inputs->is_object()) fail(Status::Format, "model spec has no \"inputs\" object");
    if (!outputs || !outputs->is_object()) fail(Status::Format, "model spec has no \"outputs\" object");
    const size_t features = json_size(*inputs, "features", 0);
    const size_t sequence = json_size(*inputs, "sequence", 0);
    const size_t output_size = json_size(*outputs, "size", 0);
    if (features == 0 || features > kMaxExtent) fail(Status::Format, "inputs.features must be in [1, 2^24]");
    if (sequence > kMaxExtent || features * std::max<size_t>(sequence, 1) > (size_t(1) << 28)) {
        fail(Status::Format, "inputs.sequence is too large");
    }
    if (output_size == 0 || output_size > kMaxExtent) fail(Status::Format, "outputs.size must be in [1, 2^24]");
    for (const char* key : {"names"}) {
        const json::Value* names = inputs->find(key);
        if (names && !names->is_null() && (!names->is_array() || names->as_array().size() != features)) {
            fail(Status::Format, "inputs.names must list one name per feature");
        }
    }

    if (const json::Value* pre = spec.find("preprocess"); pre && !pre->is_null()) {
        if (!pre->is_object()) fail(Status::Format, "\"preprocess\" must be an object");
        const json::Value* mean = pre->find("mean");
        const json::Value* scale = pre->find("scale");
        const bool has_mean = mean && !mean->is_null();
        const bool has_scale = scale && !scale->is_null();
        if (has_mean != has_scale) fail(Status::Format, "preprocess needs both \"mean\" and \"scale\"");
        if (has_mean) {
            if (!mean->is_string() || !scale->is_string()) {
                fail(Status::Format, "preprocess mean/scale name tensors in the file");
            }
            const int64_t f = static_cast<int64_t>(features);
            mean_ = archive_->f64(mean->as_string(), {f});
            scale_ = archive_->f64(scale->as_string(), {f});
            for (size_t i = 0; i < features; ++i) {
                if (!std::isfinite(mean_[i]) || !std::isfinite(scale_[i]) || !(scale_[i] > 0)) {
                    fail(Status::Format, "preprocess mean must be finite and scale finite and positive");
                }
            }
            standardize_ = true;
        }
        if (const json::Value* clip = pre->find("clip"); clip && !clip->is_null()) {
            if (!clip->is_array() || clip->as_array().size() != 2 || !clip->as_array()[0].is_number() ||
                !clip->as_array()[1].is_number()) {
                fail(Status::Format, "preprocess.clip must be [low, high]");
            }
            clip_lo_ = clip->as_array()[0].as_number();
            clip_hi_ = clip->as_array()[1].as_number();
            if (!(clip_lo_ < clip_hi_)) fail(Status::Format, "preprocess.clip needs low < high");
            clip_ = true;
        }
    }

    const FamilyFactory factory = Registry::instance().find(family_);
    if (!factory) {
        std::string known;
        for (const auto& name : Registry::instance().names()) known += (known.empty() ? "" : ", ") + name;
        fail(Status::Unsupported, "unknown model family \"" + family_ + "\" (registered: " + known + ")");
    }
    const json::Value* config = spec.find("config");
    if (config && !config->is_null() && !config->is_object()) fail(Status::Format, "\"config\" must be an object");
    const BuildContext context{archive_, spec, config && config->is_object() ? *config : empty_object(), features,
                               sequence};
    impl_ = factory(context);
    if (!impl_) fail(Status::Internal, "family \"" + family_ + "\" returned no model");
    shape_ = impl_->shape();
    if (shape_.features != features || shape_.sequence != sequence || shape_.outputs != output_size) {
        fail(Status::Shape, "family \"" + family_ + "\" built a model of features=" + std::to_string(shape_.features) +
                                ", sequence=" + std::to_string(shape_.sequence) +
                                ", outputs=" + std::to_string(shape_.outputs) + " but the spec declares features=" +
                                std::to_string(features) + ", sequence=" + std::to_string(sequence) +
                                ", outputs=" + std::to_string(output_size));
    }
}

std::shared_ptr<const Model> Model::build(std::shared_ptr<const Archive> archive) {
    return std::shared_ptr<const Model>(new Model(std::move(archive)));
}

std::shared_ptr<const Model> Model::open(const std::string& path) { return build(Archive::from_file(path)); }

std::shared_ptr<const Model> Model::from_bytes(const void* data, size_t size) {
    return build(Archive::from_bytes(data, size));
}

void Model::preprocess(InputType type, const void* x, size_t rows, double* out) const {
    const size_t features = shape_.features;
    const size_t count = rows * features;
    if (type == InputType::F32) {
        const float* in = static_cast<const float*>(x);
        for (size_t i = 0; i < count; ++i) out[i] = in[i];
    } else {
        const double* in = static_cast<const double*>(x);
        std::copy(in, in + count, out);
    }
    if (standardize_) {
        for (size_t r = 0; r < rows; ++r) {
            double* row = out + r * features;
            for (size_t f = 0; f < features; ++f) row[f] = (row[f] - mean_[f]) / scale_[f];
        }
    }
    if (clip_) {
        // NaN stays NaN, as with numpy.clip.
        for (size_t i = 0; i < count; ++i) out[i] = std::min(std::max(out[i], clip_lo_), clip_hi_);
    }
}

void Model::predict(InputType type, const void* x, size_t batch, double* out) const {
    if (batch == 0) return;
    if (!x || !out) fail(Status::InvalidArgument, "null input or output buffer");
    if (type != InputType::F32 && type != InputType::F64) fail(Status::InvalidArgument, "unknown input dtype");
    const size_t per_sample = shape_.steps() * shape_.features;
    const size_t element = type == InputType::F32 ? sizeof(float) : sizeof(double);
    const bool direct = type == InputType::F64 && !has_preprocess();

    auto run = [&](size_t begin, size_t end) {
        const auto* base = static_cast<const uint8_t*>(x) + begin * per_sample * element;
        if (direct) {
            impl_->predict(reinterpret_cast<const double*>(base), end - begin, out + begin * shape_.outputs);
            return;
        }
        thread_local std::vector<double> scratch;
        const size_t chunk = std::max<size_t>(1, kChunkValues / per_sample);
        scratch.resize(std::min(chunk, end - begin) * per_sample);
        for (size_t s = begin; s < end; s += chunk) {
            const size_t n = std::min(chunk, end - s);
            preprocess(type, base + (s - begin) * per_sample * element, n * shape_.steps(), scratch.data());
            impl_->predict(scratch.data(), n, out + s * shape_.outputs);
        }
    };

    const double cost = std::max(impl_->cost_per_sample(), 1.0);
    if (batch > 1 && cost * static_cast<double>(batch) >= kParallelCost && threads() > 1) {
        // At least two samples per task so batched kernels can pair them.
        const auto grain = static_cast<size_t>(std::max(2.0, std::ceil(kGrainCost / cost)));
        parallel_for(batch, grain, run);
    } else {
        run(0, batch);
    }
}

std::unique_ptr<Stream> Model::stream(size_t batch) const {
    if (batch == 0 || batch > kMaxExtent) fail(Status::InvalidArgument, "stream batch must be in [1, 2^24]");
    if (!shape_.streamable) fail(Status::Unsupported, "model family \"" + family_ + "\" is not streamable");
    return std::make_unique<Stream>(shared_from_this(), impl_->stream(batch), batch);
}

Stream::Stream(std::shared_ptr<const Model> model, std::unique_ptr<FamilyStream> impl, size_t batch)
    : model_(std::move(model)), impl_(std::move(impl)), batch_(batch) {
    if (!impl_) fail(Status::Internal, "family returned no stream");
    scratch_.resize(batch_ * model_->shape().features);
}

void Stream::step(InputType type, const void* x, double* out) {
    if (!x || !out) fail(Status::InvalidArgument, "null input or output buffer");
    if (type != InputType::F32 && type != InputType::F64) fail(Status::InvalidArgument, "unknown input dtype");
    model_->preprocess(type, x, batch_, scratch_.data());
    impl_->step(scratch_.data(), out);
}

void Stream::reset() { impl_->reset(); }

}  // namespace montlok::core
