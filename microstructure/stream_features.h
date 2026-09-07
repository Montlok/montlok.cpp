// Incremental L1/trade stream features for microstructure research.
#pragma once

#include "../rdt4quant_cpu/montlok_kernel.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace montlok::microstructure {

constexpr size_t kFeatureCount = 32;
constexpr const char* kSchema = "xstock-observable-l1-v2-32";
constexpr int64_t kMs = 1000000;
constexpr std::array<int64_t, 3> kWindows = {100 * kMs, 500 * kMs, 2000 * kMs};

struct Sample {
    int64_t time;
    double value;
    double signed_volume = 0;
    double volume = 0;
    double count = 0;
};

// Capacity is fixed at construction. An overflow is visible to the caller;
// observations must never be silently overwritten inside a retained window.
class Ring {
public:
    explicit Ring(size_t capacity) : storage_(capacity) {
        if (capacity < 2) throw std::invalid_argument("capacity must be >= 2");
    }
    size_t size() const noexcept { return size_; }
    const Sample& at(size_t i) const noexcept { return storage_[(head_ + i) % storage_.size()]; }
    void clear() noexcept { head_ = size_ = 0; }
    void pop() noexcept { head_ = (head_ + 1) % storage_.size(); --size_; }
    void push(const Sample& sample) {
        if (size_ == storage_.size()) throw std::overflow_error("retained event capacity exceeded");
        storage_[(head_ + size_) % storage_.size()] = sample;
        ++size_;
    }
private:
    std::vector<Sample> storage_;
    size_t head_ = 0, size_ = 0;
};

class History {
public:
    explicit History(size_t capacity) : ring_(capacity) {}
    void clear() noexcept { ring_.clear(); }
    void expire(int64_t time) noexcept {
        // Keep the last observation at/before the longest lag, for causal as-of lookup.
        while (ring_.size() > 1 && ring_.at(1).time <= time - kWindows.back()) ring_.pop();
    }
    void push(int64_t time, double value) { expire(time); ring_.push({time, value}); }
    bool asof(int64_t time, double& value) const noexcept {
        size_t low = 0, high = ring_.size();
        while (low < high) {
            const size_t middle = low + (high - low) / 2;
            if (ring_.at(middle).time <= time) low = middle + 1;
            else high = middle;
        }
        if (low == 0) return false;
        value = ring_.at(low - 1).value;
        return true;
    }
private:
    Ring ring_;
};

class Window {
public:
    Window(size_t capacity, int64_t width) : ring_(capacity), width_(width) {}
    void clear() noexcept { ring_.clear(); ofi = signed_volume = volume = count = 0; }
    void expire(int64_t time) noexcept {
        // Flow windows are (t-W, t], price returns use as-of(t-W).
        while (ring_.size() && ring_.at(0).time <= time - width_) {
            const auto& x = ring_.at(0);
            ofi -= x.value; signed_volume -= x.signed_volume;
            volume -= x.volume; count -= x.count;
            ring_.pop();
        }
        if (!ring_.size()) ofi = signed_volume = volume = count = 0;
    }
    void push(const Sample& x) {
        expire(x.time); ring_.push(x);
        ofi += x.value; signed_volume += x.signed_volume;
        volume += x.volume; count += x.count;
    }
    double ofi = 0, signed_volume = 0, volume = 0, count = 0;
private:
    Ring ring_;
    int64_t width_;
};

struct Event {
    int64_t receive_ns; // one collector clock domain, monotonic, actual availability order
    int32_t kind;       // 0 reference, 1 L1 book, 2 execution message
    int32_t side;       // execution aggressor: +1 buy, -1 sell
    double a, b, c, d; // ref: raw price,FV,uncertainty,0; book: bid,ask,bidQty,askQty; trade: px,qty,0,0
};

inline double bps(double a, double b) noexcept { return 10000.0 * (a / b - 1.0); }

class FeatureEngine {
public:
    explicit FeatureEngine(size_t capacity)
        : reference_(capacity), depth_(capacity),
          windows_{Window(capacity, kWindows[0]), Window(capacity, kWindows[1]),
                   Window(capacity, kWindows[2])} {}

    void reset() noexcept {
        reference_.clear(); depth_.clear();
        for (auto& window : windows_) window.clear();
        signed_ema_.fill(0);
        now_ = ref_time_ = book_time_ = bid_time_ = ask_time_ = -1;
        refill_start_ = refill_finish_ = -1;
        ref_ = fair_ = uncertainty_ = bid_ = ask_ = bid_qty_ = ask_qty_ = 0;
        refill_target_ = refill_duration_ms_ = 0;
        faulted_ = false;
    }

    void update(const Event& e) {
        if (faulted_) throw std::logic_error("reset required after invalid input or overflow");
        try {
            validate(e);
            if (now_ >= 0) {
                for (size_t i = 0; i < kWindows.size(); ++i)
                    signed_ema_[i] *= std::exp(-static_cast<double>(e.receive_ns - now_) / kWindows[i]);
            }
            now_ = e.receive_ns;
            for (auto& window : windows_) window.expire(now_);
            reference_.expire(now_); depth_.expire(now_);
            if (e.kind == 0) {
                reference_.push(now_, e.a);
                ref_time_ = now_; ref_ = e.a; fair_ = e.b; uncertainty_ = e.c;
            } else if (e.kind == 1) {
                if (book_time_ >= 0) {
                    const double ofi = (e.a >= bid_ ? e.c : 0) - (e.a <= bid_ ? bid_qty_ : 0)
                                     - (e.b <= ask_ ? e.d : 0) + (e.b >= ask_ ? ask_qty_ : 0);
                    for (auto& window : windows_) window.push({now_, ofi});
                    const double spread = e.b - e.a;
                    if (refill_start_ < 0 && spread > ask_ - bid_ + 1e-10) {
                        refill_start_ = now_; refill_target_ = ask_ - bid_;
                    } else if (refill_start_ >= 0 && spread <= refill_target_ + 1e-10) {
                        refill_duration_ms_ = static_cast<double>(now_ - refill_start_) / kMs;
                        refill_finish_ = now_; refill_start_ = -1;
                    }
                }
                if (book_time_ < 0 || e.a != bid_) bid_time_ = now_;
                if (book_time_ < 0 || e.b != ask_) ask_time_ = now_;
                bid_ = e.a; ask_ = e.b; bid_qty_ = e.c; ask_qty_ = e.d; book_time_ = now_;
                depth_.push(now_, bid_qty_ + ask_qty_);
            } else {
                const double signed_volume = e.side * e.b;
                for (auto& window : windows_) window.push({now_, 0, signed_volume, e.b, 1});
                for (double& state : signed_ema_) state += signed_volume;
            }
        } catch (...) { faulted_ = true; throw; }
    }

    bool snapshot(std::array<double, kFeatureCount>& out) const noexcept {
        out.fill(0);
        if (faulted_ || ref_time_ < 0 || book_time_ < 0) return false;
        double r100, r200, r500, r2000, d100, d500;
        if (!reference_.asof(now_ - 100*kMs, r100) || !reference_.asof(now_ - 200*kMs, r200)
            || !reference_.asof(now_ - 500*kMs, r500) || !reference_.asof(now_ - 2000*kMs, r2000)
            || !depth_.asof(now_ - 100*kMs, d100) || !depth_.asof(now_ - 500*kMs, d500)) return false;
        const double mid = (bid_ + ask_) / 2, depth = bid_qty_ + ask_qty_;
        const double micro = (ask_ * bid_qty_ + bid_ * ask_qty_) / depth;
        out = {bps(mid, fair_), bps(fair_, ask_), bps(bid_, fair_), uncertainty_,
               bps(ref_, r100), bps(ref_, r500), bps(ref_, r2000), bps(ref_, r100)-bps(r100, r200),
               10000*(ask_-bid_)/mid, (bid_qty_-ask_qty_)/depth, bps(micro, mid),
               windows_[0].ofi/depth, windows_[1].ofi/depth,
               windows_[0].volume > 0 ? windows_[0].signed_volume/windows_[0].volume : 0,
               windows_[1].volume > 0 ? windows_[1].signed_volume/windows_[1].volume : 0,
               windows_[0].volume/depth, windows_[1].volume/depth,
               windows_[0].count, windows_[1].count,
               signed_ema_[0]/depth, signed_ema_[1]/depth, signed_ema_[2]/depth,
               depth/d100-1, depth/d500-1,
               static_cast<double>(now_-bid_time_)/kMs, static_cast<double>(now_-ask_time_)/kMs,
               static_cast<double>(now_-ref_time_)/kMs, static_cast<double>(now_-book_time_)/kMs,
               refill_duration_ms_, refill_finish_ >= 0 ? static_cast<double>(now_-refill_finish_)/kMs : 0,
               refill_start_ >= 0 ? static_cast<double>(now_-refill_start_)/kMs : 0,
               refill_finish_ >= 0 ? 1.0 : 0.0};
        for (double value : out) if (!std::isfinite(value)) { out.fill(0); return false; }
        return true;
    }

private:
    void validate(const Event& e) const {
        if (e.receive_ns < 0 || (now_ >= 0 && e.receive_ns < now_))
            throw std::invalid_argument("nonmonotonic receive time");
        if (!std::isfinite(e.a) || !std::isfinite(e.b) || !std::isfinite(e.c) || !std::isfinite(e.d))
            throw std::invalid_argument("nonfinite event");
        if (e.kind == 0) {
            if (!(e.a > 0 && e.b > 0 && e.c >= 0)) throw std::invalid_argument("invalid reference");
        } else if (e.kind == 1) {
            if (!(e.a > 0 && e.b >= e.a && e.c >= 0 && e.d >= 0 && e.c+e.d > 0))
                throw std::invalid_argument("invalid L1 book");
        } else if (e.kind == 2) {
            if (!(e.a > 0 && e.b > 0 && (e.side == 1 || e.side == -1)))
                throw std::invalid_argument("invalid execution");
        } else throw std::invalid_argument("unknown event kind");
    }
    History reference_, depth_;
    std::array<Window, 3> windows_;
    std::array<double, 3> signed_ema_{};
    int64_t now_ = -1, ref_time_ = -1, book_time_ = -1, bid_time_ = -1, ask_time_ = -1;
    int64_t refill_start_ = -1, refill_finish_ = -1;
    double ref_ = 0, fair_ = 0, uncertainty_ = 0, bid_ = 0, ask_ = 0, bid_qty_ = 0, ask_qty_ = 0;
    double refill_target_ = 0, refill_duration_ms_ = 0;
    bool faulted_ = false;
};

// Scalers are folded into affine coefficients offline. The raw values remain
// float64 so a threshold near a one-cent price change is not rounded early.
// This reuses Montlok's portable SIMD primitives, without a PyTorch thread pool.
inline void linear_heads(const double* features, const double* weights, const double* bias,
                         size_t heads, double* out) noexcept {
    for (size_t head = 0; head < heads; ++head) {
        const double* w = weights + head*kFeatureCount;
        double total = bias[head];
#ifdef MONTLOK_VECTOR_EXT
        v4d acc = splat4d(0);
        for (size_t i = 0; i < kFeatureCount; i += 4) acc += load4d(features+i)*load4d(w+i);
        total += hsum4d(acc);
#else
        for (size_t i = 0; i < kFeatureCount; ++i) total += features[i]*w[i];
#endif
        out[head] = total;
    }
}

} // namespace montlok::microstructure
