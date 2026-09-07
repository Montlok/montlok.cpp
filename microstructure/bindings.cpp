#include "stream_features.h"

using montlok::microstructure::Event;
using montlok::microstructure::FeatureEngine;
using montlok::microstructure::kFeatureCount;

extern "C" {
const char* montlok_micro_schema() noexcept { return montlok::microstructure::kSchema; }
size_t montlok_micro_event_size() noexcept { return sizeof(Event); }
void* montlok_micro_create(size_t capacity) noexcept {
    try { return new FeatureEngine(capacity); } catch (...) { return nullptr; }
}
void montlok_micro_destroy(void* ptr) noexcept { delete static_cast<FeatureEngine*>(ptr); }
void montlok_micro_reset(void* ptr) noexcept { if (ptr) static_cast<FeatureEngine*>(ptr)->reset(); }
// 0 success, -1 invalid event/overflow; reset is required after an error.
int montlok_micro_update(void* ptr, const Event* event) noexcept {
    if (!ptr || !event) return -1;
    try { static_cast<FeatureEngine*>(ptr)->update(*event); return 0; } catch (...) { return -1; }
}
// 1 ready, 0 warmup/fault; output must contain kFeatureCount doubles.
int montlok_micro_snapshot(void* ptr, double* out, size_t count) noexcept {
    if (!ptr || !out || count != kFeatureCount) return -1;
    std::array<double, kFeatureCount> result;
    const bool ready = static_cast<FeatureEngine*>(ptr)->snapshot(result);
    std::copy(result.begin(), result.end(), out);
    return ready ? 1 : 0;
}
int montlok_micro_linear(const double* x, const double* w, const double* bias,
                        size_t heads, double* out) noexcept {
    if (!x || !w || !bias || !out || heads == 0 || heads > 16) return -1;
    montlok::microstructure::linear_heads(x, w, bias, heads, out);
    return 0;
}
}
