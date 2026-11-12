// C ABI of the streaming microstructure features (declared in
// include/montlok/montlok.h). Self-contained: also builds on its own with
//   c++ -O3 -std=c++17 -shared -fPIC bindings.cpp -o libmontlok_micro.so
#include <array>
#include <cstddef>
#include <cstring>

#include "../include/montlok/montlok.h"
#include "stream_features.h"

using montlok::microstructure::Event;
using montlok::microstructure::FeatureEngine;
using montlok::microstructure::kFeatureCount;

static_assert(sizeof(montlok_micro_event) == sizeof(Event), "montlok_micro_event must mirror Event");
static_assert(offsetof(montlok_micro_event, receive_ns) == offsetof(Event, receive_ns), "event layout");
static_assert(offsetof(montlok_micro_event, kind) == offsetof(Event, kind), "event layout");
static_assert(offsetof(montlok_micro_event, side) == offsetof(Event, side), "event layout");
static_assert(offsetof(montlok_micro_event, a) == offsetof(Event, a), "event layout");
static_assert(offsetof(montlok_micro_event, d) == offsetof(Event, d), "event layout");

extern "C" {
const char* montlok_micro_schema(void) { return montlok::microstructure::kSchema; }
size_t montlok_micro_event_size(void) { return sizeof(Event); }
void* montlok_micro_create(size_t capacity) {
    try { return new FeatureEngine(capacity); } catch (...) { return nullptr; }
}
void montlok_micro_destroy(void* ptr) { delete static_cast<FeatureEngine*>(ptr); }
void montlok_micro_reset(void* ptr) { if (ptr) static_cast<FeatureEngine*>(ptr)->reset(); }
// 0 success, -1 invalid event/overflow; reset is required after an error.
int montlok_micro_update(void* ptr, const montlok_micro_event* event) {
    if (!ptr || !event) return -1;
    Event copy;
    std::memcpy(&copy, event, sizeof(copy));
    try { static_cast<FeatureEngine*>(ptr)->update(copy); return 0; } catch (...) { return -1; }
}
// 1 ready, 0 warmup/fault; output must contain kFeatureCount doubles.
int montlok_micro_snapshot(void* ptr, double* out, size_t count) {
    if (!ptr || !out || count != kFeatureCount) return -1;
    std::array<double, kFeatureCount> result;
    const bool ready = static_cast<FeatureEngine*>(ptr)->snapshot(result);
    std::copy(result.begin(), result.end(), out);
    return ready ? 1 : 0;
}
int montlok_micro_linear(const double* x, const double* w, const double* bias,
                         size_t heads, double* out) {
    if (!x || !w || !bias || !out || heads == 0 || heads > 16) return -1;
    montlok::microstructure::linear_heads(x, w, bias, heads, out);
    return 0;
}
}
