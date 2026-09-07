// Synthetic benchmark of the stream feature kernels.
#include "stream_features.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>

static bool track_allocations = false;
static size_t event_allocations = 0;
void* operator new(std::size_t n) {
    if (track_allocations) ++event_allocations;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() {
    using namespace montlok::microstructure;
    constexpr size_t count = 32000, warmup = 12000;
    FeatureEngine engine(65536);
    std::vector<Event> events;
    events.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const double price = 100+0.001*static_cast<double>(i%127);
        const auto time = static_cast<int64_t>(i)*200000;
        if (i%3 == 0) events.push_back({time, 0, 0, price, price*1.0001, 0.5, 0});
        else if (i%3 == 1) events.push_back({time, 1, 0, price-0.01, price+0.01+(i%5)*0.01, 10.0+i%9, 11.0+i%7});
        else events.push_back({time, 2, i%2 ? 1 : -1, price, 0.1+0.01*(i%11), 0, 0});
    }
    std::array<double, kFeatureCount> features;
    std::array<double, kFeatureCount*4> weights;
    std::array<double, 4> bias{}, heads{};
    for (size_t i=0; i<weights.size(); ++i) weights[i] = static_cast<double>(i%13)*0.0001;
    for (size_t i=0; i<warmup; ++i) engine.update(events[i]);
    std::vector<double> durations(count-warmup);
    double checksum = 0;
    track_allocations = true;
    for (size_t i=warmup; i<count; ++i) {
        const auto start = std::chrono::steady_clock::now();
        engine.update(events[i]);
        if (!engine.snapshot(features)) return 2;
        linear_heads(features.data(), weights.data(), bias.data(), heads.size(), heads.data());
        for (double value : heads) checksum += value;
        const auto end = std::chrono::steady_clock::now();
        durations[i-warmup] = std::chrono::duration<double, std::nano>(end-start).count();
    }
    track_allocations = false;
    std::sort(durations.begin(), durations.end());
    const auto quantile = [&](double q) { return durations[static_cast<size_t>((durations.size()-1)*q)]/1000; };
    std::cout << std::setprecision(12)
              << "{\"fixture\":\"synthetic_engineering_only\",\"schema\":\"" << kSchema
              << "\",\"timed_events\":" << durations.size()
              << ",\"operation\":\"update_32_features_plus_4_affine_heads\""
              << ",\"p50_us\":" << quantile(0.50) << ",\"p95_us\":" << quantile(0.95)
              << ",\"p99_us\":" << quantile(0.99) << ",\"max_us\":" << durations.back()/1000
              << ",\"hot_path_cpp_heap_allocations\":" << event_allocations
              << ",\"checksum\":" << checksum << "}\n";
    return event_allocations ? 3 : 0;
}
