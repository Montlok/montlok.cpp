// Copyright 2026 Montlok. Apache-2.0.
// In-process latency of one model through the C ABI.
//
//   montlok_bench <model.montlok> [--batch B] [--iterations N] [--rows R] [--threads T]
//                                 [--latency-threads L]
//
// R distinct rows of standard normal inputs are used in turn, so every call
// sees a new sample. The measuring thread runs at the highest QoS on macOS.
// Prints JSON for predict and, on streamable models, stream step: the p50 and
// p99 of single calls and the mean of back-to-back calls. Single calls are
// quantized to the printed clock resolution.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "montlok/montlok.hpp"

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Summary {
    double p50 = 0, p99 = 0, mean = 0;
};

double nanoseconds(Clock::duration d) { return std::chrono::duration<double, std::nano>(d).count(); }

template <typename Call>
Summary measure(size_t iterations, size_t rows, Call&& call) {
    for (size_t i = 0; i < std::max<size_t>(2000, iterations / 10); ++i) call(i % rows);
    std::vector<double> samples(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        call(i % rows);
        samples[i] = nanoseconds(Clock::now() - start);
    }
    const auto start = Clock::now();
    for (size_t i = 0; i < iterations; ++i) call(i % rows);
    const double mean = nanoseconds(Clock::now() - start) / static_cast<double>(iterations);
    std::sort(samples.begin(), samples.end());
    return {samples[iterations / 2], samples[(iterations - 1) * 99 / 100], mean};
}

double clock_resolution() {
    double best = 1e9;
    for (int i = 0; i < 1000; ++i) {
        const auto a = Clock::now();
        auto b = Clock::now();
        while (b == a) b = Clock::now();
        best = std::min(best, nanoseconds(b - a));
    }
    return best;
}

void print(const char* name, const Summary& s, const char* tail) {
    std::printf("  \"%s\": {\"p50_ns\": %.0f, \"p99_ns\": %.0f, \"mean_ns\": %.1f}%s\n", name, s.p50, s.p99, s.mean, tail);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc % 2 != 0) {
        std::fprintf(stderr, "usage: montlok_bench <model.montlok> [--batch B] [--iterations N] [--rows R] "
                             "[--threads T] [--latency-threads L]\n");
        return 2;
    }
    size_t batch = 1, iterations = 20000, rows = 1024;
    int threads = 1, latency = 0;
    for (int i = 2; i < argc; i += 2) {
        const std::string flag = argv[i];
        const auto value = std::strtoull(argv[i + 1], nullptr, 10);
        if (flag == "--batch") batch = value;
        else if (flag == "--iterations") iterations = value;
        else if (flag == "--rows") rows = value;
        else if (flag == "--threads") threads = static_cast<int>(value);
        else if (flag == "--latency-threads") latency = static_cast<int>(value);
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (batch == 0 || iterations == 0 || rows == 0) {
        std::fprintf(stderr, "--batch, --iterations and --rows must be positive\n");
        return 2;
    }
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    try {
        montlok::set_threads(threads);
        montlok::set_latency_threads(latency);
        const montlok::Model model = montlok::Model::open(argv[1]);
        const montlok::ModelInfo info = model.info();
        const size_t width = batch * info.input_size();
        std::vector<double> x(rows * width);
        std::mt19937_64 rng(7);
        std::normal_distribution<double> normal;
        for (double& v : x) v = normal(rng);
        std::vector<double> out(batch * info.outputs);

        const Summary predict = measure(iterations, rows, [&](size_t r) {
            model.predict(x.data() + r * width, batch, out.data());
        });
        std::printf("{\n  \"family\": \"%s\",\n  \"features\": %u,\n  \"sequence\": %u,\n  \"batch\": %zu,\n"
                    "  \"threads\": %d,\n  \"latency_threads\": %d,\n  \"clock_resolution_ns\": %.1f,\n",
                    model.family().c_str(), info.features, info.sequence, batch, threads, latency, clock_resolution());
        print("predict", predict, info.streamable ? "," : "");
        if (info.streamable) {
            montlok::Stream stream = model.stream(batch);
            const size_t step = batch * info.features;
            const Summary s = measure(iterations, rows, [&](size_t r) { stream.step(x.data() + r * step, out.data()); });
            print("stream_step", s, "");
        }
        std::printf("}\n");
    } catch (const montlok::Error& e) {
        std::fprintf(stderr, "montlok_bench: %s\n", e.what());
        return 1;
    }
    return 0;
}
