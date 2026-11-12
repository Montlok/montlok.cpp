// Copyright 2026 Montlok. Apache-2.0.
// C++ consumer test: the fixtures of tests/data through montlok.hpp, the
// stream/predict identity, error reporting and (when built) the example plugin.
//
//   montlok_test_core <tests/data>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "montlok/montlok.hpp"

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("%-58s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

std::vector<double> read_f64(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<double> values(bytes.size() / sizeof(double));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(double));
    return values;
}

bool close(const std::vector<double>& a, const double* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(a[i]) != std::isnan(b[i])) return false;
        if (!std::isnan(a[i]) && std::fabs(a[i] - b[i]) > 1e-12 * std::max(1.0, std::fabs(b[i]))) return false;
    }
    return true;
}

void fixture(const std::string& dir, const std::string& name) {
    const size_t batch = 3;
    const montlok::Model model = montlok::Model::open(dir + "/" + name + ".montlok");
    const montlok::ModelInfo info = model.info();
    const std::vector<double> io = read_f64(dir + "/" + name + ".io");
    const size_t inputs = batch * info.input_size();
    expect(io.size() == inputs + batch * info.outputs, name + ": fixture size");
    const std::vector<double> x(io.begin(), io.begin() + static_cast<std::ptrdiff_t>(inputs));
    const std::vector<double> y = model.predict(x, batch);
    expect(close(y, io.data() + inputs, y.size()), name + ": predict matches the Python binding");

    if (info.streamable) {
        montlok::Stream stream = model.stream(1);
        std::vector<double> out(info.outputs);
        for (size_t t = 0; t < info.steps(); ++t) stream.step(x.data() + t * info.features, out.data());
        bool same = true;
        for (size_t k = 0; k < info.outputs; ++k) same = same && (out[k] == y[k] || (std::isnan(out[k]) && std::isnan(y[k])));
        expect(same, name + ": stream after the window equals predict");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "tests/data";
    std::printf("montlok %s, families: %s\n", montlok::version().c_str(), montlok_families());
    try {
        fixture(dir, "gru");
        fixture(dir, "gbr");
        montlok::set_latency_threads(3);
        expect(montlok::latency_threads() == 3, "latency threads start");
        fixture(dir, "gbr");
        montlok::set_latency_threads(0);
        expect(montlok::latency_threads() == 0, "latency threads stop");
    } catch (const montlok::Error& e) {
        expect(false, std::string("fixtures: ") + e.what());
    }
    expect(montlok_set_latency_threads(-1) == MONTLOK_ERROR_INVALID_ARGUMENT, "negative latency threads are rejected");

    try {
        const char garbage[] = "not a model";
        montlok::Model::from_memory(garbage, sizeof(garbage));
        expect(false, "garbage is rejected");
    } catch (const montlok::Error& e) {
        expect(e.status() == MONTLOK_ERROR_FORMAT, "garbage is rejected with MONTLOK_ERROR_FORMAT");
    }

    const montlok::Model model = montlok::Model::open(dir + "/gru.montlok");
    try {
        std::vector<double> wrong(7);
        model.predict(wrong, 1);
        expect(false, "wrong input size is rejected");
    } catch (const montlok::Error& e) {
        expect(e.status() == MONTLOK_ERROR_SHAPE, "wrong input size is rejected");
    }

#ifdef MONTLOK_TEST_PLUGIN
    try {
        montlok::load_plugin(MONTLOK_TEST_PLUGIN);
        expect(std::string(montlok_families()).find("ema_dot") != std::string::npos, "plugin family is registered");
    } catch (const montlok::Error& e) {
        expect(false, std::string("plugin: ") + e.what());
    }
#endif

    std::printf("%s\n", failures ? "FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
