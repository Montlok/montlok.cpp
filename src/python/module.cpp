// Copyright 2026 Montlok. Apache-2.0.
// montlok._core: the Python binding. Models go through the C ABI (montlok.hpp),
// like the Rust and C++ bindings.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "montlok/montlok.hpp"
#include "stream_features.h"

namespace py = pybind11;

namespace {

PyObject* g_format_error = nullptr;

void translate(const montlok::Error& e) {
    switch (e.status()) {
        case MONTLOK_ERROR_IO: PyErr_SetString(PyExc_OSError, e.what()); return;
        case MONTLOK_ERROR_UNSUPPORTED: PyErr_SetString(PyExc_NotImplementedError, e.what()); return;
        case MONTLOK_ERROR_FORMAT: PyErr_SetString(g_format_error, e.what()); return;
        case MONTLOK_ERROR_INVALID_ARGUMENT:
        case MONTLOK_ERROR_SHAPE: PyErr_SetString(PyExc_ValueError, e.what()); return;
        default: PyErr_SetString(PyExc_RuntimeError, e.what()); return;
    }
}

// A C-contiguous float32 or float64 view of `x`; other dtypes convert to
// float64.
py::array as_input(const py::handle& x) {
    py::array a = py::array::ensure(x);
    if (!a) throw py::type_error("input must be array-like");
    if (a.dtype().is(py::dtype::of<float>())) return py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(a);
    return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
}

bool is_f32(const py::array& a) { return a.dtype().is(py::dtype::of<float>()); }

// Calls with fewer input values keep the GIL: releasing and taking it back
// costs more than they do.
constexpr py::ssize_t kReleaseValues = 4096;

// The array results go to: `out` when given (a writable C-contiguous float64
// array of the result's size), otherwise a new array of `shape`.
py::object output(const py::object& out, std::vector<py::ssize_t> shape, double*& data) {
    py::ssize_t size = 1;
    for (const py::ssize_t extent : shape) size *= extent;
    if (out.is_none()) {
        py::array_t<double> fresh(shape);
        data = fresh.mutable_data();
        return std::move(fresh);
    }
    if (!py::isinstance<py::array>(out)) throw py::type_error("out must be a NumPy array");
    auto array = py::reinterpret_borrow<py::array>(out);
    if (!array.dtype().is(py::dtype::of<double>()) || !(array.flags() & py::array::c_style) || !array.writeable() ||
        array.size() != size) {
        throw py::value_error("out must be a writable C-contiguous float64 array of " + std::to_string(size) +
                              " values");
    }
    data = static_cast<double*>(array.mutable_data());
    return out;
}

py::object predict(const montlok::Model& model, const py::handle& x_in, const py::object& out_in) {
    const montlok::ModelInfo info = model.info();
    py::array x = as_input(x_in);
    const bool tabular = info.sequence == 0;
    const py::ssize_t rank_one = tabular ? 1 : 2;  // a single sample
    bool single = false;
    size_t batch = 0;
    if (x.ndim() == rank_one) {
        single = true;
        batch = 1;
    } else if (x.ndim() == rank_one + 1) {
        batch = static_cast<size_t>(x.shape(0));
    } else {
        throw py::value_error(tabular ? "expected [batch, features] or [features]"
                                      : "expected [batch, sequence, features] or [sequence, features]");
    }
    const py::ssize_t* shape = x.shape() + (single ? 0 : 1);
    if (tabular ? shape[0] != static_cast<py::ssize_t>(info.features)
                : (shape[0] != static_cast<py::ssize_t>(info.sequence) ||
                   shape[1] != static_cast<py::ssize_t>(info.features))) {
        throw py::value_error("input shape does not match the model: features=" + std::to_string(info.features) +
                              ", sequence=" + std::to_string(info.sequence));
    }
    const auto outputs = static_cast<py::ssize_t>(info.outputs);
    double* dst = nullptr;
    py::object out = output(out_in, single ? std::vector<py::ssize_t>{outputs}
                                           : std::vector<py::ssize_t>{static_cast<py::ssize_t>(batch), outputs},
                            dst);
    const void* src = x.data();
    const bool f32 = is_f32(x);
    if (x.size() < kReleaseValues) {
        if (f32) model.predict(static_cast<const float*>(src), batch, dst);
        else model.predict(static_cast<const double*>(src), batch, dst);
    } else {
        py::gil_scoped_release release;
        if (f32) model.predict(static_cast<const float*>(src), batch, dst);
        else model.predict(static_cast<const double*>(src), batch, dst);
    }
    return out;
}

// Per-call latency of the C ABI with the GIL released: `rows` holds input
// samples used in turn. Returns (nanoseconds of each timed call, mean
// nanoseconds of back-to-back calls).
py::tuple bench(const montlok::Model& model, const py::array_t<double, py::array::c_style | py::array::forcecast>& rows,
                size_t batch, size_t iterations, bool stream) {
    const montlok::ModelInfo info = model.info();
    const size_t width = batch * (stream ? info.features : info.input_size());
    if (batch == 0 || iterations == 0 || rows.ndim() != 2 || rows.shape(0) == 0 ||
        rows.shape(1) != static_cast<py::ssize_t>(width)) {
        throw py::value_error("rows must be [count, " + std::to_string(width) + "] and batch, iterations positive");
    }
    std::unique_ptr<montlok::Stream> live;
    if (stream) live = std::make_unique<montlok::Stream>(model.stream(batch));
    const size_t count = static_cast<size_t>(rows.shape(0));
    const double* x = rows.data();
    std::vector<double> out(batch * info.outputs);
    py::array_t<double> samples(static_cast<py::ssize_t>(iterations));
    double* timed = samples.mutable_data();
    double mean = 0;
    {
        py::gil_scoped_release release;
        using Clock = std::chrono::steady_clock;
        auto call = [&](size_t i) {
            const double* row = x + (i % count) * width;
            if (live) live->step(row, out.data());
            else model.predict(row, batch, out.data());
        };
        for (size_t i = 0; i < std::max<size_t>(1000, iterations / 10); ++i) call(i);
        for (size_t i = 0; i < iterations; ++i) {
            const auto start = Clock::now();
            call(i);
            timed[i] = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        }
        const auto start = Clock::now();
        for (size_t i = 0; i < iterations; ++i) call(i);
        mean = std::chrono::duration<double, std::nano>(Clock::now() - start).count() / static_cast<double>(iterations);
    }
    return py::make_tuple(std::move(samples), mean);
}

class PyStream {
public:
    PyStream(montlok::Stream stream, montlok::ModelInfo info) : stream_(std::move(stream)), info_(info) {}

    py::object step(const py::handle& x_in, const py::object& out_in) {
        py::array x = as_input(x_in);
        const size_t batch = stream_.batch();
        const bool single = x.ndim() == 1;
        if (!(single && batch == 1) &&
            !(x.ndim() == 2 && x.shape(0) == static_cast<py::ssize_t>(batch))) {
            throw py::value_error("expected [batch, features] (or [features] for a batch of 1)");
        }
        if (x.shape(x.ndim() - 1) != static_cast<py::ssize_t>(info_.features)) {
            throw py::value_error("expected " + std::to_string(info_.features) + " features");
        }
        const auto outputs = static_cast<py::ssize_t>(info_.outputs);
        double* dst = nullptr;
        py::object out = output(out_in, single ? std::vector<py::ssize_t>{outputs}
                                               : std::vector<py::ssize_t>{static_cast<py::ssize_t>(batch), outputs},
                                dst);
        const void* src = x.data();
        const bool f32 = is_f32(x);
        auto step = [&] {
            std::lock_guard<std::mutex> lock(mutex_);
            if (f32) stream_.step(static_cast<const float*>(src), dst);
            else stream_.step(static_cast<const double*>(src), dst);
        };
        if (x.size() < kReleaseValues) {
            step();
        } else {
            py::gil_scoped_release release;
            step();
        }
        return out;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_.reset();
    }

    size_t batch() const { return stream_.batch(); }

private:
    montlok::Stream stream_;
    montlok::ModelInfo info_;
    std::mutex mutex_;
};

// ---------------------------------------------------------------------------
// Microstructure features
// ---------------------------------------------------------------------------

using montlok::microstructure::Event;
using montlok::microstructure::FeatureEngine;
using montlok::microstructure::kFeatureCount;

class PyFeatureEngine {
public:
    explicit PyFeatureEngine(size_t capacity) : engine_(capacity) {}

    void update(int64_t receive_ns, int32_t kind, int32_t side, double a, double b, double c, double d) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            engine_.update(Event{receive_ns, kind, side, a, b, c, d});
        } catch (const std::exception& e) {
            throw py::value_error(std::string("event rejected, reset required: ") + e.what());
        }
    }

    py::object snapshot() {
        std::array<double, kFeatureCount> values;
        bool ready;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready = engine_.snapshot(values);
        }
        if (!ready) return py::none();
        return py::array_t<double>(static_cast<py::ssize_t>(kFeatureCount), values.data());
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        engine_.reset();
    }

    // Columns of N events -> (features [N, 32], ready [N]); row i is the
    // snapshot right after event i (zeros while warming up).
    py::tuple replay(py::array_t<int64_t, py::array::c_style | py::array::forcecast> receive_ns,
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> kind,
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> side,
                     py::array_t<double, py::array::c_style | py::array::forcecast> a,
                     py::array_t<double, py::array::c_style | py::array::forcecast> b,
                     py::array_t<double, py::array::c_style | py::array::forcecast> c,
                     py::array_t<double, py::array::c_style | py::array::forcecast> d) {
        const auto n = receive_ns.size();
        for (const py::ssize_t size : {kind.size(), side.size(), a.size(), b.size(), c.size(), d.size()}) {
            if (size != n) throw py::value_error("event columns must have equal length");
        }
        py::array_t<double> features({n, static_cast<py::ssize_t>(kFeatureCount)});
        py::array_t<bool> ready(n);
        double* f = features.mutable_data();
        bool* r = ready.mutable_data();
        const int64_t* t = receive_ns.data();
        const int32_t* k = kind.data();
        const int32_t* s = side.data();
        const double *pa = a.data(), *pb = b.data(), *pc = c.data(), *pd = d.data();
        py::ssize_t failed = -1;
        std::string why;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            std::array<double, kFeatureCount> values;
            for (py::ssize_t i = 0; i < n; ++i) {
                try {
                    engine_.update(Event{t[i], k[i], s[i], pa[i], pb[i], pc[i], pd[i]});
                } catch (const std::exception& e) {
                    failed = i;
                    why = e.what();
                    break;
                }
                r[i] = engine_.snapshot(values);
                std::copy(values.begin(), values.end(), f + i * static_cast<py::ssize_t>(kFeatureCount));
            }
        }
        if (failed >= 0) {
            throw py::value_error("event " + std::to_string(failed) + " rejected, reset required: " + why);
        }
        return py::make_tuple(features, ready);
    }

private:
    FeatureEngine engine_;
    std::mutex mutex_;
};

py::array_t<double> linear_heads(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> bias) {
    const auto heads = bias.size();
    if (x.size() != static_cast<py::ssize_t>(kFeatureCount) || heads < 1 || heads > 16 ||
        w.size() != heads * static_cast<py::ssize_t>(kFeatureCount)) {
        throw py::value_error("expected x[32], w[heads, 32], bias[heads] with 1 <= heads <= 16");
    }
    py::array_t<double> out(heads);
    montlok::microstructure::linear_heads(x.data(), w.data(), bias.data(), static_cast<size_t>(heads),
                                          out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "montlok runtime: model families, plugins and microstructure features";

    // A malformed or foreign model file; a ValueError subclass. The reference
    // lives as long as the process.
    g_format_error = PyErr_NewException("montlok._core.FormatError", PyExc_ValueError, nullptr);
    if (!g_format_error) throw py::error_already_set();
    m.attr("FormatError") = py::handle(g_format_error);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const montlok::Error& e) {
            translate(e);
        }
    });

    m.def("version", &montlok::version);
    m.def("abi_version", &montlok_abi_version);
    m.def("set_threads", &montlok::set_threads, py::arg("threads"),
          "Worker threads for large batches; 0 means one per hardware thread.");
    m.def("threads", &montlok::threads);
    m.def("set_latency_threads", &montlok::set_latency_threads, py::arg("threads"),
          "Threads that split a single prediction across cores; each spins on its core while it exists. "
          "0 stops them.");
    m.def("latency_threads", &montlok::latency_threads);
    m.def("families", [] {
        std::vector<std::string> names;
        std::string all = montlok_families();
        size_t start = 0;
        while (start < all.size()) {
            const size_t end = all.find(',', start);
            names.push_back(all.substr(start, end == std::string::npos ? std::string::npos : end - start));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return names;
    });
    m.def("load_plugin", &montlok::load_plugin, py::arg("path"));
    m.def("bench", &bench, py::arg("model"), py::arg("rows"), py::arg("batch") = 1, py::arg("iterations") = 20000,
          py::arg("stream") = false);

    py::class_<PyStream, std::unique_ptr<PyStream>>(m, "Stream")
        .def("step", &PyStream::step, py::arg("x"), py::arg("out") = py::none())
        .def("reset", &PyStream::reset)
        .def_property_readonly("batch", &PyStream::batch);

    py::class_<montlok::Model>(m, "Model")
        .def_static("open", &montlok::Model::open, py::arg("path"))
        .def_static(
            "from_bytes",
            [](const py::bytes& data) {
                const std::string_view view = data;
                return montlok::Model::from_memory(view.data(), view.size());
            },
            py::arg("data"))
        .def_property_readonly("family", &montlok::Model::family)
        .def_property_readonly("spec_json", &montlok::Model::spec)
        .def_property_readonly("features", [](const montlok::Model& self) { return self.info().features; })
        .def_property_readonly("sequence", [](const montlok::Model& self) { return self.info().sequence; })
        .def_property_readonly("outputs", [](const montlok::Model& self) { return self.info().outputs; })
        .def_property_readonly("streamable", [](const montlok::Model& self) { return self.info().streamable; })
        .def_property_readonly("preprocess", [](const montlok::Model& self) { return self.info().preprocess; })
        .def("predict", &predict, py::arg("x"), py::arg("out") = py::none())
        .def(
            "stream",
            [](const montlok::Model& self, size_t batch) {
                return std::make_unique<PyStream>(self.stream(batch), self.info());
            },
            py::arg("batch") = 1);

    py::module_ micro = m.def_submodule("micro", "Streaming L1/trade microstructure features");
    micro.attr("SCHEMA") = montlok::microstructure::kSchema;
    micro.attr("FEATURE_COUNT") = kFeatureCount;
    py::class_<PyFeatureEngine>(micro, "FeatureEngine")
        .def(py::init<size_t>(), py::arg("capacity") = 65536)
        .def("update", &PyFeatureEngine::update, py::arg("receive_ns"), py::arg("kind"), py::arg("side") = 0,
             py::arg("a") = 0.0, py::arg("b") = 0.0, py::arg("c") = 0.0, py::arg("d") = 0.0)
        .def("snapshot", &PyFeatureEngine::snapshot)
        .def("reset", &PyFeatureEngine::reset)
        .def("replay", &PyFeatureEngine::replay, py::arg("receive_ns"), py::arg("kind"), py::arg("side"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"));
    micro.def("linear_heads", &linear_heads, py::arg("x"), py::arg("weights"), py::arg("bias"));
}
