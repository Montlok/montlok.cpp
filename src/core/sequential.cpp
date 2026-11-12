// Copyright 2026 Montlok. Apache-2.0.
// The "sequential" family: a stack of layers over activations of shape
// [batch, steps, width]. It covers linear and logistic models, MLPs, GRU/LSTM
// encoders with any head, and pooling over a window.
//
// Config: {"precision": "f32" | "f64", "layers": [{"type": ...}, ...]}.
// Activations are held in double. With precision "f32" (PyTorch exports) the
// input and every operator's output are rounded to float32, as the float32
// module rounds them; each operator still accumulates in double. With "f64"
// (scikit-learn and other float64 models) values stay in double. Weights keep
// their stored values either way. Recurrent state follows the same rule.
#include <algorithm>
#include <cmath>
#include <map>

#include "kernels.hpp"
#include "runtime.hpp"

namespace montlok::core {
namespace {

struct Dims {
    size_t steps;
    size_t width;
};

// Per-thread scratch, grown on demand.
struct Workspace {
    std::vector<double> buffers[8];

    double* get(size_t index, size_t n) {
        std::vector<double>& v = buffers[index];
        if (v.size() < n) v.resize(n);
        return v.data();
    }
    static double* get(std::vector<double>& v, size_t n) {
        if (v.size() < n) v.resize(n);
        return v.data();
    }
};

// Rounding policy of a model.
struct Precision {
    bool f32 = true;
    double operator()(double v) const noexcept { return f32 ? static_cast<double>(static_cast<float>(v)) : v; }
    void apply(double* v, size_t n) const noexcept {
        if (!f32) return;
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<double>(static_cast<float>(v[i]));
    }
};

struct LayerContext {
    const Archive& archive;
    const json::Value& spec;
    Dims input;
    size_t index;
    Precision precision;

    std::string where() const { return "layer " + std::to_string(index) + " (" + json_string(spec, "type", "?") + ")"; }

    [[noreturn]] void error(Status status, const std::string& message) const { fail(status, where() + ": " + message); }

    std::string name(std::string_view key, bool required) const {
        const json::Value* v = spec.find(key);
        if (!v || v->is_null()) {
            if (required) error(Status::Format, "missing tensor name \"" + std::string(key) + "\"");
            return {};
        }
        if (!v->is_string()) error(Status::Format, "\"" + std::string(key) + "\" must name a tensor");
        return v->as_string();
    }
};

class Layer {
public:
    Layer(const LayerContext& c, Dims out) : in_(c.input), out_(out), q_(c.precision) {}
    virtual ~Layer() = default;

    Dims in() const noexcept { return in_; }
    Dims out() const noexcept { return out_; }

    // [batch, in.steps, in.width] -> [batch, out.steps, out.width]
    virtual void forward(const double* x, size_t batch, double* y, Workspace& ws) const = 0;
    // One time step: [batch, in.width] -> [batch, out.width]; state holds
    // batch * state_size() values.
    virtual void step(double* state, const double* x, size_t batch, double* y, Workspace& ws) const = 0;
    virtual bool streamable() const noexcept { return true; }
    virtual size_t state_size() const noexcept { return 0; }
    virtual double cost() const noexcept { return static_cast<double>(in_.steps * in_.width); }

protected:
    Dims in_, out_;
    Precision q_;
};

// A layer that maps every row (time step) independently.
class RowLayer : public Layer {
public:
    using Layer::Layer;
    void forward(const double* x, size_t batch, double* y, Workspace& ws) const override {
        apply(x, batch * in_.steps, y, ws);
    }
    void step(double*, const double* x, size_t batch, double* y, Workspace& ws) const override {
        apply(x, batch, y, ws);
    }

protected:
    virtual void apply(const double* x, size_t rows, double* y, Workspace& ws) const = 0;
};

struct Dense {
    size_t out = 0, in = 0;
    std::vector<float> w32;   // [out, in] when every weight is exactly a float (PyTorch exports)
    std::vector<double> w64;  // otherwise
    std::vector<double> bias; // [out] or empty

    void load(const LayerContext& c, const std::string& weight_name, const std::string& bias_name, size_t rows,
              size_t cols) {
        out = rows;
        in = cols;
        std::vector<double> w = c.archive.f64(weight_name, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
        const bool exact = std::all_of(w.begin(), w.end(), [](double v) {
            return static_cast<double>(static_cast<float>(v)) == v;
        });
        if (exact) w32.assign(w.begin(), w.end());
        else w64 = std::move(w);
        if (!bias_name.empty() && c.archive.find(bias_name)) bias = c.archive.f64(bias_name, {static_cast<int64_t>(rows)});
    }

    // y[i, :] = W x[i, :] + bias for m contiguous rows.
    void apply(const double* x, size_t m, double* y) const {
        const double* b = bias.empty() ? nullptr : bias.data();
        if (!w32.empty()) kernels::gemm(w32.data(), b, out, in, x, in, m, y, out);
        else kernels::gemm(w64.data(), b, out, in, x, in, m, y, out);
    }
};

// ---------------------------------------------------------------------------
// Row layers
// ---------------------------------------------------------------------------

class Linear final : public RowLayer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        const std::string weight = c.name("weight", true);
        const Tensor& t = c.archive.tensor(weight);
        if (t.shape.size() != 2) c.error(Status::Shape, "weight must be [out, in], got " + t.describe());
        const auto out = static_cast<size_t>(t.shape[0]);
        const auto in = static_cast<size_t>(t.shape[1]);
        if (in != c.input.width) {
            c.error(Status::Shape, "weight expects " + std::to_string(in) + " inputs but the previous layer gives " +
                                       std::to_string(c.input.width));
        }
        const std::string bias = c.name("bias", false);
        if (!bias.empty() && !c.archive.find(bias)) c.error(Status::Format, "bias tensor " + bias + " is missing");
        auto layer = std::unique_ptr<Linear>(new Linear(c, Dims{c.input.steps, out}));
        layer->dense_.load(c, weight, bias, out, in);
        return layer;
    }

    double cost() const noexcept override { return static_cast<double>(in_.steps * dense_.in * dense_.out); }

protected:
    using RowLayer::RowLayer;
    void apply(const double* x, size_t rows, double* y, Workspace&) const override {
        dense_.apply(x, rows, y);
        q_.apply(y, rows * dense_.out);
    }

private:
    Dense dense_;
};

class Norm final : public RowLayer {
public:
    // nn.LayerNorm (centered) or nn.RMSNorm.
    static std::unique_ptr<Layer> make(const LayerContext& c, bool rms) {
        auto layer = std::unique_ptr<Norm>(new Norm(c, c.input));
        layer->rms_ = rms;
        layer->eps_ = json_double(c.spec, "eps", rms ? 1.1920928955078125e-07 : 1e-5);
        if (!(layer->eps_ >= 0)) c.error(Status::Format, "eps must be non-negative");
        const auto width = static_cast<int64_t>(c.input.width);
        const std::string weight = c.name("weight", false);
        const std::string bias = c.name("bias", false);
        if (!weight.empty()) layer->weight_ = c.archive.f64(weight, {width});
        if (!bias.empty()) {
            if (rms) c.error(Status::Format, "rmsnorm has no bias");
            layer->bias_ = c.archive.f64(bias, {width});
        }
        return layer;
    }

protected:
    using RowLayer::RowLayer;
    void apply(const double* x, size_t rows, double* y, Workspace&) const override {
        const size_t n = in_.width;
        for (size_t r = 0; r < rows; ++r) {
            const double* row = x + r * n;
            double* dst = y + r * n;
            double mean = 0;
            if (!rms_) {
                for (size_t i = 0; i < n; ++i) mean += row[i];
                mean /= static_cast<double>(n);
            }
            double var = 0;
            for (size_t i = 0; i < n; ++i) {
                const double d = row[i] - mean;
                var += d * d;
            }
            var /= static_cast<double>(n);
            const double inv = 1.0 / std::sqrt(var + eps_);
            for (size_t i = 0; i < n; ++i) {
                double v = (row[i] - mean) * inv;
                if (!weight_.empty()) v *= weight_[i];
                if (!bias_.empty()) v += bias_[i];
                dst[i] = q_(v);
            }
        }
    }

private:
    bool rms_ = false;
    double eps_ = 1e-5;
    std::vector<double> weight_, bias_;
};

enum class Fn { Identity, Relu, LeakyRelu, Elu, Gelu, GeluTanh, Tanh, Sigmoid, Silu, Softplus };

bool parse_fn(const std::string& name, Fn& fn) noexcept {
    static const std::map<std::string, Fn> table = {
        {"identity", Fn::Identity}, {"relu", Fn::Relu},       {"leaky_relu", Fn::LeakyRelu},
        {"elu", Fn::Elu},           {"gelu", Fn::Gelu},       {"gelu_tanh", Fn::GeluTanh},
        {"tanh", Fn::Tanh},         {"sigmoid", Fn::Sigmoid}, {"silu", Fn::Silu},
        {"softplus", Fn::Softplus},
    };
    auto it = table.find(name);
    if (it == table.end()) return false;
    fn = it->second;
    return true;
}

class Activation final : public RowLayer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c, const std::string& fn_name) {
        Fn fn;
        if (!parse_fn(fn_name, fn)) c.error(Status::Unsupported, "unknown activation \"" + fn_name + "\"");
        auto layer = std::unique_ptr<Activation>(new Activation(c, c.input));
        layer->fn_ = fn;
        layer->alpha_ = json_double(c.spec, "alpha", fn == Fn::LeakyRelu ? 0.01 : 1.0);
        return layer;
    }

protected:
    using RowLayer::RowLayer;
    void apply(const double* x, size_t rows, double* y, Workspace&) const override {
        const size_t n = rows * in_.width;
        if (fn_ == Fn::Sigmoid || fn_ == Fn::Tanh || fn_ == Fn::Silu) {
            std::copy_n(x, n, y);
            if (fn_ == Fn::Tanh) kernels::tanh_n(y, n);
            else kernels::sigmoid_n(y, n);
            if (fn_ == Fn::Silu) {
                for (size_t i = 0; i < n; ++i) y[i] *= x[i];
            }
            q_.apply(y, n);
            return;
        }
        for (size_t i = 0; i < n; ++i) {
            const double v = x[i];
            double r = v;
            switch (fn_) {
                case Fn::Identity: break;
                case Fn::Relu: r = v > 0 ? v : 0.0; break;
                case Fn::LeakyRelu: r = kernels::leaky_relu(v, alpha_); break;
                case Fn::Elu: r = kernels::elu(v, alpha_); break;
                case Fn::Gelu: r = kernels::gelu(v); break;
                case Fn::GeluTanh: r = kernels::gelu_tanh(v); break;
                case Fn::Tanh:
                case Fn::Sigmoid:
                case Fn::Silu: break;  // vectorized above
                case Fn::Softplus: r = kernels::softplus(v); break;
            }
            y[i] = q_(r);
        }
    }

private:
    Fn fn_ = Fn::Identity;
    double alpha_ = 1.0;
};

class Softmax final : public RowLayer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        return std::unique_ptr<Layer>(new Softmax(c, c.input));
    }

protected:
    using RowLayer::RowLayer;
    void apply(const double* x, size_t rows, double* y, Workspace& ws) const override {
        const size_t n = in_.width;
        double* e = ws.get(0, n);
        for (size_t r = 0; r < rows; ++r) {
            const double* row = x + r * n;
            double peak = row[0];
            for (size_t i = 1; i < n; ++i) peak = std::max(peak, row[i]);
            for (size_t i = 0; i < n; ++i) e[i] = row[i] - peak;
            kernels::map4(e, n, [](kernels::v4d v) { return kernels::vexp(v); });
            double total = 0;
            for (size_t i = 0; i < n; ++i) total += e[i];
            for (size_t i = 0; i < n; ++i) y[r * n + i] = q_(e[i] / total);
        }
    }
};

// y = x * scale + shift, each a number or a [width] tensor.
class Affine final : public RowLayer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        auto layer = std::unique_ptr<Affine>(new Affine(c, c.input));
        layer->scale_ = read(c, "scale", 1.0);
        layer->shift_ = read(c, "shift", 0.0);
        return layer;
    }

protected:
    using RowLayer::RowLayer;
    void apply(const double* x, size_t rows, double* y, Workspace&) const override {
        const size_t n = in_.width;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < n; ++i) {
                const double s = scale_.size() == 1 ? scale_[0] : scale_[i];
                const double b = shift_.size() == 1 ? shift_[0] : shift_[i];
                y[r * n + i] = q_(x[r * n + i] * s + b);
            }
        }
    }

private:
    static std::vector<double> read(const LayerContext& c, std::string_view key, double fallback) {
        const json::Value* v = c.spec.find(key);
        if (!v || v->is_null()) return {fallback};
        if (v->is_number()) return {v->as_number()};
        if (v->is_string()) return c.archive.f64(v->as_string(), {static_cast<int64_t>(c.input.width)});
        c.error(Status::Format, "\"" + std::string(key) + "\" must be a number or a tensor name");
    }

    std::vector<double> scale_, shift_;
};

// ---------------------------------------------------------------------------
// Pooling over time
// ---------------------------------------------------------------------------

class Last final : public Layer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        return std::unique_ptr<Layer>(new Last(c, Dims{1, c.input.width}));
    }
    void forward(const double* x, size_t batch, double* y, Workspace&) const override {
        const size_t n = in_.width, steps = in_.steps;
        for (size_t b = 0; b < batch; ++b) std::copy_n(x + (b * steps + steps - 1) * n, n, y + b * n);
    }
    void step(double*, const double* x, size_t batch, double* y, Workspace&) const override {
        std::copy_n(x, batch * in_.width, y);
    }

private:
    using Layer::Layer;
};

class Mean final : public Layer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        return std::unique_ptr<Layer>(new Mean(c, Dims{1, c.input.width}));
    }
    void forward(const double* x, size_t batch, double* y, Workspace&) const override {
        const size_t n = in_.width, steps = in_.steps;
        for (size_t b = 0; b < batch; ++b) {
            double* dst = y + b * n;
            std::fill_n(dst, n, 0.0);
            for (size_t t = 0; t < steps; ++t) {
                const double* row = x + (b * steps + t) * n;
                for (size_t i = 0; i < n; ++i) dst[i] += row[i];
            }
            for (size_t i = 0; i < n; ++i) dst[i] = q_(dst[i] / static_cast<double>(steps));
        }
    }
    void step(double*, const double* x, size_t batch, double* y, Workspace&) const override {
        if (in_.steps != 1) fail(Status::Unsupported, "a \"mean\" layer needs the whole window; use predict");
        std::copy_n(x, batch * in_.width, y);
    }
    bool streamable() const noexcept override { return in_.steps == 1; }

private:
    using Layer::Layer;
};

class Flatten final : public Layer {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        return std::unique_ptr<Layer>(new Flatten(c, Dims{1, c.input.steps * c.input.width}));
    }
    void forward(const double* x, size_t batch, double* y, Workspace&) const override {
        std::copy_n(x, batch * in_.steps * in_.width, y);
    }
    void step(double*, const double* x, size_t batch, double* y, Workspace&) const override {
        if (in_.steps != 1) fail(Status::Unsupported, "a \"flatten\" layer needs the whole window; use predict");
        std::copy_n(x, batch * in_.width, y);
    }
    bool streamable() const noexcept override { return in_.steps == 1; }

private:
    using Layer::Layer;
};

// ---------------------------------------------------------------------------
// Recurrent layers (PyTorch nn.GRU / nn.LSTM, unidirectional, zero h0/c0)
// ---------------------------------------------------------------------------

class Recurrent : public Layer {
public:
    // Layer by layer over the whole sub-batch: one GEMM for the input
    // projections of every (sample, step), then per step one GEMM of the
    // recurrent weights against all samples' states.
    void forward(const double* x, size_t batch, double* y, Workspace& ws) const override {
        const size_t T = in_.steps, H = hidden_, G = gates_ * H;
        const double* src = x;
        for (size_t l = 0; l < layers_; ++l) {
            double* dst = l + 1 == layers_ ? y : ws.get(l % 2 == 0 ? 4 : 5, batch * T * H);
            double* gi = ws.get(0, batch * T * G);
            double* gh = ws.get(1, batch * G);
            double* scratch = ws.get(2, G + H);
            double* state = ws.get(3, batch * state_per_layer());
            std::fill_n(state, batch * state_per_layer(), 0.0);
            cells_[l].ih.apply(src, batch * T, gi);
            for (size_t t = 0; t < T; ++t) {
                cells_[l].hh.apply(state, batch, gh);  // h rows lead the per-layer state
                for (size_t b = 0; b < batch; ++b) {
                    gates(gi + (b * T + t) * G, gh + b * G, state, batch, b, scratch);
                    std::copy_n(state + b * H, H, dst + (b * T + t) * H);
                }
            }
            src = dst;
        }
    }

    // State: per layer [batch, H] of h (LSTM: then [batch, H] of c).
    void step(double* state, const double* x, size_t batch, double* y, Workspace& ws) const override {
        const size_t H = hidden_, G = gates_ * H;
        double* gi = ws.get(0, batch * G);
        double* gh = ws.get(1, batch * G);
        double* scratch = ws.get(2, G + H);
        const double* input = x;
        for (size_t l = 0; l < layers_; ++l) {
            double* s = state + l * batch * state_per_layer();
            cells_[l].ih.apply(input, batch, gi);
            cells_[l].hh.apply(s, batch, gh);
            for (size_t b = 0; b < batch; ++b) gates(gi + b * G, gh + b * G, s, batch, b, scratch);
            input = s;
        }
        std::copy_n(input, batch * H, y);
    }

    size_t state_size() const noexcept override { return layers_ * state_per_layer(); }

    double cost() const noexcept override {
        double total = 0;
        for (size_t l = 0; l < layers_; ++l) {
            const double in = l == 0 ? static_cast<double>(in_.width) : static_cast<double>(hidden_);
            total += static_cast<double>(gates_ * hidden_) * (in + static_cast<double>(hidden_));
        }
        return total * static_cast<double>(in_.steps);
    }

protected:
    struct Cell {
        Dense ih, hh;
    };

    Recurrent(const LayerContext& c, size_t gates)
        : Layer(c, Dims{c.input.steps, json_size(c.spec, "hidden", 0)}), gates_(gates) {
        hidden_ = out_.width;
        layers_ = json_size(c.spec, "layers", 1);
        if (hidden_ == 0 || hidden_ > 65536) c.error(Status::Format, "\"hidden\" must be in [1, 65536]");
        if (layers_ == 0 || layers_ > 64) c.error(Status::Format, "\"layers\" must be in [1, 64]");
        const std::string prefix = c.name("weights", true);
        for (size_t l = 0; l < layers_; ++l) {
            const std::string k = std::to_string(l);
            const size_t in = l == 0 ? c.input.width : hidden_;
            const Tensor* wih = c.archive.find(prefix + ".weight_ih_l" + k);
            if (!wih) c.error(Status::Format, "missing tensor " + prefix + ".weight_ih_l" + k);
            if (wih->shape.size() != 2 || static_cast<size_t>(wih->shape[0]) != gates_ * hidden_ ||
                static_cast<size_t>(wih->shape[1]) != in) {
                c.error(Status::Shape, wih->describe() + " must be [" + std::to_string(gates_ * hidden_) + ", " +
                                           std::to_string(in) + "] for this input and hidden size");
            }
            Cell cell;
            cell.ih.load(c, prefix + ".weight_ih_l" + k, prefix + ".bias_ih_l" + k, gates_ * hidden_, in);
            cell.hh.load(c, prefix + ".weight_hh_l" + k, prefix + ".bias_hh_l" + k, gates_ * hidden_, hidden_);
            cells_.push_back(std::move(cell));
        }
    }

    virtual size_t state_per_layer() const noexcept = 0;
    // Updates sample b of one layer's state from its gate pre-activations.
    virtual void gates(const double* gi, const double* gh, double* state, size_t batch, size_t b,
                       double* scratch) const = 0;

    size_t gates_;
    size_t hidden_ = 0, layers_ = 1;
    std::vector<Cell> cells_;
};

class Gru final : public Recurrent {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) { return std::unique_ptr<Layer>(new Gru(c)); }

protected:
    explicit Gru(const LayerContext& c) : Recurrent(c, 3) {}
    size_t state_per_layer() const noexcept override { return hidden_; }

    void gates(const double* gi, const double* gh, double* state, size_t, size_t b, double* s) const override {
        const size_t H = hidden_;
        double* h = state + b * H;
        for (size_t j = 0; j < 2 * H; ++j) s[j] = gi[j] + gh[j];
        kernels::sigmoid_n(s, 2 * H);  // r, z
        double* n = s + 2 * H;
        for (size_t j = 0; j < H; ++j) n[j] = gi[2 * H + j] + s[j] * gh[2 * H + j];
        kernels::tanh_n(n, H);
        for (size_t j = 0; j < H; ++j) h[j] = q_(n[j] + s[H + j] * (h[j] - n[j]));  // ATen: n + z * (h - n)
    }
};

class Lstm final : public Recurrent {
public:
    static std::unique_ptr<Layer> make(const LayerContext& c) {
        if (json_size(c.spec, "proj_size", 0) != 0) c.error(Status::Unsupported, "LSTM projections are not supported");
        return std::unique_ptr<Layer>(new Lstm(c));
    }

protected:
    explicit Lstm(const LayerContext& c) : Recurrent(c, 4) {}
    size_t state_per_layer() const noexcept override { return 2 * hidden_; }

    void gates(const double* gi, const double* gh, double* state, size_t batch, size_t b, double* s) const override {
        const size_t H = hidden_;
        double* h = state + b * H;
        double* c = state + batch * H + b * H;
        for (size_t j = 0; j < 4 * H; ++j) s[j] = gi[j] + gh[j];
        kernels::sigmoid_n(s, 2 * H);       // i, f
        kernels::tanh_n(s + 2 * H, H);      // g
        kernels::sigmoid_n(s + 3 * H, H);   // o
        double* tc = s + 4 * H;
        for (size_t j = 0; j < H; ++j) tc[j] = c[j] = q_(s[H + j] * c[j] + s[j] * s[2 * H + j]);
        kernels::tanh_n(tc, H);
        for (size_t j = 0; j < H; ++j) h[j] = q_(s[3 * H + j] * tc[j]);
    }
};

// ---------------------------------------------------------------------------
// Layer registry and the family
// ---------------------------------------------------------------------------

using LayerFactory = std::unique_ptr<Layer> (*)(const LayerContext&);

const std::map<std::string, LayerFactory>& layer_factories() {
    static const std::map<std::string, LayerFactory> table = {
        {"linear", &Linear::make},
        {"layernorm", [](const LayerContext& c) { return Norm::make(c, false); }},
        {"rmsnorm", [](const LayerContext& c) { return Norm::make(c, true); }},
        {"activation", [](const LayerContext& c) { return Activation::make(c, json_string(c.spec, "fn", "")); }},
        {"softmax", &Softmax::make},
        {"affine", &Affine::make},
        {"identity", [](const LayerContext& c) { return Activation::make(c, "identity"); }},
        {"last", &Last::make},
        {"mean", &Mean::make},
        {"flatten", &Flatten::make},
        {"gru", &Gru::make},
        {"lstm", &Lstm::make},
    };
    return table;
}

std::unique_ptr<Layer> make_layer(const LayerContext& c) {
    if (!c.spec.is_object()) c.error(Status::Format, "layer spec must be an object");
    const std::string type = json_string(c.spec, "type", "");
    Fn fn;
    if (parse_fn(type, fn)) return Activation::make(c, type);  // {"type": "relu"} shorthand
    const auto& table = layer_factories();
    auto it = table.find(type);
    if (it == table.end()) c.error(Status::Unsupported, "unknown layer type \"" + type + "\"");
    return it->second(c);
}

class Sequential;

class SequentialStream final : public FamilyStream {
public:
    SequentialStream(const Sequential& model, size_t batch);
    void step(const double* x, double* out) override;
    void reset() override;

private:
    const Sequential& model_;
    size_t batch_;
    std::vector<std::vector<double>> state_;
    Workspace ws_;
    std::vector<double> a_, b_;
};

class Sequential final : public FamilyModel {
public:
    Sequential(Shape shape, Precision precision, std::vector<std::unique_ptr<Layer>> layers)
        : shape_(shape), q_(precision), layers_(std::move(layers)) {
        max_values_ = shape_.steps() * shape_.features;
        max_width_ = shape_.features;
        for (const auto& layer : layers_) {
            max_values_ = std::max(max_values_, layer->out().steps * layer->out().width);
            max_width_ = std::max(max_width_, layer->out().width);
            cost_ += layer->cost();
        }
    }

    Shape shape() const override { return shape_; }
    double cost_per_sample() const override { return cost_; }

    void predict(const double* x, size_t batch, double* out) const override {
        thread_local Workspace ws;
        thread_local std::vector<double> ping, pong;
        const size_t per_sample = shape_.steps() * shape_.features;
        const size_t sub = std::max<size_t>(1, (size_t(1) << 17) / max_values_);
        for (size_t s = 0; s < batch; s += sub) {
            const size_t n = std::min(sub, batch - s);
            double* cur = Workspace::get(ping, n * max_values_);
            double* next = Workspace::get(pong, n * max_values_);
            std::copy_n(x + s * per_sample, n * per_sample, cur);
            q_.apply(cur, n * per_sample);
            for (const auto& layer : layers_) {
                layer->forward(cur, n, next, ws);
                std::swap(cur, next);
            }
            std::copy_n(cur, n * shape_.outputs, out + s * shape_.outputs);
        }
    }

    std::unique_ptr<FamilyStream> stream(size_t batch) const override {
        return std::make_unique<SequentialStream>(*this, batch);
    }

    const std::vector<std::unique_ptr<Layer>>& layers() const noexcept { return layers_; }
    size_t max_width() const noexcept { return max_width_; }
    const Precision& precision() const noexcept { return q_; }

private:
    Shape shape_;
    Precision q_;
    std::vector<std::unique_ptr<Layer>> layers_;
    size_t max_values_ = 0, max_width_ = 0;
    double cost_ = 0;
};

SequentialStream::SequentialStream(const Sequential& model, size_t batch) : model_(model), batch_(batch) {
    for (const auto& layer : model_.layers()) state_.emplace_back(batch_ * layer->state_size(), 0.0);
    a_.resize(batch_ * model_.max_width());
    b_.resize(batch_ * model_.max_width());
}

void SequentialStream::step(const double* x, double* out) {
    const Shape shape = model_.shape();
    std::copy_n(x, batch_ * shape.features, a_.data());
    model_.precision().apply(a_.data(), batch_ * shape.features);
    double* cur = a_.data();
    double* next = b_.data();
    const auto& layers = model_.layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->step(state_[i].data(), cur, batch_, next, ws_);
        std::swap(cur, next);
    }
    std::copy_n(cur, batch_ * shape.outputs, out);
}

void SequentialStream::reset() {
    for (auto& s : state_) std::fill(s.begin(), s.end(), 0.0);
}

std::unique_ptr<FamilyModel> build(const BuildContext& ctx) {
    const json::Value* layers = ctx.config.find("layers");
    if (!layers || !layers->is_array() || layers->as_array().empty()) {
        fail(Status::Format, "sequential config needs a non-empty \"layers\" array");
    }
    const std::string precision = json_string(ctx.config, "precision", "f32");
    if (precision != "f32" && precision != "f64") fail(Status::Format, "precision must be \"f32\" or \"f64\"");
    const Precision q{precision == "f32"};
    Dims dims{ctx.sequence == 0 ? 1 : ctx.sequence, ctx.features};
    std::vector<std::unique_ptr<Layer>> built;
    bool streamable = true;
    for (size_t i = 0; i < layers->as_array().size(); ++i) {
        const LayerContext c{*ctx.archive, layers->as_array()[i], dims, i, q};
        auto layer = make_layer(c);
        streamable = streamable && layer->streamable();
        dims = layer->out();
        built.push_back(std::move(layer));
    }
    if (dims.steps != 1) {
        fail(Status::Shape, "a sequential model must end with one step per sample; add a \"last\", \"mean\" or "
                            "\"flatten\" layer");
    }
    const Shape shape{ctx.features, ctx.sequence, dims.width, streamable};
    return std::make_unique<Sequential>(shape, q, std::move(built));
}

}  // namespace

void register_sequential(Registry& registry) { registry.add("sequential", &build, "builtin"); }

}  // namespace montlok::core
