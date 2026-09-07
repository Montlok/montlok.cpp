// Copyright 2026 OpenAI.
// Torch-free self-test and micro-benchmark for the montlok CPU kernels.
//
// Checks every kernel in montlok_kernel.h / montlok_mhc.h / montlok_layers.h
// against a naive double-precision implementation of the same mathematics on
// several shapes (model geometry, odd sizes that hit the scalar and
// partial-block paths, strided inputs) and then times the model shapes
// single-threaded. Exits non-zero on the first tolerance violation.
//
// Build (native):
//   clang++ -O3 -std=c++17 -march=native -fno-math-errno -ffp-contract=fast \
//       montlok_selftest.cpp -o montlok_selftest && ./montlok_selftest
// Cross-build for x86-64 AVX2/FMA from arm64 macOS (runs under Rosetta 2):
//   clang++ -O3 -std=c++17 -arch x86_64 -mavx2 -mfma -fno-math-errno -ffp-contract=fast \
//       montlok_selftest.cpp -o montlok_selftest_x86 && arch -x86_64 ./montlok_selftest_x86

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "montlok_kernel.h"
#include "montlok_layers.h"
#include "montlok_mhc.h"

namespace {

int g_failures = 0;

double seconds_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void check(const std::string& name, double max_err, double tolerance) {
    const bool ok = std::isfinite(max_err) && max_err <= tolerance;
    std::printf("%-48s max abs err %.3e (tol %.1e) %s\n", name.c_str(), max_err, tolerance, ok ? "ok" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

template <class T>
double max_abs_diff(const std::vector<T>& a, const std::vector<double>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return worst;
}

struct Rng {
    std::mt19937_64 gen;
    std::normal_distribution<double> normal{0.0, 1.0};
    explicit Rng(uint64_t seed) : gen(seed) {}
    float gauss(double scale = 1.0) { return static_cast<float>(normal(gen) * scale); }
    void fill(std::vector<float>& v, double scale = 1.0) {
        for (auto& x : v) {
            x = gauss(scale);
        }
    }
};

// ---------------------------------------------------------------------------
// Mamba-3 SISO recurrence
// ---------------------------------------------------------------------------

struct SisoCase {
    int64_t B, L, H, N, D;
    bool strided_v;
};

void test_siso(const SisoCase& c, bool timing) {
    Rng rng(7);
    const int64_t B = c.B, L = c.L, H = c.H, N = c.N, D = c.D;
    const int64_t v_head = c.strided_v ? 2 * D : D;  // pad the head stride to exercise Strides3
    std::vector<float> q(B * L * H * N), k(B * L * H * N), v(B * L * H * v_head), gate(B * L * H * D);
    std::vector<float> alpha(B * L * H), beta(B * L * H), gamma(B * L * H), skip(H), out(B * L * H * D, 0.0f);
    rng.fill(q, 0.5);
    rng.fill(k, 0.5);
    rng.fill(v);
    rng.fill(gate);
    rng.fill(skip);
    for (int64_t i = 0; i < B * L * H; ++i) {
        const float dt = montlok::softplus_f(rng.gauss() - 2.0f);
        const float a = -montlok::heavy_tail_f(rng.gauss());
        const montlok::StepCoefficients coef = montlok::step_coefficients(a * dt, dt, rng.gauss());
        alpha[i] = coef.alpha;
        beta[i] = coef.beta;
        gamma[i] = coef.gamma;
    }

    montlok::SisoPlan p{};
    p.B = B; p.L = L; p.H = H; p.N = N; p.D = D;
    p.q = q.data(); p.qs = {L * H * N, H * N, N};
    p.k = k.data(); p.ks = {L * H * N, H * N, N};
    p.v = v.data(); p.vs = {L * H * v_head, H * v_head, v_head};
    p.gate = gate.data(); p.gs = {L * H * D, H * D, D};
    p.alpha = alpha.data(); p.beta = beta.data(); p.gamma = gamma.data(); p.skip = skip.data();
    p.out = out.data(); p.os = {L * H * D, H * D, D};

    std::vector<float> scratch(montlok::siso_scratch_floats(p));
    auto run_all = [&]() {
        const int64_t items = montlok::siso_work_items(p);
        for (int64_t item = 0; item < items; ++item) {
            montlok::siso_run_work_item(p, item, scratch.data());
        }
    };
    run_all();

    // Naive double reference.
    std::vector<double> ref(B * L * H * D);
    std::vector<double> state(N * D);
    double absmax = 0.0;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            std::fill(state.begin(), state.end(), 0.0);
            for (int64_t t = 0; t < L; ++t) {
                const int64_t ci = (b * L + t) * H + h;
                const float* qt = q.data() + ci * N;
                const float* kt = k.data() + ci * N;
                const float* kp = kt - H * N;
                const float* vt = v.data() + ((b * L + t) * H + h) * v_head;
                const float* vp = vt - H * v_head;
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t d = 0; d < D; ++d) {
                        double s = alpha[ci] * state[n * D + d] + static_cast<double>(gamma[ci]) * kt[n] * vt[d];
                        if (t > 0) {
                            s += static_cast<double>(beta[ci]) * kp[n] * vp[d];
                        }
                        state[n * D + d] = s;
                    }
                }
                for (int64_t d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        acc += qt[n] * state[n * D + d];
                    }
                    const double y = (acc + static_cast<double>(skip[h]) * vt[d]) * gate[ci * D + d];
                    ref[ci * D + d] = y;
                    absmax = std::max(absmax, std::abs(y));
                }
            }
        }
    }
    char name[128];
    std::snprintf(name, sizeof(name), "siso B%lld L%lld H%lld N%lld D%lld%s (width %d)", (long long)B, (long long)L,
                  (long long)H, (long long)N, (long long)D, c.strided_v ? " strided" : "", montlok::siso_block_width(D));
    // fp32 recurrence over L steps vs double: allow ~1e-5 relative to the output scale.
    check(name, max_abs_diff(out, ref), 2e-6 * (1.0 + absmax));

    if (timing) {
        const int reps = 50;
        const double t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) {
            run_all();
        }
        const double dt = (seconds_now() - t0) / reps;
        std::printf("  timing: recurrence %.1f us/pass single-threaded (%lld work items)\n", dt * 1e6,
                    (long long)montlok::siso_work_items(p));
    }
}

// ---------------------------------------------------------------------------
// Fused pre-processing of the in_proj output
// ---------------------------------------------------------------------------

struct PrepCase {
    int64_t B, L, H, D, N, G, R;
};

void test_prep(const PrepCase& c, bool timing) {
    Rng rng(11);
    const int64_t B = c.B, L = c.L, H = c.H, D = c.D, N = c.N, G = c.G, R = c.R;
    const int64_t d_inner = H * D;
    const int64_t W = 2 * d_inner + 2 * G * N + 3 * H + R;
    std::vector<float> proj(B * L * W), b_bias(H * N), c_bias(H * N), b_norm_w(N), c_norm_w(N), dt_bias(H);
    rng.fill(proj);
    rng.fill(b_bias, 0.3);
    rng.fill(c_bias, 0.3);
    for (int64_t i = 0; i < N; ++i) {
        b_norm_w[i] = 1.0f + rng.gauss(0.2);
        c_norm_w[i] = 1.0f + rng.gauss(0.2);
    }
    rng.fill(dt_bias, 0.5);
    std::vector<float> q(B * L * H * N), k(B * L * H * N), alpha(B * L * H), beta(B * L * H), gamma(B * L * H);
    std::vector<float> angle_term(B * L * H * R), cum(B * H * R * L);

    montlok::PrepPlan p{};
    p.B = B; p.L = L; p.W = W; p.proj = proj.data(); p.proj_sb = L * W; p.proj_st = W;
    p.d_inner = d_inner; p.N = N; p.G = G; p.H = H; p.D = D; p.R = R;
    p.b_bias = b_bias.data(); p.c_bias = c_bias.data(); p.b_norm_w = b_norm_w.data(); p.c_norm_w = c_norm_w.data();
    p.b_eps = 1e-5f; p.c_eps = 1e-5f; p.dt_bias = dt_bias.data(); p.a_floor = 0.05f;
    p.q = q.data(); p.k = k.data(); p.alpha = alpha.data(); p.beta = beta.data(); p.gamma = gamma.data();
    p.angle_term = angle_term.data(); p.cum = cum.data();

    std::vector<float> scratch(montlok::prep_scratch_floats(p));
    auto run_all = [&]() {
        for (int64_t tok = 0; tok < B * L; ++tok) {
            montlok::prep_token(p, tok / L, tok % L, scratch.data());
        }
        for (int64_t chain = 0; chain < B * H * R; ++chain) {
            montlok::prep_chain(p, chain);
        }
        for (int64_t tok = 0; tok < B * L; ++tok) {
            montlok::prep_rotate(p, tok / L, tok % L);
        }
    };
    run_all();

    // Double reference, same formulas as Mamba3CPUReference.forward.
    std::vector<double> q_ref(B * L * H * N), k_ref(B * L * H * N), alpha_ref(B * L * H), beta_ref(B * L * H),
        gamma_ref(B * L * H);
    std::vector<double> bn(N), cn(N), running(H * R);
    const double pi = 3.14159265358979323846;
    for (int64_t b = 0; b < B; ++b) {
        std::fill(running.begin(), running.end(), 0.0);
        for (int64_t t = 0; t < L; ++t) {
            const float* row = proj.data() + (b * L + t) * W;
            const float* b_raw = row + 2 * d_inner;
            const float* c_raw = b_raw + G * N;
            const float* raw_dt = c_raw + G * N;
            const float* raw_a = raw_dt + H;
            const float* raw_trap = raw_a + H;
            const float* raw_angles = raw_trap + H;
            for (int64_t h = 0; h < H; ++h) {
                const int64_t g = h / (H / G);
                auto norm = [&](const float* x, const std::vector<float>& w, std::vector<double>& out_v) {
                    double ss = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        ss += static_cast<double>(x[n]) * x[n];
                    }
                    const double scale = 1.0 / std::sqrt(ss / N + 1e-5);
                    for (int64_t n = 0; n < N; ++n) {
                        out_v[n] = x[n] * scale * w[n];
                    }
                };
                norm(b_raw + g * N, b_norm_w, bn);
                norm(c_raw + g * N, c_norm_w, cn);
                const int64_t ci = (b * L + t) * H + h;
                for (int64_t n = 0; n < N; ++n) {
                    k_ref[ci * N + n] = bn[n] + b_bias[h * N + n];
                    q_ref[ci * N + n] = cn[n] + c_bias[h * N + n];
                }
                const double dt_pre = static_cast<double>(raw_dt[h]) + dt_bias[h];
                const double dt = dt_pre > 20.0 ? dt_pre : std::log1p(std::exp(dt_pre));
                const double ht = std::max(static_cast<double>(raw_a[h]), 0.0) +
                                  1.0 / (1.0 - std::min(static_cast<double>(raw_a[h]), 0.0));
                const double a = -std::max(ht, 0.05);
                const double al = std::exp(a * dt);
                const double trap = 1.0 / (1.0 + std::exp(-static_cast<double>(raw_trap[h])));
                alpha_ref[ci] = al;
                beta_ref[ci] = (1.0 - trap) * dt * al;
                gamma_ref[ci] = trap * dt;
                for (int64_t r = 0; r < R; ++r) {
                    running[h * R + r] += std::tanh(static_cast<double>(raw_angles[r])) * pi * dt;
                    const double angle = std::fmod(running[h * R + r], 2.0 * pi);
                    const double s = std::sin(angle), co = std::cos(angle);
                    double* qp = q_ref.data() + ci * N + 2 * r;
                    double* kp = k_ref.data() + ci * N + 2 * r;
                    const double q0 = qp[0], q1 = qp[1], k0 = kp[0], k1 = kp[1];
                    qp[0] = q0 * co - q1 * s;
                    qp[1] = q0 * s + q1 * co;
                    kp[0] = k0 * co - k1 * s;
                    kp[1] = k0 * s + k1 * co;
                }
            }
        }
    }
    char name[128];
    std::snprintf(name, sizeof(name), "prep B%lld L%lld H%lld D%lld N%lld G%lld R%lld", (long long)B, (long long)L,
                  (long long)H, (long long)D, (long long)N, (long long)G, (long long)R);
    // The reference keeps the cumulative angle in double while the kernel (like
    // PyTorch) wraps it modulo 2*pi in fp32, so rotated lanes carry ~ulp(cum) error.
    check(std::string(name) + " q", max_abs_diff(q, q_ref), 2e-3);
    check(std::string(name) + " k", max_abs_diff(k, k_ref), 2e-3);
    check(std::string(name) + " alpha", max_abs_diff(alpha, alpha_ref), 1e-6);
    check(std::string(name) + " beta", max_abs_diff(beta, beta_ref), 1e-6);
    check(std::string(name) + " gamma", max_abs_diff(gamma, gamma_ref), 1e-6);

    if (timing) {
        const int reps = 50;
        double t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int64_t tok = 0; tok < B * L; ++tok) {
                montlok::prep_token(p, tok / L, tok % L, scratch.data());
            }
        }
        const double a_us = (seconds_now() - t0) / reps * 1e6;
        t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int64_t chain = 0; chain < B * H * R; ++chain) {
                montlok::prep_chain(p, chain);
            }
        }
        const double b_us = (seconds_now() - t0) / reps * 1e6;
        t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int64_t tok = 0; tok < B * L; ++tok) {
                montlok::prep_rotate(p, tok / L, tok % L);
            }
        }
        const double c_us = (seconds_now() - t0) / reps * 1e6;
        std::printf("  timing: prep phase A %.1f us, B %.1f us, C %.1f us per pass single-threaded\n", a_us, b_us, c_us);
    }
}

// Vectorised sincos against libm in double over the wrapped-angle range,
// including the slightly-negative / slightly-above-2pi values fp32 wrapping
// can produce.
void test_sincos() {
    std::mt19937_64 gen(5);
    std::uniform_real_distribution<double> uni(-0.01, 2.0 * 3.14159265358979323846 + 0.01);
    double worst = 0.0;
    int64_t float_mismatch = 0;
    const int64_t count = 1 << 18;
    for (int64_t i = 0; i < count; i += 4) {
        float angles[4];
        float sines[4];
        float cosines[4];
        for (int lane = 0; lane < 4; ++lane) {
            angles[lane] = static_cast<float>(uni(gen));
        }
        montlok::sincos_angles(angles, 4, sines, cosines);
        for (int lane = 0; lane < 4; ++lane) {
            const double x = static_cast<double>(angles[lane]);
            const float sin_ref = static_cast<float>(std::sin(x));
            const float cos_ref = static_cast<float>(std::cos(x));
            worst = std::max(worst, std::abs(static_cast<double>(sines[lane]) - std::sin(x)));
            worst = std::max(worst, std::abs(static_cast<double>(cosines[lane]) - std::cos(x)));
            float_mismatch += (sines[lane] != sin_ref) + (cosines[lane] != cos_ref);
        }
    }
    check("sincos (double poly vs libm, fp32-rounded)", worst, 6e-8);
    std::printf("  %lld of %lld fp32 results differ from libm's rounding\n", (long long)float_mismatch, (long long)(2 * count));
    if (float_mismatch > count / 1000) {
        std::printf("  too many mismatches: FAIL\n");
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Manifold hyper-connection
// ---------------------------------------------------------------------------

struct MhcCase {
    int64_t T, n, d;
    int iters;
};

void test_mhc(const MhcCase& c, bool timing) {
    Rng rng(23);
    const int64_t T = c.T, n = c.n, d = c.d;
    const int64_t rows = montlok::mhc_proj_rows(n);
    std::vector<float> streams(T * n * d), norm_w(d), pre_bias(n), post_bias(n), res_bias(n * n), out(T * d);
    std::vector<double> proj(rows * d);
    rng.fill(streams, 1.5);
    rng.fill(pre_bias, 0.5);
    rng.fill(post_bias, 0.5);
    rng.fill(res_bias, 1.0);
    rng.fill(out);
    for (int64_t i = 0; i < n; ++i) {
        res_bias[i * n + i] += 2.0f;
    }
    for (auto& w : norm_w) {
        w = 0.5f + static_cast<float>(std::abs(rng.gauss()));
    }
    for (auto& w : proj) {
        w = static_cast<float>(rng.gauss(4.0 / std::sqrt(static_cast<double>(d))));  // fp32-representable like the model
    }
    std::vector<float> agg(T * d), pre(T * n), post(T * n), res(T * n * n), new_streams(T * n * d);

    montlok::MhcPlan p{};
    p.streams = streams.data(); p.norm_w = norm_w.data(); p.proj_w = proj.data();
    p.pre_bias = pre_bias.data(); p.post_bias = post_bias.data(); p.res_bias = res_bias.data();
    p.pre_alpha = 0.7; p.post_alpha = 0.6; p.res_alpha = 0.8; p.eps = static_cast<double>(1e-6f);
    p.tokens = T; p.n = n; p.d = d; p.sinkhorn_iters = c.iters;
    p.agg = agg.data(); p.pre = pre.data(); p.post = post.data(); p.res = res.data();
    std::vector<double> scratch(montlok::mhc_scratch_doubles(p));
    auto run_prepare = [&]() { montlok::mhc_prepare_range(p, 0, T, scratch.data()); };
    auto run_combine = [&]() {
        for (int64_t t = 0; t < T; ++t) {
            montlok::mhc_combine_token(streams.data() + t * n * d, res.data() + t * n * n, post.data() + t * n,
                                       out.data() + t * d, new_streams.data() + t * n * d, n, d);
        }
    };
    run_prepare();
    run_combine();

    // Naive double reference, Sinkhorn in log space exactly like Model/layers/mhc.py.
    std::vector<double> agg_ref(T * d), pre_ref(T * n), post_ref(T * n), res_ref(T * n * n), new_ref(T * n * d);
    std::vector<double> ref(d), logits(rows), work(n * n);
    for (int64_t t = 0; t < T; ++t) {
        const float* s = streams.data() + t * n * d;
        double ss = 0.0;
        for (int64_t k = 0; k < d; ++k) {
            double m = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                m += s[i * d + k];
            }
            m /= n;
            ref[k] = m;
            ss += m * m;
        }
        const double scale = 1.0 / std::sqrt(ss / d + p.eps);
        for (int64_t k = 0; k < d; ++k) {
            ref[k] *= scale * norm_w[k];
        }
        for (int64_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (int64_t k = 0; k < d; ++k) {
                acc += proj[r * d + k] * ref[k];
            }
            logits[r] = std::tanh(acc);
        }
        for (int64_t i = 0; i < n; ++i) {
            pre_ref[t * n + i] = 1.0 / (1.0 + std::exp(-(pre_bias[i] + p.pre_alpha * logits[i])));
            post_ref[t * n + i] = 2.0 / (1.0 + std::exp(-(post_bias[i] + p.post_alpha * logits[n + i])));
        }
        for (int64_t e = 0; e < n * n; ++e) {
            work[e] = res_bias[e] + p.res_alpha * logits[2 * n + e];
        }
        for (int it = 0; it < std::max(c.iters, 1); ++it) {
            for (int64_t i = 0; i < n; ++i) {
                double mx = -INFINITY;
                for (int64_t j = 0; j < n; ++j) mx = std::max(mx, work[i * n + j]);
                double sum = 0.0;
                for (int64_t j = 0; j < n; ++j) sum += std::exp(work[i * n + j] - mx);
                const double lse = mx + std::log(sum);
                for (int64_t j = 0; j < n; ++j) work[i * n + j] -= lse;
            }
            for (int64_t j = 0; j < n; ++j) {
                double mx = -INFINITY;
                for (int64_t i = 0; i < n; ++i) mx = std::max(mx, work[i * n + j]);
                double sum = 0.0;
                for (int64_t i = 0; i < n; ++i) sum += std::exp(work[i * n + j] - mx);
                const double lse = mx + std::log(sum);
                for (int64_t i = 0; i < n; ++i) work[i * n + j] -= lse;
            }
        }
        for (int64_t e = 0; e < n * n; ++e) {
            res_ref[t * n * n + e] = std::exp(work[e]);
        }
        for (int64_t k = 0; k < d; ++k) {
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                acc += static_cast<double>(pre[t * n + i]) * s[i * d + k];  // uses the kernel's fp32 pre like the model
            }
            agg_ref[t * d + k] = acc;
        }
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t k = 0; k < d; ++k) {
                double acc = static_cast<double>(post[t * n + i]) * out[t * d + k];
                for (int64_t j = 0; j < n; ++j) {
                    acc += static_cast<double>(res[t * n * n + i * n + j]) * s[j * d + k];
                }
                new_ref[(t * n + i) * d + k] = acc;
            }
        }
    }
    char name[128];
    std::snprintf(name, sizeof(name), "mhc T%lld n%lld d%lld iters%d", (long long)T, (long long)n, (long long)d, c.iters);
    check(std::string(name) + " pre", max_abs_diff(pre, pre_ref), 2e-7);
    check(std::string(name) + " post", max_abs_diff(post, post_ref), 4e-7);
    check(std::string(name) + " res", max_abs_diff(res, res_ref), 2e-7);
    double agg_max = 0.0, new_max = 0.0;
    for (double x : agg_ref) agg_max = std::max(agg_max, std::abs(x));
    for (double x : new_ref) new_max = std::max(new_max, std::abs(x));
    check(std::string(name) + " agg", max_abs_diff(agg, agg_ref), 1.2e-7 * (1.0 + agg_max));
    check(std::string(name) + " combine", max_abs_diff(new_streams, new_ref), 1.2e-7 * (1.0 + new_max));

    if (timing) {
        const int reps = 200;
        double t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) run_prepare();
        const double prep_us = (seconds_now() - t0) / reps * 1e6;
        t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) run_combine();
        const double comb_us = (seconds_now() - t0) / reps * 1e6;
        std::printf("  timing: mhc prepare %.1f us/pass, combine %.1f us/pass single-threaded\n", prep_us, comb_us);
    }
}

// ---------------------------------------------------------------------------
// RMSNorm / MLA rope
// ---------------------------------------------------------------------------

void test_rms_norm(int64_t rows, int64_t dim, int64_t groups, int64_t row_stride, bool timing) {
    Rng rng(31);
    std::vector<float> x(rows * row_stride), w(dim), out(rows * dim);
    rng.fill(x, 2.0);
    for (auto& v : w) v = 1.0f + rng.gauss(0.3);
    const double eps = static_cast<double>(1e-6f);
    montlok::rms_norm_rows(x.data(), row_stride, w.data(), eps, dim, groups, 0, rows, out.data());
    std::vector<double> ref(rows * dim);
    const int64_t gs = dim / groups;
    double absmax = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < groups; ++g) {
            double ss = 0.0;
            for (int64_t k = 0; k < gs; ++k) {
                const double v = x[r * row_stride + g * gs + k];
                ss += v * v;
            }
            const double scale = 1.0 / std::sqrt(ss / gs + eps);
            for (int64_t k = 0; k < gs; ++k) {
                const double y = x[r * row_stride + g * gs + k] * scale * w[g * gs + k];
                ref[r * dim + g * gs + k] = y;
                absmax = std::max(absmax, std::abs(y));
            }
        }
    }
    char name[128];
    std::snprintf(name, sizeof(name), "rms_norm rows%lld dim%lld groups%lld stride%lld", (long long)rows, (long long)dim,
                  (long long)groups, (long long)row_stride);
    check(name, max_abs_diff(out, ref), 1.2e-7 * (1.0 + absmax));
    if (timing) {
        const int reps = 500;
        const double t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) {
            montlok::rms_norm_rows(x.data(), row_stride, w.data(), eps, dim, groups, 0, rows, out.data());
        }
        std::printf("  timing: rms_norm %.1f us/pass single-threaded\n", (seconds_now() - t0) / reps * 1e6);
    }
}

void test_mla_rope(int64_t tokens, int64_t length, int64_t heads, int64_t nope, int64_t rope, bool timing) {
    Rng rng(41);
    const int64_t head_dim = nope + rope;
    std::vector<float> q(tokens * heads * head_dim), kv(tokens * heads * (nope + head_dim)), k_rope(tokens * rope);
    std::vector<float> cosine(length * rope), sine(length * rope), k(tokens * heads * head_dim), scratch(rope);
    rng.fill(q);
    rng.fill(kv);
    rng.fill(k_rope);
    for (int64_t pos = 0; pos < length; ++pos) {
        for (int64_t i = 0; i < rope / 2; ++i) {
            const double angle = pos * std::pow(10000.0, -2.0 * i / rope);
            cosine[pos * rope + i] = cosine[pos * rope + i + rope / 2] = static_cast<float>(std::cos(angle));
            sine[pos * rope + i] = sine[pos * rope + i + rope / 2] = static_cast<float>(std::sin(angle));
        }
    }
    std::vector<float> q_in = q;
    montlok::MlaRopePlan p{};
    p.q = q.data(); p.q_token_stride = heads * head_dim; p.q_head_stride = head_dim;
    p.kv = kv.data(); p.kv_token_stride = heads * (nope + head_dim); p.kv_head_stride = nope + head_dim;
    p.k_rope = k_rope.data(); p.k_rope_token_stride = rope;
    p.cosine = cosine.data(); p.sine = sine.data(); p.k = k.data();
    p.length = length; p.heads = heads; p.head_dim = head_dim; p.nope_dim = nope; p.rope_dim = rope;
    for (int64_t t = 0; t < tokens; ++t) {
        montlok::mla_rope_token(p, t, scratch.data());
    }
    std::vector<double> q_ref(q.size()), k_ref(k.size());
    auto rope_ref = [&](const float* x, int64_t pos, double* out_v) {
        const int64_t half = rope / 2;
        for (int64_t i = 0; i < half; ++i) {
            out_v[i] = x[i] * static_cast<double>(cosine[pos * rope + i]) - x[i + half] * static_cast<double>(sine[pos * rope + i]);
            out_v[i + half] = x[i + half] * static_cast<double>(cosine[pos * rope + i + half]) +
                              x[i] * static_cast<double>(sine[pos * rope + i + half]);
        }
    };
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t pos = t % length;
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t base = (t * heads + h) * head_dim;
            for (int64_t i = 0; i < nope; ++i) {
                q_ref[base + i] = q_in[base + i];
                k_ref[base + i] = kv[(t * heads + h) * (nope + head_dim) + i];
            }
            rope_ref(q_in.data() + base + nope, pos, q_ref.data() + base + nope);
            rope_ref(k_rope.data() + t * rope, pos, k_ref.data() + base + nope);
        }
    }
    char name[128];
    std::snprintf(name, sizeof(name), "mla_rope tokens%lld heads%lld nope%lld rope%lld", (long long)tokens, (long long)heads,
                  (long long)nope, (long long)rope);
    check(std::string(name) + " q", max_abs_diff(q, q_ref), 6e-7);
    check(std::string(name) + " k", max_abs_diff(k, k_ref), 6e-7);
    if (timing) {
        const int reps = 500;
        const double t0 = seconds_now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int64_t t = 0; t < tokens; ++t) {
                montlok::mla_rope_token(p, t, scratch.data());
            }
        }
        std::printf("  timing: mla_rope %.1f us/pass single-threaded\n", (seconds_now() - t0) / reps * 1e6);
    }
}

}  // namespace

int main() {
    std::printf("montlok selftest:");
#if defined(__AVX2__)
    std::printf(" avx2");
#endif
#if defined(__FMA__)
    std::printf(" fma");
#endif
#if defined(__AVX512F__)
    std::printf(" avx512f");
#endif
#if defined(__aarch64__)
    std::printf(" aarch64");
#endif
#ifdef MONTLOK_VECTOR_EXT
    std::printf(" vector-ext");
#endif
    std::printf("\n");

    // Model geometry first (timed), then odd shapes for the other paths.
    test_siso({1, 480, 8, 64, 64, false}, true);
    test_siso({2, 37, 3, 14, 8, true}, false);     // 8-lane blocks, strided v
    test_siso({1, 20, 2, 5, 12, false}, false);    // scalar path (headdim not a multiple of 8)
    test_siso({1, 1, 2, 4, 16, false}, false);     // single step
    test_prep({1, 480, 8, 64, 64, 1, 16}, true);
    test_prep({2, 37, 6, 8, 14, 3, 5}, false);     // grouped B/C, odd sizes
    test_sincos();
    test_mhc({480, 4, 256, 20}, true);
    test_mhc({37, 4, 256, 20}, false);             // partial token block
    test_mhc({50, 3, 100, 5}, false);
    test_mhc({17, 5, 33, 0}, false);               // iters=0 still runs one Sinkhorn step
    test_mhc({9, 1, 16, 3}, false);
    test_mhc({6, 9, 20, 2}, false);                // n > 8: generic combine path
    test_rms_norm(480, 256, 1, 256, true);
    test_rms_norm(480, 256, 8, 256, false);        // GroupedRMSNorm as used by Mamba3Layer
    test_rms_norm(37, 100, 4, 133, false);         // odd group size, strided rows
    test_mla_rope(480, 480, 4, 32, 32, true);
    test_mla_rope(74, 37, 3, 12, 10, false);       // batch 2, odd dims

    if (g_failures) {
        std::printf("%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
