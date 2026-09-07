// Copyright 2026 OpenAI.
// Apache-2.0 weight-compatible CPU kernels for Mamba-3 SISO.
//
// This header is torch-free so the same code can be compiled into the PyTorch
// extension (montlok.cpp) and into the standalone self-test
// (montlok_selftest.cpp). The recurrence follows the public state-spaces/mamba
// reference equations; the pre-processing mirrors Mamba3CPUReference.forward in
// mamba3_cpu.py operation by operation so the fused path stays weight- and
// numerics-compatible with the pure PyTorch reference.
//
// Performance design (AVX2 + FMA on x86-64, NEON on arm64):
//   * The SSM state for one (batch, head) is [d_state, headdim]. Rows of the
//     state along headdim are independent, so a work item is (batch, head,
//     16-lane headdim block): B*H*headdim/16 items instead of B*H (32 items
//     for the [1, 480, 8, 64] model shape).
//   * Inside a work item the state is stored [d_state][block] and processed
//     with 256-bit GCC/Clang vector extensions (one AVX2 ymm register per 8
//     floats, two NEON registers on arm64). Every 8 state elements cost one
//     multiply and three FMAs, which is the arithmetic minimum for the
//     trapezoidal update; k/q enter as broadcast scalars so there is no
//     horizontal reduction.
//   * Two independent accumulators hide the 5-cycle FMA latency of the
//     q . state reduction across the d_state loop.
//   * The state block (d_state * 16 * 4 = 4 KiB for d_state = 64) lives in L1.
//   * FTZ/DAZ is enabled on x86 for the duration of a work item; decaying
//     states otherwise reach the microcode-assisted denormal path.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#define MONTLOK_X86 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MONTLOK_VECTOR_EXT 1
#endif

namespace montlok {

// float(math.pi) and float(2 * math.pi) exactly as PyTorch casts the Python
// scalars when they meet a float32 tensor.
constexpr float kPiF = 3.14159265358979323846f;
constexpr float kTwoPiF = 6.28318530717958647692f;

// torch.nn.functional.softplus(x) with beta=1, threshold=20.
inline float softplus_f(float x) noexcept {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

inline float sigmoid_f(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

// mamba3_cpu.heavy_tail_activation: clamp_min(x, 0) + 1 / (1 - clamp_max(x, 0)).
inline float heavy_tail_f(float x) noexcept {
    const float negative = std::min(x, 0.0f);
    const float positive = std::max(x, 0.0f);
    return positive + 1.0f / (1.0f - negative);
}

// Element strides of the leading [B, L, H] dimensions of a tensor whose last
// dimension is contiguous.
struct Strides3 {
    int64_t b;
    int64_t t;
    int64_t h;
};

// Per (batch, time, head) coefficients of the trapezoidal recurrence
//   state_t = alpha_t * state_{t-1} + beta_t * k_{t-1} (x) v_{t-1} + gamma_t * k_t (x) v_t
struct StepCoefficients {
    float alpha;
    float beta;
    float gamma;
};

inline StepCoefficients step_coefficients(float adt, float dt, float trap_logit) noexcept {
    const float alpha = std::exp(adt);
    const float trap = sigmoid_f(trap_logit);
    return {alpha, (1.0f - trap) * dt * alpha, trap * dt};
}

// Enables flush-to-zero / denormals-are-zero on x86 for the lifetime of the
// object. Denormal operands take a microcode assist (~100 cycles) on most x86
// cores; the flushed values are below 1.2e-38 and do not affect the result.
struct DenormalGuard {
#ifdef MONTLOK_X86
    unsigned int saved;
    DenormalGuard() noexcept : saved(_mm_getcsr()) { _mm_setcsr(saved | 0x8040u); }
    ~DenormalGuard() { _mm_setcsr(saved); }
#else
    DenormalGuard() noexcept {}
#endif
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;
};

// ---------------------------------------------------------------------------
// Recurrence
// ---------------------------------------------------------------------------

struct SisoPlan {
    int64_t B;
    int64_t L;
    int64_t H;
    int64_t N;  // d_state
    int64_t D;  // headdim
    const float* q;     // [B, L, H, N]
    Strides3 qs;
    const float* k;     // [B, L, H, N]
    Strides3 ks;
    const float* v;     // [B, L, H, D]
    Strides3 vs;
    const float* gate;  // [B, L, H, D], already z * sigmoid(z)
    Strides3 gs;
    const float* alpha; // [B, L, H] contiguous
    const float* beta;  // [B, L, H] contiguous
    const float* gamma; // [B, L, H] contiguous
    const float* skip;  // [H]
    float* out;         // [B, L, H, D]
    Strides3 os;
};

// Width of one headdim block handled by a work item; 0 selects the scalar path
// that processes a whole head.
inline int siso_block_width(int64_t headdim) noexcept {
#ifdef MONTLOK_VECTOR_EXT
    if (headdim % 16 == 0) {
        return 16;
    }
    if (headdim % 8 == 0) {
        return 8;
    }
#endif
    (void)headdim;
    return 0;
}

inline int64_t siso_blocks_per_head(int64_t headdim) noexcept {
    const int width = siso_block_width(headdim);
    return width ? headdim / width : 1;
}

inline int64_t siso_work_items(const SisoPlan& plan) noexcept {
    return plan.B * plan.H * siso_blocks_per_head(plan.D);
}

// Scratch floats a work item needs (state block plus, for the scalar path, an
// accumulator row).
inline int64_t siso_scratch_floats(const SisoPlan& plan) noexcept {
    const int width = siso_block_width(plan.D);
    return width ? plan.N * width : plan.N * plan.D + plan.D;
}

#ifdef MONTLOK_VECTOR_EXT

typedef float v8f __attribute__((vector_size(32)));

inline v8f load8(const float* p) noexcept {
    v8f r;
    std::memcpy(&r, p, sizeof(r));
    return r;
}

inline void store8(float* p, const v8f& v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}

inline v8f splat8(float x) noexcept {
    return v8f{x, x, x, x, x, x, x, x};
}

typedef float v4f __attribute__((vector_size(16)));
typedef double v4d __attribute__((vector_size(32)));

inline v4d load4d(const double* p) noexcept {
    v4d r;
    std::memcpy(&r, p, sizeof(r));
    return r;
}

inline void store4d(double* p, const v4d& v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}

// Four floats widened to double (vcvtps2pd on x86, fcvtl on arm64).
inline v4d load4f_as_d(const float* p) noexcept {
    v4f f;
    std::memcpy(&f, p, sizeof(f));
    return __builtin_convertvector(f, v4d);
}

inline void store4d_as_f(float* p, const v4d& v) noexcept {
    v4f f = __builtin_convertvector(v, v4f);
    std::memcpy(p, &f, sizeof(f));
}

inline v4d splat4d(double x) noexcept {
    return v4d{x, x, x, x};
}

inline double hsum4d(const v4d& v) noexcept {
    return (v[0] + v[1]) + (v[2] + v[3]);
}

// sin and cos of four angles in double precision. The angles are the fp32
// rotary phases already wrapped to [0, 2pi) by the caller, so a Cody-Waite
// reduction to |r| <= pi/4 with a two-part pi/2 is exact and Taylor
// polynomials to r^13 / r^14 leave < 1e-13 truncation error: after rounding to
// float32 each result is within 0.5 ulp (+ epsilon) of the true value, which is
// at least as accurate as libm's sincosf and ATen's Sleef _u10 kernels.
inline void sincos4_d(const v4d& x, v4d* sine, v4d* cosine) noexcept {
    const v4d magic = splat4d(6755399441055744.0);  // 1.5 * 2^52: adds round-to-nearest for |y| < 2^51
    const v4d kf = (x * splat4d(0.63661977236758134308) + magic) - magic;
    const v4d r = (x - kf * splat4d(1.5707963267341256)) - kf * splat4d(6.0771005065061922e-11);
    const v4d r2 = r * r;
    v4d s = splat4d(1.0 / 6227020800.0);
    s = s * r2 - splat4d(1.0 / 39916800.0);
    s = s * r2 + splat4d(1.0 / 362880.0);
    s = s * r2 - splat4d(1.0 / 5040.0);
    s = s * r2 + splat4d(1.0 / 120.0);
    s = s * r2 - splat4d(1.0 / 6.0);
    s = s * r2 * r + r;
    v4d c = splat4d(-1.0 / 87178291200.0);
    c = c * r2 + splat4d(1.0 / 479001600.0);
    c = c * r2 - splat4d(1.0 / 3628800.0);
    c = c * r2 + splat4d(1.0 / 40320.0);
    c = c * r2 - splat4d(1.0 / 720.0);
    c = c * r2 + splat4d(1.0 / 24.0);
    c = c * r2 - splat4d(0.5);
    c = c * r2 + splat4d(1.0);
    for (int lane = 0; lane < 4; ++lane) {
        const int64_t quadrant = static_cast<int64_t>(kf[lane]) & 3;
        const double sv = s[lane];
        const double cv = c[lane];
        double so = (quadrant & 1) ? cv : sv;
        double co = (quadrant & 1) ? -sv : cv;
        if (quadrant & 2) {
            so = -so;
            co = -co;
        }
        (*sine)[lane] = so;
        (*cosine)[lane] = co;
    }
}

// One (batch, head, headdim block) work item. NV is the number of 8-lane
// vectors per block (2 -> 16 lanes). `state` holds N * 8 * NV floats laid out
// [n][lane].
template <int NV>
inline void siso_block_vec(const SisoPlan& p, int64_t b, int64_t h, int64_t d0, float* __restrict state) noexcept {
    constexpr int DB = NV * 8;
    const int64_t N = p.N;
    const int64_t L = p.L;
    std::fill(state, state + N * DB, 0.0f);

    const v8f zero = splat8(0.0f);
    const v8f skipv = splat8(p.skip[h]);
    const float* q_row = p.q + b * p.qs.b + h * p.qs.h;
    const float* k_row = p.k + b * p.ks.b + h * p.ks.h;
    const float* v_row = p.v + b * p.vs.b + h * p.vs.h + d0;
    const float* g_row = p.gate + b * p.gs.b + h * p.gs.h + d0;
    float* o_row = p.out + b * p.os.b + h * p.os.h + d0;
    const float* coef_alpha = p.alpha + b * L * p.H + h;
    const float* coef_beta = p.beta + b * L * p.H + h;
    const float* coef_gamma = p.gamma + b * L * p.H + h;

    for (int64_t t = 0; t < L; ++t) {
        const int64_t ci = t * p.H;
        const v8f al = splat8(coef_alpha[ci]);
        const v8f be = splat8(coef_beta[ci]);
        const v8f ga = splat8(coef_gamma[ci]);

        v8f pv[NV];
        v8f cv[NV];
        v8f acc0[NV];
        v8f acc1[NV];
        const float* k_prev = k_row;
        if (t > 0) {
            k_prev = k_row - p.ks.t;
            const float* v_prev = v_row - p.vs.t;
            for (int i = 0; i < NV; ++i) {
                pv[i] = be * load8(v_prev + 8 * i);
            }
        } else {
            for (int i = 0; i < NV; ++i) {
                pv[i] = zero;
            }
        }
        for (int i = 0; i < NV; ++i) {
            cv[i] = ga * load8(v_row + 8 * i);
            acc0[i] = zero;
            acc1[i] = zero;
        }

        int64_t n = 0;
        for (; n + 2 <= N; n += 2) {
            float* s0 = state + n * DB;
            float* s1 = s0 + DB;
            const v8f kp0 = splat8(k_prev[n]);
            const v8f kc0 = splat8(k_row[n]);
            const v8f qq0 = splat8(q_row[n]);
            const v8f kp1 = splat8(k_prev[n + 1]);
            const v8f kc1 = splat8(k_row[n + 1]);
            const v8f qq1 = splat8(q_row[n + 1]);
            for (int i = 0; i < NV; ++i) {
                v8f s = load8(s0 + 8 * i);
                s = s * al + kp0 * pv[i] + kc0 * cv[i];
                store8(s0 + 8 * i, s);
                acc0[i] += s * qq0;
            }
            for (int i = 0; i < NV; ++i) {
                v8f s = load8(s1 + 8 * i);
                s = s * al + kp1 * pv[i] + kc1 * cv[i];
                store8(s1 + 8 * i, s);
                acc1[i] += s * qq1;
            }
        }
        for (; n < N; ++n) {
            float* s0 = state + n * DB;
            const v8f kp0 = splat8(k_prev[n]);
            const v8f kc0 = splat8(k_row[n]);
            const v8f qq0 = splat8(q_row[n]);
            for (int i = 0; i < NV; ++i) {
                v8f s = load8(s0 + 8 * i);
                s = s * al + kp0 * pv[i] + kc0 * cv[i];
                store8(s0 + 8 * i, s);
                acc0[i] += s * qq0;
            }
        }

        for (int i = 0; i < NV; ++i) {
            const v8f y = (acc0[i] + acc1[i]) + skipv * load8(v_row + 8 * i);
            store8(o_row + 8 * i, y * load8(g_row + 8 * i));
        }

        q_row += p.qs.t;
        k_row += p.ks.t;
        v_row += p.vs.t;
        g_row += p.gs.t;
        o_row += p.os.t;
    }
}

#endif  // MONTLOK_VECTOR_EXT

// Portable path for headdim values that are not a multiple of 8 (or compilers
// without vector extensions). Same association order as the vector path.
// `state` holds N * D floats laid out [n][d] followed by D accumulator floats.
inline void siso_head_scalar(const SisoPlan& p, int64_t b, int64_t h, float* __restrict state) noexcept {
    const int64_t N = p.N;
    const int64_t D = p.D;
    const int64_t L = p.L;
    float* __restrict acc = state + N * D;
    std::fill(state, state + N * D, 0.0f);

    const float skip = p.skip[h];
    const float* q_row = p.q + b * p.qs.b + h * p.qs.h;
    const float* k_row = p.k + b * p.ks.b + h * p.ks.h;
    const float* v_row = p.v + b * p.vs.b + h * p.vs.h;
    const float* g_row = p.gate + b * p.gs.b + h * p.gs.h;
    float* o_row = p.out + b * p.os.b + h * p.os.h;
    const float* coef_alpha = p.alpha + b * L * p.H + h;
    const float* coef_beta = p.beta + b * L * p.H + h;
    const float* coef_gamma = p.gamma + b * L * p.H + h;

    for (int64_t t = 0; t < L; ++t) {
        const int64_t ci = t * p.H;
        const float alpha = coef_alpha[ci];
        const float beta = coef_beta[ci];
        const float gamma = coef_gamma[ci];
        for (int64_t d = 0; d < D; ++d) {
            acc[d] = 0.0f;
        }
        if (t > 0) {
            const float* k_prev = k_row - p.ks.t;
            const float* v_prev = v_row - p.vs.t;
            for (int64_t n = 0; n < N; ++n) {
                float* __restrict s = state + n * D;
                const float kp = k_prev[n];
                const float kc = k_row[n];
                const float qn = q_row[n];
                for (int64_t d = 0; d < D; ++d) {
                    const float updated = s[d] * alpha + kp * (beta * v_prev[d]) + kc * (gamma * v_row[d]);
                    s[d] = updated;
                    acc[d] += updated * qn;
                }
            }
        } else {
            for (int64_t n = 0; n < N; ++n) {
                float* __restrict s = state + n * D;
                const float kc = k_row[n];
                const float qn = q_row[n];
                for (int64_t d = 0; d < D; ++d) {
                    const float updated = s[d] * alpha + kc * (gamma * v_row[d]);
                    s[d] = updated;
                    acc[d] += updated * qn;
                }
            }
        }
        for (int64_t d = 0; d < D; ++d) {
            o_row[d] = (acc[d] + skip * v_row[d]) * g_row[d];
        }
        q_row += p.qs.t;
        k_row += p.ks.t;
        v_row += p.vs.t;
        g_row += p.gs.t;
        o_row += p.os.t;
    }
}

// Runs one work item in [0, siso_work_items(plan)). `scratch` must hold
// siso_scratch_floats(plan) floats and may be reused across items.
inline void siso_run_work_item(const SisoPlan& plan, int64_t item, float* scratch) noexcept {
    const int64_t blocks = siso_blocks_per_head(plan.D);
    const int64_t block = item % blocks;
    const int64_t bh = item / blocks;
    const int64_t h = bh % plan.H;
    const int64_t b = bh / plan.H;
    DenormalGuard guard;
    switch (siso_block_width(plan.D)) {
#ifdef MONTLOK_VECTOR_EXT
        case 16:
            siso_block_vec<2>(plan, b, h, block * 16, scratch);
            break;
        case 8:
            siso_block_vec<1>(plan, b, h, block * 8, scratch);
            break;
#endif
        default:
            siso_head_scalar(plan, b, h, scratch);
            break;
    }
}

// ---------------------------------------------------------------------------
// Fused pre-processing of the in_proj output
// ---------------------------------------------------------------------------
//
// The in_proj row layout is [z | x | B | C | dt | A | trap | angles] with widths
// [d_inner, d_inner, G*N, G*N, H, H, H, R]. Three phases replace ~60 small
// PyTorch ops:
//   A  per token : BCNorm + bias -> unrotated q/k, dt/A/trap -> alpha/beta/gamma,
//                  tanh(angles) * pi * dt -> angle_term
//   B  per (b, h, r) chain : cumulative sum over time (double accumulation,
//                  exactly like ATen's CPU cumsum) -> cum
//   C  per token : wrap to [0, 2pi), sincos, rotate the first 2R lanes of q/k

struct PrepPlan {
    int64_t B;
    int64_t L;
    int64_t W;           // in_proj width
    const float* proj;   // [B, L, W], last dim contiguous
    int64_t proj_sb;
    int64_t proj_st;
    int64_t d_inner;
    int64_t N;           // d_state
    int64_t G;           // B/C groups
    int64_t H;           // heads
    int64_t D;           // headdim
    int64_t R;           // rope angles (pairs rotated)
    const float* b_bias;    // [H, N]
    const float* c_bias;    // [H, N]
    const float* b_norm_w;  // [N]
    const float* c_norm_w;  // [N]
    float b_eps;
    float c_eps;
    const float* dt_bias;   // [H]
    float a_floor;
    float* q;            // [B, L, H, N] contiguous (output)
    float* k;            // [B, L, H, N] contiguous (output)
    float* alpha;        // [B, L, H] (output)
    float* beta;         // [B, L, H] (output)
    float* gamma;        // [B, L, H] (output)
    float* angle_term;   // [B, L, H, R] workspace
    float* cum;          // [B, H, R, L] workspace
};

inline int64_t prep_expected_width(const PrepPlan& p) noexcept {
    return 2 * p.d_inner + 2 * p.G * p.N + 3 * p.H + p.R;
}

inline int64_t prep_scratch_floats(const PrepPlan& p) noexcept {
    return 2 * p.G * p.N + p.R;
}

// BCNorm: x * rsqrt(mean(x^2) + eps) * weight, all in fp32.
inline void rms_norm_row(const float* __restrict x, const float* __restrict w, float eps, int64_t n,
                         float* __restrict out) noexcept {
    float sum_squares = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum_squares += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (x[i] * scale) * w[i];
    }
}

// Phase A for token (b, t). `scratch` holds prep_scratch_floats(p) floats.
inline void prep_token(const PrepPlan& p, int64_t b, int64_t t, float* __restrict scratch) noexcept {
    const int64_t N = p.N;
    const int64_t GN = p.G * N;
    const float* row = p.proj + b * p.proj_sb + t * p.proj_st;
    const float* b_raw = row + 2 * p.d_inner;
    const float* c_raw = b_raw + GN;
    const float* raw_dt = c_raw + GN;
    const float* raw_a = raw_dt + p.H;
    const float* raw_trap = raw_a + p.H;
    const float* raw_angles = raw_trap + p.H;

    float* b_norm = scratch;
    float* c_norm = scratch + GN;
    float* angles = scratch + 2 * GN;
    for (int64_t g = 0; g < p.G; ++g) {
        rms_norm_row(b_raw + g * N, p.b_norm_w, p.b_eps, N, b_norm + g * N);
        rms_norm_row(c_raw + g * N, p.c_norm_w, p.c_eps, N, c_norm + g * N);
    }
    for (int64_t r = 0; r < p.R; ++r) {
        angles[r] = std::tanh(raw_angles[r]) * kPiF;
    }

    const int64_t heads_per_group = p.H / p.G;
    const int64_t tok = b * p.L + t;
    for (int64_t h = 0; h < p.H; ++h) {
        const int64_t g = h / heads_per_group;
        const float* bn = b_norm + g * N;
        const float* cn = c_norm + g * N;
        const float* bb = p.b_bias + h * N;
        const float* cb = p.c_bias + h * N;
        float* __restrict k_out = p.k + (tok * p.H + h) * N;
        float* __restrict q_out = p.q + (tok * p.H + h) * N;
        for (int64_t n = 0; n < N; ++n) {
            k_out[n] = bn[n] + bb[n];
            q_out[n] = cn[n] + cb[n];
        }

        const float dt = softplus_f(raw_dt[h] + p.dt_bias[h]);
        const float a = -std::max(heavy_tail_f(raw_a[h]), p.a_floor);
        const StepCoefficients coef = step_coefficients(a * dt, dt, raw_trap[h]);
        const int64_t ci = tok * p.H + h;
        p.alpha[ci] = coef.alpha;
        p.beta[ci] = coef.beta;
        p.gamma[ci] = coef.gamma;
        float* __restrict term = p.angle_term + ci * p.R;
        for (int64_t r = 0; r < p.R; ++r) {
            term[r] = angles[r] * dt;
        }
    }
}

// Phase B for chain index (b * H + h) * R + r: prefix sum over time.
inline void prep_chain(const PrepPlan& p, int64_t chain) noexcept {
    const int64_t r = chain % p.R;
    const int64_t bh = chain / p.R;
    const int64_t h = bh % p.H;
    const int64_t b = bh / p.H;
    const float* term = p.angle_term + ((b * p.L) * p.H + h) * p.R + r;
    const int64_t stride = p.H * p.R;
    float* __restrict out = p.cum + chain * p.L;
    double running = 0.0;
    for (int64_t t = 0; t < p.L; ++t) {
        running += static_cast<double>(term[t * stride]);
        out[t] = static_cast<float>(running);
    }
}

// sin/cos of up to 4 fp32 angles, computed in double and rounded to fp32 once.
inline void sincos_angles(const float* angles, int count, float* sines, float* cosines) noexcept {
#ifdef MONTLOK_VECTOR_EXT
    v4d x = splat4d(0.0);
    for (int i = 0; i < count; ++i) {
        x[i] = static_cast<double>(angles[i]);
    }
    v4d s;
    v4d c;
    sincos4_d(x, &s, &c);
    for (int i = 0; i < count; ++i) {
        sines[i] = static_cast<float>(s[i]);
        cosines[i] = static_cast<float>(c[i]);
    }
#else
    for (int i = 0; i < count; ++i) {
        sines[i] = static_cast<float>(std::sin(static_cast<double>(angles[i])));
        cosines[i] = static_cast<float>(std::cos(static_cast<double>(angles[i])));
    }
#endif
}

inline void rotate_pair(float* __restrict pair, float sine, float cosine) noexcept {
    const float left = pair[0];
    const float right = pair[1];
    pair[0] = left * cosine - right * sine;
    pair[1] = left * sine + right * cosine;
}

// Phase C for token (b, t): rotary embedding of q/k in place.
inline void prep_rotate(const PrepPlan& p, int64_t b, int64_t t) noexcept {
    const int64_t tok = b * p.L + t;
    for (int64_t h = 0; h < p.H; ++h) {
        float* q_out = p.q + (tok * p.H + h) * p.N;
        float* k_out = p.k + (tok * p.H + h) * p.N;
        const float* cum = p.cum + ((b * p.H + h) * p.R) * p.L + t;
        for (int64_t r0 = 0; r0 < p.R; r0 += 4) {
            const int count = static_cast<int>(std::min<int64_t>(4, p.R - r0));
            float angle[4];
            float sine[4];
            float cosine[4];
            for (int i = 0; i < count; ++i) {
                const float cumulative = cum[(r0 + i) * p.L];
                // cumulative - 2*pi*floor(cumulative / (2*pi)); the volatile keeps
                // the product separately rounded like the two PyTorch ops.
                volatile float wraps = kTwoPiF * std::floor(cumulative / kTwoPiF);
                angle[i] = cumulative - wraps;
            }
            sincos_angles(angle, count, sine, cosine);
            for (int i = 0; i < count; ++i) {
                rotate_pair(q_out + 2 * (r0 + i), sine[i], cosine[i]);
                rotate_pair(k_out + 2 * (r0 + i), sine[i], cosine[i]);
            }
        }
    }
}

}  // namespace montlok
