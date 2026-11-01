// Copyright 2026 OpenAI.
// Apache-2.0 weight-compatible CPU kernels for RDT4quant: the plain data and
// geometry helpers shared by every instruction-set variant of the kernels.
//
// montlok_kernel.h, montlok_mhc.h and montlok_layers.h are compiled once per
// instruction set (montlok_isa_*.cpp, see montlok_isa.h), each inside its own
// inline namespace so the symbols of the variants never collide. Everything in
// this header is independent of the instruction set and stays in plain
// namespace montlok: the plan structs that cross the variant boundary, the
// work decomposition the PyTorch bindings schedule, and small scalar helpers.
// It is torch-free like the kernel headers.

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

// Bytes of the widest float vector the recurrence kernel uses. 32 is one AVX2
// ymm register (two NEON q registers; two SSE xmm registers on the x86-64
// baseline); the AVX-512 variant compiles with 64. The arithmetic per lane is
// identical at every width, so variants with the same FMA availability
// produce bit-identical results.
#ifndef MONTLOK_VEC_BYTES
#define MONTLOK_VEC_BYTES 32
#endif

// A variant translation unit defines MONTLOK_ISA_NAMESPACE (isa_avx2, ...);
// the kernel headers then live in montlok::<variant> and every inline kernel
// gets a symbol of its own. A direct include (self-test, microstructure) sees
// the kernels in plain namespace montlok, as before.
#ifdef MONTLOK_ISA_NAMESPACE
#define MONTLOK_ISA_NAMESPACE_BEGIN inline namespace MONTLOK_ISA_NAMESPACE {
#define MONTLOK_ISA_NAMESPACE_END }
#else
#define MONTLOK_ISA_NAMESPACE_BEGIN
#define MONTLOK_ISA_NAMESPACE_END
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
// arm64 cores handle denormals at full speed, so nothing is changed there.
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
// Mamba-3 SISO recurrence
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
    const float* gate;  // [B, L, H, D]: z * sigmoid(z), or the raw z lanes when gate_logits
    Strides3 gs;
    bool gate_logits;   // apply silu to `gate` inside the kernel (double, one rounding)
    const float* alpha; // [B, L, H] contiguous
    const float* beta;  // [B, L, H] contiguous
    const float* gamma; // [B, L, H] contiguous
    const float* skip;  // [H]
    float* out;         // [B, L, H, D]
    Strides3 os;
};

// Width of one headdim block handled by a work item; 0 selects the scalar path
// that processes a whole head. The same for every variant, so the bindings can
// schedule work without knowing which variant runs it.
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

// ---------------------------------------------------------------------------
// Manifold-Constrained Hyper-Connections
// ---------------------------------------------------------------------------

// Everything one ManifoldHyperConnection.forward needs before the wrapped
// layer runs. Tokens are B*L flattened; `streams` is [tokens, n, d] with the
// stream rows `stream_stride` floats apart (d for a contiguous tensor, 0 when
// every stream is the same [tokens, d] row, i.e. the expanded backbone).
struct MhcPlan {
    const float* streams;
    int64_t stream_stride;
    int64_t token_stride;   // floats between tokens of `streams`
    const float* norm_w;    // [d] RMSNorm weight of dyn_norm
    const double* proj_w;   // [(2n + n*n), d] pre_proj, post_proj, res_proj rows widened
    const float* pre_bias;  // [n]
    const float* post_bias; // [n]
    const float* res_bias;  // [n, n]
    double pre_alpha;
    double post_alpha;
    double res_alpha;
    double eps;
    int64_t tokens;
    int64_t n;
    int64_t d;
    int sinkhorn_iters;
    float* agg;   // [tokens, d]   sum_i pre_i * streams_i
    const float* agg_norm_w;  // optional [d], normalizes agg in place for the wrapped layer
    double agg_norm_eps;
    float* pre;   // [tokens, n]   sigmoid(pre logits)
    float* post;  // [tokens, n]   2 * sigmoid(post logits)
    float* res;   // [tokens, n, n] doubly stochastic mixing matrix
};

inline int64_t mhc_proj_rows(int64_t n) noexcept {
    return 2 * n + n * n;
}

// Tokens whose Sinkhorn projections are iterated together, one per SIMD lane.
constexpr int kMhcTokenBlock = 4;

// Doubles of per-thread scratch needed by mhc_prepare_block.
inline int64_t mhc_scratch_doubles(const MhcPlan& p) noexcept {
    return (p.d + mhc_proj_rows(p.n) + p.n * p.n + p.n) * kMhcTokenBlock;
}

// Second half of a hyper-connection over all tokens: inputs of the residual
// that just ran, plus the new streams it writes.
struct MhcCombinePlan {
    const float* streams;   // [tokens, n, d] rows stream_stride apart, tokens token_stride apart
    int64_t stream_stride;
    int64_t token_stride;
    const float* res;       // [tokens, n, n]
    const float* post;      // [tokens, n]
    const float* out;       // [tokens, d]
    float* new_streams;     // [tokens, n, d] contiguous
    float* collapsed;       // [tokens, d] or null
    int64_t n;
    int64_t d;
};

// ---------------------------------------------------------------------------
// MLA rotary embedding
// ---------------------------------------------------------------------------

// MLA rotary step for one token, all heads:
//   q[h, nope:] <- rope(q[h, nope:])            (in place, strided [heads, head_dim])
//   k[h, :nope] <- kv[h, :nope]; k[h, nope:] <- rope(k_rope)   (k_rope shared by heads)
// Queries may cover only the trailing `q_length` positions of each sequence
// (q_length == length for the full forward); keys are always assembled for
// every token.
struct MlaRopePlan {
    float* q;             // [batch, q_length, heads, head_dim] (strides below)
    int64_t q_token_stride;
    int64_t q_head_stride;
    int64_t q_length;     // queries are positions [length - q_length, length)
    const float* kv;      // [tokens, heads, nope + head_dim]
    int64_t kv_token_stride;
    int64_t kv_head_stride;
    const float* k_rope;  // [tokens, rope_dim]
    int64_t k_rope_token_stride;
    const float* cosine;  // [positions, rope_dim] contiguous, position = token % length
    const float* sine;
    float* k;             // [tokens, heads, head_dim] contiguous output
    int64_t length;
    int64_t heads;
    int64_t head_dim;
    int64_t nope_dim;
    int64_t rope_dim;
};

}  // namespace montlok
