// Copyright 2026 OpenAI.
// Apache-2.0 weight-compatible CPU kernels for the small per-token layers
// around the RDT stage-1/stage-2 blocks: RMSNorm / GroupedRMSNorm and the MLA
// rotary embedding + key assembly.
//
// Torch-free like montlok_kernel.h. These exist purely to cut operator count:
// in eager PyTorch an RMSNorm is seven tiny ops (float, pow, mean, add, rsqrt,
// mul, mul) and the MLA key path is ~15 (split, two rotate_half cats, four
// muls, expand, cat, ...); with 12-32 threads each op costs a thread fork/join
// that is 5-10x the arithmetic. All arithmetic here is done in double and
// rounded to float32 once.

#pragma once

#include "montlok_mhc.h"

namespace montlok {

// out[row] = x[row] * rsqrt(mean(x[row]^2 over each group) + eps) * w, for
// `rows` rows of `dim` floats split into `groups` equal groups.
// x rows may be strided (row_stride floats apart); out is contiguous.
inline void rms_norm_rows(const float* __restrict x, int64_t row_stride, const float* __restrict w,
                          double eps, int64_t dim, int64_t groups, int64_t row_begin, int64_t row_end,
                          float* __restrict out) noexcept {
    const int64_t group_size = dim / groups;
    for (int64_t row = row_begin; row < row_end; ++row) {
        const float* __restrict xr = x + row * row_stride;
        float* __restrict outr = out + row * dim;
        for (int64_t g = 0; g < groups; ++g) {
            const float* __restrict xg = xr + g * group_size;
            const float* __restrict wg = w + g * group_size;
            float* __restrict og = outr + g * group_size;
            double sumsq = 0.0;
            int64_t k = 0;
#ifdef MONTLOK_VECTOR_EXT
            {
                v4d acc0 = splat4d(0.0);
                v4d acc1 = splat4d(0.0);
                for (; k + 8 <= group_size; k += 8) {
                    const v4d a = load4f_as_d(xg + k);
                    const v4d b = load4f_as_d(xg + k + 4);
                    acc0 += a * a;
                    acc1 += b * b;
                }
                sumsq = hsum4d(acc0 + acc1);
            }
#endif
            for (; k < group_size; ++k) {
                const double v = static_cast<double>(xg[k]);
                sumsq += v * v;
            }
            const double scale = 1.0 / std::sqrt(sumsq / static_cast<double>(group_size) + eps);
            k = 0;
#ifdef MONTLOK_VECTOR_EXT
            {
                const v4d vscale = splat4d(scale);
                for (; k + 4 <= group_size; k += 4) {
                    store4d_as_f(og + k, load4f_as_d(xg + k) * vscale * load4f_as_d(wg + k));
                }
            }
#endif
            for (; k < group_size; ++k) {
                og[k] = static_cast<float>(static_cast<double>(xg[k]) * scale * static_cast<double>(wg[k]));
            }
        }
    }
}

// Model/layers/rope.py apply_rope with rotate_half on one vector of `rope_dim`
// floats: out = x * cos + cat(-x2, x1) * sin, x1/x2 the two halves.
// cos/sin are the [rope_dim] rows for this position (cat([angles, angles])).
// `out` may alias `x` (both halves of a pair are read before either is written).
inline void rope_vector(const float* x, const float* __restrict cosine, const float* __restrict sine,
                        int64_t rope_dim, float* out) noexcept {
    const int64_t half = rope_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const double x1 = static_cast<double>(x[i]);
        const double x2 = static_cast<double>(x[i + half]);
        out[i] = static_cast<float>(x1 * static_cast<double>(cosine[i]) - x2 * static_cast<double>(sine[i]));
        out[i + half] = static_cast<float>(x2 * static_cast<double>(cosine[i + half]) +
                                           x1 * static_cast<double>(sine[i + half]));
    }
}

// MLA rotary step for one token, all heads:
//   q[h, nope:] <- rope(q[h, nope:])            (in place, strided [heads, head_dim])
//   k[h, :nope] <- kv[h, :nope]; k[h, nope:] <- rope(k_rope)   (k_rope shared by heads)
struct MlaRopePlan {
    float* q;             // [tokens, heads, head_dim] (strides below)
    int64_t q_token_stride;
    int64_t q_head_stride;
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

inline void mla_rope_token(const MlaRopePlan& p, int64_t token, float* __restrict scratch) noexcept {
    const int64_t position = token % p.length;
    const float* __restrict cosine = p.cosine + position * p.rope_dim;
    const float* __restrict sine = p.sine + position * p.rope_dim;
    // Rotated shared key part, computed once per token.
    rope_vector(p.k_rope + token * p.k_rope_token_stride, cosine, sine, p.rope_dim, scratch);
    for (int64_t h = 0; h < p.heads; ++h) {
        float* __restrict q = p.q + token * p.q_token_stride + h * p.q_head_stride + p.nope_dim;
        rope_vector(q, cosine, sine, p.rope_dim, q);
        const float* __restrict kv = p.kv + token * p.kv_token_stride + h * p.kv_head_stride;
        float* __restrict k = p.k + (token * p.heads + h) * p.head_dim;
        std::memcpy(k, kv, static_cast<size_t>(p.nope_dim) * sizeof(float));
        std::memcpy(k + p.nope_dim, scratch, static_cast<size_t>(p.rope_dim) * sizeof(float));
    }
}

}  // namespace montlok
