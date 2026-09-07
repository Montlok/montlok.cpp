// Copyright 2026 OpenAI.
// Apache-2.0 weight-compatible CPU kernels for Manifold-Constrained
// Hyper-Connections (mHC, arXiv:2512.24880) as used by the RDT stage-2 layer.
//
// Torch-free like montlok_kernel.h. One mHC residual costs, in the PyTorch
// reference, ~100 tiny ops per layer call (20 Sinkhorn iterations x 2
// logsumexp x 2 subtractions on a [B*L, n, n] tensor, two batched matmuls over
// B*L 4x4 matrices, ...). Each op is a few microseconds of work plus a thread
// fork. Here the whole coefficient computation for one token is a single pass
// in one thread.
//
// Numerics: every intermediate is accumulated in double precision and rounded
// to float32 exactly once, so each output is the correctly rounded value of the
// fp32 reference expression (the reference itself carries a few ulp of fp32
// rounding). The Sinkhorn projection is iterated in linear space in double
// precision, which is the same fixed point as the reference's fp32 log-space
// iteration without its per-iteration exp/log rounding.

#pragma once

#include "montlok_kernel.h"

namespace montlok {

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

inline double sigmoid_d(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
}

// out[lane * rows + r] = w[r] . ref[lane] for R consecutive rows of w (stride
// d) and the kMhcTokenBlock refs (stride d). Each weight vector loaded once
// feeds W FMAs and each ref vector feeds R, so the loop runs FMA bound instead
// of load bound, and the weight block is streamed once per W tokens rather than
// once per token (in double it exceeds a 32 KB L1).
template <int R>
inline void dot_rows_block(const double* __restrict w, const double* __restrict ref, int64_t d, int64_t rows,
                           double* __restrict out) noexcept {
    constexpr int W = kMhcTokenBlock;
    int64_t i = 0;
#ifdef MONTLOK_VECTOR_EXT
    v4d acc[R][W];
    for (int r = 0; r < R; ++r) {
        for (int lane = 0; lane < W; ++lane) {
            acc[r][lane] = splat4d(0.0);
        }
    }
    for (; i + 4 <= d; i += 4) {
        v4d wv[R];
        for (int r = 0; r < R; ++r) {
            wv[r] = load4d(w + r * d + i);
        }
        for (int lane = 0; lane < W; ++lane) {
            const v4d rv = load4d(ref + lane * d + i);
            for (int r = 0; r < R; ++r) {
                acc[r][lane] += wv[r] * rv;
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        for (int lane = 0; lane < W; ++lane) {
            out[lane * rows + r] = hsum4d(acc[r][lane]);
        }
    }
#else
    for (int r = 0; r < R; ++r) {
        for (int lane = 0; lane < W; ++lane) {
            out[lane * rows + r] = 0.0;
        }
    }
#endif
    for (; i < d; ++i) {
        for (int r = 0; r < R; ++r) {
            for (int lane = 0; lane < W; ++lane) {
                out[lane * rows + r] += w[r * d + i] * ref[lane * d + i];
            }
        }
    }
}

// logits[lane * rows + r] = w[r] . ref[lane] for all rows and lanes.
inline void dot_all_rows_block(const double* __restrict w, const double* __restrict ref, int64_t rows, int64_t d,
                               double* __restrict logits) noexcept {
    int64_t r = 0;
    for (; r + 2 <= rows; r += 2) {
        dot_rows_block<2>(w + r * d, ref, d, rows, logits + r);
    }
    for (; r < rows; ++r) {
        dot_rows_block<1>(w + r * d, ref, d, rows, logits + r);
    }
}

// Sinkhorn-Knopp for kMhcTokenBlock matrices at once, entry e of token `lane`
// at m[e * kMhcTokenBlock + lane]. exp(logits) is projected onto the doubly
// stochastic matrices by alternating row and column normalisation (the
// linear-space form of sinkhorn_knopp in Model/layers/mhc.py; like the
// reference it runs at least one iteration and finishes with the column step).
// The lanes are independent, so the 2n reciprocals per iteration, which
// dominate the latency-bound chain, are shared by four tokens.
inline void sinkhorn_block(double* __restrict m, double* __restrict sums, int64_t n, int iters) noexcept {
    constexpr int W = kMhcTokenBlock;
    for (int64_t e = 0; e < n * n * W; ++e) {
        m[e] = std::exp(m[e]);
    }
#ifdef MONTLOK_VECTOR_EXT
    const v4d one = splat4d(1.0);
    for (int it = 0; it < std::max(iters, 1); ++it) {
        for (int64_t i = 0; i < n; ++i) {
            double* __restrict row = m + i * n * W;
            v4d s = load4d(row);
            for (int64_t j = 1; j < n; ++j) {
                s += load4d(row + j * W);
            }
            const v4d inv = one / s;
            for (int64_t j = 0; j < n; ++j) {
                store4d(row + j * W, load4d(row + j * W) * inv);
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            v4d s = load4d(m + j * W);
            for (int64_t i = 1; i < n; ++i) {
                s += load4d(m + (i * n + j) * W);
            }
            store4d(sums + j * W, one / s);
        }
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                double* __restrict cell = m + (i * n + j) * W;
                store4d(cell, load4d(cell) * load4d(sums + j * W));
            }
        }
    }
#else
    for (int it = 0; it < std::max(iters, 1); ++it) {
        for (int lane = 0; lane < W; ++lane) {
            for (int64_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (int64_t j = 0; j < n; ++j) {
                    s += m[(i * n + j) * W + lane];
                }
                const double inv = 1.0 / s;
                for (int64_t j = 0; j < n; ++j) {
                    m[(i * n + j) * W + lane] *= inv;
                }
            }
            for (int64_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    s += m[(i * n + j) * W + lane];
                }
                sums[j * W + lane] = 1.0 / s;
            }
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    m[(i * n + j) * W + lane] *= sums[j * W + lane];
                }
            }
        }
    }
#endif
}

// RMS-normalize one freshly produced aggregate in place. Keeping this beside
// mhc_prepare_block lets the aggregate stay hot in L1 and removes a separate
// PyTorch/C++ operator plus one full read pass before the wrapped layer.
inline void mhc_agg_norm_inplace(float* __restrict x, const float* __restrict w, double eps, int64_t d) noexcept {
    double sumsq = 0.0;
    int64_t k = 0;
#ifdef MONTLOK_VECTOR_EXT
    {
        v4d acc0 = splat4d(0.0);
        v4d acc1 = splat4d(0.0);
        for (; k + 8 <= d; k += 8) {
            const v4d a = load4f_as_d(x + k);
            const v4d b = load4f_as_d(x + k + 4);
            acc0 += a * a;
            acc1 += b * b;
        }
        sumsq = hsum4d(acc0 + acc1);
    }
#endif
    for (; k < d; ++k) {
        const double value = static_cast<double>(x[k]);
        sumsq += value * value;
    }
    const double scale = 1.0 / std::sqrt(sumsq / static_cast<double>(d) + eps);
    k = 0;
#ifdef MONTLOK_VECTOR_EXT
    const v4d vscale = splat4d(scale);
    for (; k + 4 <= d; k += 4) {
        store4d_as_f(x + k, load4f_as_d(x + k) * vscale * load4f_as_d(w + k));
    }
#endif
    for (; k < d; ++k) {
        x[k] = static_cast<float>(static_cast<double>(x[k]) * scale * static_cast<double>(w[k]));
    }
}

// Coefficients and aggregated input for `count` (<= kMhcTokenBlock)
// consecutive tokens starting at token0. Padding lanes carry zeros through the
// shared dot products and Sinkhorn so they stay finite; nothing is written for them.
inline void mhc_prepare_block(const MhcPlan& p, int64_t token0, int64_t count, double* __restrict scratch) noexcept {
    constexpr int W = kMhcTokenBlock;
    const int64_t n = p.n;
    const int64_t d = p.d;
    const int64_t rows = mhc_proj_rows(n);
    double* __restrict ref = scratch;                 // [W][d]
    double* __restrict logits = ref + W * d;          // [W][rows]
    double* __restrict mat = logits + W * rows;       // [n*n][W] res logits, then Sinkhorn result
    double* __restrict sums = mat + n * n * W;        // [n][W]
    const double inv_n = 1.0 / static_cast<double>(n);

    // ref = RMSNorm(mean_i streams_i) in double, one row per lane.
    for (int lane = 0; lane < W; ++lane) {
        double* __restrict ref_lane = ref + lane * d;
        if (lane >= count) {
            for (int64_t k = 0; k < d; ++k) {
                ref_lane[k] = 0.0;
            }
            continue;
        }
        const float* __restrict streams = p.streams + (token0 + lane) * p.token_stride;
        const int64_t sd = p.stream_stride;
        double sumsq = 0.0;
        int64_t k = 0;
#ifdef MONTLOK_VECTOR_EXT
        {
            v4d sq = splat4d(0.0);
            const v4d vinv = splat4d(inv_n);
            for (; k + 4 <= d; k += 4) {
                v4d m = load4f_as_d(streams + k);
                for (int64_t i = 1; i < n; ++i) {
                    m += load4f_as_d(streams + i * sd + k);
                }
                m *= vinv;
                store4d(ref_lane + k, m);
                sq += m * m;
            }
            sumsq = hsum4d(sq);
        }
#endif
        for (; k < d; ++k) {
            double m = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                m += static_cast<double>(streams[i * sd + k]);
            }
            m *= inv_n;
            ref_lane[k] = m;
            sumsq += m * m;
        }
        const double scale = 1.0 / std::sqrt(sumsq / static_cast<double>(d) + p.eps);
        k = 0;
#ifdef MONTLOK_VECTOR_EXT
        {
            const v4d vscale = splat4d(scale);
            for (; k + 4 <= d; k += 4) {
                store4d(ref_lane + k, load4d(ref_lane + k) * vscale * load4f_as_d(p.norm_w + k));
            }
        }
#endif
        for (; k < d; ++k) {
            ref_lane[k] = ref_lane[k] * scale * static_cast<double>(p.norm_w[k]);
        }
    }

    // logits = bias + alpha * tanh(W ref) for the three maps, all lanes at once.
    dot_all_rows_block(p.proj_w, ref, rows, d, logits);

    for (int lane = 0; lane < W; ++lane) {
        if (lane >= count) {
            for (int64_t e = 0; e < n * n; ++e) {
                mat[e * W + lane] = 0.0;
            }
            continue;
        }
        const int64_t token = token0 + lane;
        double* __restrict logit = logits + lane * rows;
        for (int64_t r = 0; r < rows; ++r) {
            logit[r] = std::tanh(logit[r]);
        }
        float* __restrict pre = p.pre + token * n;
        float* __restrict post = p.post + token * n;
        for (int64_t i = 0; i < n; ++i) {
            const double pre_logit = static_cast<double>(p.pre_bias[i]) + p.pre_alpha * logit[i];
            const double post_logit = static_cast<double>(p.post_bias[i]) + p.post_alpha * logit[n + i];
            pre[i] = static_cast<float>(sigmoid_d(pre_logit));
            post[i] = static_cast<float>(2.0 * sigmoid_d(post_logit));
        }
        for (int64_t e = 0; e < n * n; ++e) {
            mat[e * W + lane] = static_cast<double>(p.res_bias[e]) + p.res_alpha * logit[2 * n + e];
        }

        // agg = sum_i pre_i * streams_i: products exact in double, one rounding.
        const float* __restrict streams = p.streams + token * p.token_stride;
        const int64_t sd = p.stream_stride;
        float* __restrict agg = p.agg + token * d;
        int64_t k = 0;
#ifdef MONTLOK_VECTOR_EXT
        for (; k + 4 <= d; k += 4) {
            v4d acc = splat4d(static_cast<double>(pre[0])) * load4f_as_d(streams + k);
            for (int64_t i = 1; i < n; ++i) {
                acc += splat4d(static_cast<double>(pre[i])) * load4f_as_d(streams + i * sd + k);
            }
            store4d_as_f(agg + k, acc);
        }
#endif
        for (; k < d; ++k) {
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                acc += static_cast<double>(pre[i]) * static_cast<double>(streams[i * sd + k]);
            }
            agg[k] = static_cast<float>(acc);
        }
        if (p.agg_norm_w != nullptr) {
            mhc_agg_norm_inplace(agg, p.agg_norm_w, p.agg_norm_eps, d);
        }
    }

    sinkhorn_block(mat, sums, n, p.sinkhorn_iters);
    for (int lane = 0; lane < count; ++lane) {
        float* __restrict res = p.res + (token0 + lane) * n * n;
        for (int64_t e = 0; e < n * n; ++e) {
            res[e] = static_cast<float>(mat[e * W + lane]);
        }
    }
}

// Coefficients for tokens [begin, end).
inline void mhc_prepare_range(const MhcPlan& p, int64_t begin, int64_t end, double* __restrict scratch) noexcept {
    for (int64_t token = begin; token < end; token += kMhcTokenBlock) {
        mhc_prepare_block(p, token, std::min<int64_t>(kMhcTokenBlock, end - token), scratch);
    }
}

// new_streams_i = sum_j res_ij * streams_j + post_i * out for one token, with
// the n x n matrix and the n write weights as compile-time-unrolled registers.
// `streams` rows are `sd` floats apart (0 broadcasts one row to every stream).
// When `collapsed` is non-null the mean over the n rounded float32 streams
// (TwoStageCore._collapse) is written there as well, in double, one rounding.
template <int N>
inline void mhc_combine_token_n(const float* __restrict streams, int64_t sd, const float* __restrict res,
                                const float* __restrict post, const float* __restrict out,
                                float* __restrict new_streams, float* __restrict collapsed, int64_t d) noexcept {
    const double inv_n = 1.0 / static_cast<double>(N);
    int64_t k = 0;
#ifdef MONTLOK_VECTOR_EXT
    for (; k + 4 <= d; k += 4) {
        const v4d o = load4f_as_d(out + k);
        v4d s[N];
        for (int j = 0; j < N; ++j) {
            s[j] = load4f_as_d(streams + j * sd + k);
        }
        v4d mean = splat4d(0.0);
        for (int i = 0; i < N; ++i) {
            v4d acc = splat4d(static_cast<double>(post[i])) * o;
            for (int j = 0; j < N; ++j) {
                acc += splat4d(static_cast<double>(res[i * N + j])) * s[j];
            }
            store4d_as_f(new_streams + i * d + k, acc);
            if (collapsed != nullptr) {
                mean += load4f_as_d(new_streams + i * d + k);
            }
        }
        if (collapsed != nullptr) {
            store4d_as_f(collapsed + k, mean * splat4d(inv_n));
        }
    }
#endif
    for (; k < d; ++k) {
        double mean = 0.0;
        for (int i = 0; i < N; ++i) {
            double acc = static_cast<double>(post[i]) * static_cast<double>(out[k]);
            for (int j = 0; j < N; ++j) {
                acc += static_cast<double>(res[i * N + j]) * static_cast<double>(streams[j * sd + k]);
            }
            new_streams[i * d + k] = static_cast<float>(acc);
            mean += static_cast<double>(new_streams[i * d + k]);
        }
        if (collapsed != nullptr) {
            collapsed[k] = static_cast<float>(mean * inv_n);
        }
    }
}

inline void mhc_combine_token(const float* __restrict streams, int64_t sd, const float* __restrict res,
                              const float* __restrict post, const float* __restrict out,
                              float* __restrict new_streams, float* __restrict collapsed, int64_t n,
                              int64_t d) noexcept {
    switch (n) {
        case 1: return mhc_combine_token_n<1>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 2: return mhc_combine_token_n<2>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 3: return mhc_combine_token_n<3>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 4: return mhc_combine_token_n<4>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 5: return mhc_combine_token_n<5>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 6: return mhc_combine_token_n<6>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 7: return mhc_combine_token_n<7>(streams, sd, res, post, out, new_streams, collapsed, d);
        case 8: return mhc_combine_token_n<8>(streams, sd, res, post, out, new_streams, collapsed, d);
        default: break;
    }
    for (int64_t i = 0; i < n; ++i) {
        const float* __restrict row = res + i * n;
        float* __restrict dst = new_streams + i * d;
        const double post_i = static_cast<double>(post[i]);
        for (int64_t k = 0; k < d; ++k) {
            double acc = post_i * static_cast<double>(out[k]);
            for (int64_t j = 0; j < n; ++j) {
                acc += static_cast<double>(row[j]) * static_cast<double>(streams[j * sd + k]);
            }
            dst[k] = static_cast<float>(acc);
        }
    }
    if (collapsed != nullptr) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (int64_t k = 0; k < d; ++k) {
            double mean = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                mean += static_cast<double>(new_streams[i * d + k]);
            }
            collapsed[k] = static_cast<float>(mean * inv_n);
        }
    }
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

inline void mhc_combine_range(const MhcCombinePlan& c, int64_t begin, int64_t end) noexcept {
    for (int64_t token = begin; token < end; ++token) {
        mhc_combine_token(c.streams + token * c.token_stride, c.stream_stride, c.res + token * c.n * c.n,
                          c.post + token * c.n, c.out + token * c.d, c.new_streams + token * c.n * c.d,
                          c.collapsed != nullptr ? c.collapsed + token * c.d : nullptr, c.n, c.d);
    }
}

// Write step of one residual fused with the read step of the next: for each
// block of tokens the new streams are produced and immediately consumed by
// mhc_prepare_block while they are still in L1. `p.streams` must point at
// `c.new_streams` with contiguous strides.
inline void mhc_combine_prepare_range(const MhcCombinePlan& c, const MhcPlan& p, int64_t begin, int64_t end,
                                      double* __restrict scratch) noexcept {
    for (int64_t token = begin; token < end; token += kMhcTokenBlock) {
        const int64_t count = std::min<int64_t>(kMhcTokenBlock, end - token);
        mhc_combine_range(c, token, token + count);
        mhc_prepare_block(p, token, count, scratch);
    }
}

}  // namespace montlok
