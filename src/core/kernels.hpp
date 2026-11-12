// Copyright 2026 Montlok. Apache-2.0.
// Portable float64 kernels of the model families.
//
// Weights are float32 when their values are exactly float32 (PyTorch exports)
// and float64 otherwise; activations are double. Every dot product is the
// double-precision sum of exact products in a fixed association: four lanes
// over k in steps of 4, a pairwise horizontal sum, the scalar tail in order,
// then the bias. Blocked and unblocked paths share that association, so
// results are identical for every batch size, blocking and thread count.
//
// sigmoid and tanh are built on the vectorized exp below (Cody-Waite reduction
// and a degree-13 Taylor polynomial, within a few ulp of the correctly rounded
// result), which gives the same bits on every platform. GELU, ELU and softplus
// use libm.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace montlok::core::kernels {

#if defined(__GNUC__) || defined(__clang__)
#define MONTLOK_CORE_VECTOR 1
typedef double v4d __attribute__((vector_size(32)));
typedef float v4f __attribute__((vector_size(16)));
typedef uint64_t v4u __attribute__((vector_size(32)));
#else
// Scalar stand-in with the same lane structure.
struct v4d {
    double v[4];
    double& operator[](int i) { return v[i]; }
    double operator[](int i) const { return v[i]; }
};
#endif

inline v4d load4(const double* p) noexcept {
    v4d r;
    std::memcpy(&r, p, sizeof(r));
    return r;
}

inline v4d load4(const float* p) noexcept {
#ifdef MONTLOK_CORE_VECTOR
    v4f f;
    std::memcpy(&f, p, sizeof(f));
    return __builtin_convertvector(f, v4d);
#else
    v4d r;
    for (int i = 0; i < 4; ++i) r[i] = p[i];
    return r;
#endif
}

inline void store4(double* p, const v4d& v) noexcept { std::memcpy(p, &v, sizeof(v)); }

inline v4d splat(double x) noexcept {
#ifdef MONTLOK_CORE_VECTOR
    return v4d{x, x, x, x};
#else
    return v4d{{x, x, x, x}};
#endif
}

inline double hsum(const v4d& v) noexcept { return (v[0] + v[2]) + (v[1] + v[3]); }

#ifndef MONTLOK_CORE_VECTOR
inline v4d operator+(v4d a, const v4d& b) noexcept {
    for (int i = 0; i < 4; ++i) a[i] += b[i];
    return a;
}
inline v4d operator*(v4d a, const v4d& b) noexcept {
    for (int i = 0; i < 4; ++i) a[i] *= b[i];
    return a;
}
inline v4d& operator+=(v4d& a, const v4d& b) noexcept { return a = a + b; }
inline v4d operator-(v4d a, const v4d& b) noexcept {
    for (int i = 0; i < 4; ++i) a[i] -= b[i];
    return a;
}
inline v4d operator/(v4d a, const v4d& b) noexcept {
    for (int i = 0; i < 4; ++i) a[i] /= b[i];
    return a;
}
#endif

// ---------------------------------------------------------------------------
// Dense products. W is [rows, cols] row-major (float or double).
// ---------------------------------------------------------------------------

template <typename W>
inline double dot(const W* w, const double* x, size_t n) noexcept {
    v4d acc = splat(0.0);
    size_t k = 0;
    for (; k + 4 <= n; k += 4) acc += load4(w + k) * load4(x + k);
    double s = hsum(acc);
    for (; k < n; ++k) s += static_cast<double>(w[k]) * x[k];
    return s;
}

// y[r] = W[r, :] . x (+ bias[r])
template <typename W>
inline void matvec(const W* w, const double* bias, size_t rows, size_t cols, const double* x, double* y) noexcept {
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const W* w0 = w + r * cols;
        const W* w1 = w0 + cols;
        const W* w2 = w1 + cols;
        const W* w3 = w2 + cols;
        v4d a0 = splat(0.0), a1 = a0, a2 = a0, a3 = a0;
        size_t k = 0;
        for (; k + 4 <= cols; k += 4) {
            const v4d xv = load4(x + k);
            a0 += load4(w0 + k) * xv;
            a1 += load4(w1 + k) * xv;
            a2 += load4(w2 + k) * xv;
            a3 += load4(w3 + k) * xv;
        }
        double s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
        for (; k < cols; ++k) {
            const double xk = x[k];
            s0 += static_cast<double>(w0[k]) * xk;
            s1 += static_cast<double>(w1[k]) * xk;
            s2 += static_cast<double>(w2[k]) * xk;
            s3 += static_cast<double>(w3[k]) * xk;
        }
        if (bias) {
            s0 += bias[r];
            s1 += bias[r + 1];
            s2 += bias[r + 2];
            s3 += bias[r + 3];
        }
        y[r] = s0;
        y[r + 1] = s1;
        y[r + 2] = s2;
        y[r + 3] = s3;
    }
    for (; r < rows; ++r) y[r] = dot(w + r * cols, x, cols) + (bias ? bias[r] : 0.0);
}

// Y[i, r] = X[i, :] . W[r, :] (+ bias[r]) for m input rows: two input rows
// share every weight load, four weight rows share every input load.
template <typename W>
inline void gemm(const W* w, const double* bias, size_t rows, size_t cols, const double* x, size_t ldx, size_t m,
                 double* y, size_t ldy) noexcept {
    size_t i = 0;
    for (; i + 2 <= m; i += 2) {
        const double* x0 = x + i * ldx;
        const double* x1 = x0 + ldx;
        double* y0 = y + i * ldy;
        double* y1 = y0 + ldy;
        size_t r = 0;
        for (; r + 4 <= rows; r += 4) {
            const W* wr[4] = {w + r * cols, w + (r + 1) * cols, w + (r + 2) * cols, w + (r + 3) * cols};
            v4d a[4][2];
            for (auto& row : a) row[0] = row[1] = splat(0.0);
            size_t k = 0;
            for (; k + 4 <= cols; k += 4) {
                const v4d xv0 = load4(x0 + k);
                const v4d xv1 = load4(x1 + k);
                for (int q = 0; q < 4; ++q) {
                    const v4d wv = load4(wr[q] + k);
                    a[q][0] += wv * xv0;
                    a[q][1] += wv * xv1;
                }
            }
            for (int q = 0; q < 4; ++q) {
                double s0 = hsum(a[q][0]), s1 = hsum(a[q][1]);
                for (size_t t = k; t < cols; ++t) {
                    const double wt = static_cast<double>(wr[q][t]);
                    s0 += wt * x0[t];
                    s1 += wt * x1[t];
                }
                if (bias) {
                    s0 += bias[r + q];
                    s1 += bias[r + q];
                }
                y0[r + q] = s0;
                y1[r + q] = s1;
            }
        }
        for (; r < rows; ++r) {
            const double b = bias ? bias[r] : 0.0;
            y0[r] = dot(w + r * cols, x0, cols) + b;
            y1[r] = dot(w + r * cols, x1, cols) + b;
        }
    }
    for (; i < m; ++i) matvec(w, bias, rows, cols, x + i * ldx, y + i * ldy);
}

// ---------------------------------------------------------------------------
// exp / sigmoid / tanh
// ---------------------------------------------------------------------------

#ifdef MONTLOK_CORE_VECTOR
// mask lanes are all ones or all zeros (a vector comparison result).
template <typename M>
inline v4d vselect(M mask, v4d yes, v4d no) noexcept {
    return (v4d)((mask & (M)yes) | (~mask & (M)no));
}
inline v4d vclamp(v4d x, double lo, double hi) noexcept {
    x = vselect(x < splat(lo), splat(lo), x);  // NaN compares false and stays
    return vselect(x > splat(hi), splat(hi), x);
}
inline v4d vabs(v4d x) noexcept {
    return (v4d)((v4u)x & (v4u{} + 0x7fffffffffffffffull));
}
#else
inline v4d vclamp(v4d x, double lo, double hi) noexcept {
    for (int i = 0; i < 4; ++i) x[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
    return x;
}
inline v4d vabs(v4d x) noexcept {
    for (int i = 0; i < 4; ++i) x[i] = std::fabs(x[i]);
    return x;
}
#endif

// Taylor series of exp(r) - 1 to degree 13; |r| <= ln(2)/2.
inline v4d expm1_poly(v4d r) noexcept {
    static constexpr double c[13] = {
        1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0,
        1.0 / 40320.0,      1.0 / 5040.0,      1.0 / 720.0,      1.0 / 120.0,     1.0 / 24.0,
        1.0 / 6.0,          0.5,               1.0,
    };
    v4d p = splat(c[0]);
    for (int i = 1; i < 13; ++i) p = p * r + splat(c[i]);
    return p * r;
}

// exp(x), x clamped to [-708, 708]; NaN stays NaN.
inline v4d vexp(v4d x) noexcept {
    constexpr double kInvLn2 = 1.4426950408889634074;
    constexpr double kLn2Hi = 6.93147180369123816490e-01;
    constexpr double kLn2Lo = 1.90821492927058770002e-10;
    constexpr double kShifter = 6755399441055744.0;  // 1.5 * 2^52: round to nearest integer
    x = vclamp(x, -708.0, 708.0);
    const v4d shifted = x * splat(kInvLn2) + splat(kShifter);
    const v4d n = shifted - splat(kShifter);
    const v4d r = (x - n * splat(kLn2Hi)) - n * splat(kLn2Lo);
    const v4d p = expm1_poly(r) + splat(1.0);
#ifdef MONTLOK_CORE_VECTOR
    // The integer n sits in the low bits of `shifted`; 2^n has exponent n + 1023.
    const v4u bits = (v4u)shifted - (v4u)splat(kShifter);
    const v4d scale = (v4d)((bits + 1023u) << 52);
    return p * scale;
#else
    v4d out;
    for (int i = 0; i < 4; ++i) out[i] = std::ldexp(p[i], static_cast<int>(n[i]));
    return out;
#endif
}

inline v4d vsigmoid(v4d x) noexcept { return splat(1.0) / (splat(1.0) + vexp(splat(0.0) - x)); }

// tanh(x) = e / (e + 2) with e = expm1(2x); expm1 is the series below
// |2x| = ln(2)/2 and exp(2x) - 1 above, where the cancellation is mild.
inline v4d vtanh(v4d x) noexcept {
    const v4d y = x + x;
    const v4d e_big = vexp(y) - splat(1.0);
    const v4d e_small = expm1_poly(vclamp(y, -0.34657359027997264, 0.34657359027997264));
#ifdef MONTLOK_CORE_VECTOR
    const v4d e = vselect(vabs(y) < splat(0.34657359027997264), e_small, e_big);
#else
    v4d e;
    for (int i = 0; i < 4; ++i) e[i] = std::fabs(y[i]) < 0.34657359027997264 ? e_small[i] : e_big[i];
#endif
    return e / (e + splat(2.0));
}

// Applies f to n values in place; the tail goes through the same vector code.
template <typename F>
inline void map4(double* v, size_t n, F f) noexcept {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) store4(v + i, f(load4(v + i)));
    if (i < n) {
        double tail[4] = {0, 0, 0, 0};
        std::memcpy(tail, v + i, (n - i) * sizeof(double));
        store4(tail, f(load4(tail)));
        std::memcpy(v + i, tail, (n - i) * sizeof(double));
    }
}

inline void sigmoid_n(double* v, size_t n) noexcept { map4(v, n, [](v4d x) { return vsigmoid(x); }); }
inline void tanh_n(double* v, size_t n) noexcept { map4(v, n, [](v4d x) { return vtanh(x); }); }

inline double sigmoid(double x) noexcept {
    double v[4] = {x, 0, 0, 0};
    store4(v, vsigmoid(load4(v)));
    return v[0];
}

// ---------------------------------------------------------------------------
// libm activations (PyTorch definitions)
// ---------------------------------------------------------------------------

inline double gelu(double x) noexcept { return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440)); }

inline double gelu_tanh(double x) noexcept {
    const double inner = 0.79788456080286535588 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
}

// torch.nn.functional.softplus with beta=1, threshold=20.
inline double softplus(double x) noexcept { return x > 20.0 ? x : std::log1p(std::exp(x)); }

inline double elu(double x, double alpha) noexcept { return x > 0 ? x : alpha * std::expm1(x); }

inline double leaky_relu(double x, double slope) noexcept { return x > 0 ? x : x * slope; }

}  // namespace montlok::core::kernels
