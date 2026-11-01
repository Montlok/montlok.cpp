// Copyright 2026 OpenAI.
// Apache-2.0. Instruction-set variants of the montlok kernels and the runtime
// selection between them.
//
// One build of the extension runs on every CPU of its architecture. On x86-64
// the kernel headers are compiled three times, each in its own translation
// unit and inline namespace: a baseline that only needs what the compiler's
// default target offers (plain x86-64, i.e. SSE2, unless MONTLOK_MARCH says
// otherwise), an AVX2+FMA variant and an AVX-512 variant. On arm64 there is
// the NEON baseline; any other architecture gets the baseline built from the
// generic vector extensions (or scalar code). At run time cpuid plus the OS
// state in XCR0 pick the highest variant the machine supports; the MONTLOK_ISA
// environment variable or set_active_kernels() override the choice
// (`avx2`, `baseline`, ...; `force:<name>` skips the CPU check, for emulators
// that execute instructions they do not advertise).
//
// Torch-free: the self-test links the same variant translation units.

#pragma once

#include <cstddef>
#include <cstdint>

#include "montlok_types.h"

namespace montlok {

// Entry points of one compiled variant. Every pointer has the signature of
// the inline kernel of the same name in montlok_kernel.h / montlok_mhc.h /
// montlok_layers.h.
struct KernelTable {
    const char* name;   // "baseline" (x86-64), "neon" (arm64), "avx2", "avx512"
    int level;          // 0 baseline, 1 AVX2+FMA, 2 AVX-512; the highest supported level is used
    int vec_bytes;      // MONTLOK_VEC_BYTES the variant was compiled with
    bool vector_ext;    // compiled with GCC/Clang vector extensions (false: scalar code only)
    bool fma;           // the compiler could fuse multiply-adds (changes rounding)
    bool avx2;
    bool avx512;
    void (*siso_run_work_item)(const SisoPlan& plan, int64_t item, float* scratch);
    void (*prep_token)(const PrepPlan& plan, int64_t b, int64_t t, float* scratch);
    void (*prep_chain)(const PrepPlan& plan, int64_t chain);
    void (*prep_rotate)(const PrepPlan& plan, int64_t b, int64_t t);
    void (*mhc_prepare_range)(const MhcPlan& plan, int64_t begin, int64_t end, double* scratch);
    void (*mhc_combine_range)(const MhcCombinePlan& plan, int64_t begin, int64_t end);
    void (*mhc_combine_collapse_range)(const MhcCombinePlan& plan, int64_t begin, int64_t end, float* scratch);
    void (*mhc_combine_prepare_range)(const MhcCombinePlan& combine, const MhcPlan& prepare, int64_t begin,
                                      int64_t end, double* scratch);
    void (*mhc_combine_prepare_inplace_range)(const MhcCombinePlan& combine, const MhcPlan& prepare, int64_t begin,
                                              int64_t end, double* scratch);
    void (*rms_norm_rows)(const float* x, int64_t row_stride, const float* w, double eps, int64_t dim, int64_t groups,
                          int64_t row_begin, int64_t row_end, float* out);
    void (*add_rms_norm_rows)(const float* left, const float* right, const float* w, double eps, int64_t dim,
                              int64_t groups, int64_t row_begin, int64_t row_end, float* residual, float* normalized);
    void (*swiglu_rows)(const float* x, int64_t row_stride, int64_t hidden, int64_t row_begin, int64_t row_end,
                        float* out);
    void (*layer_norm_rows)(const float* x, int64_t row_stride, const float* w, const float* bias, double eps,
                            int64_t dim, int64_t row_begin, int64_t row_end, float* out);
    void (*mla_rope_range)(const MlaRopePlan& plan, int64_t begin, int64_t end, float* scratch);
};

struct CpuFeatures {
    bool x86;
    bool arm64;
    bool sse2;
    bool sse4_2;
    bool avx;
    bool fma;
    bool avx2;
    bool avx512f;
    bool avx512dq;
    bool avx512bw;
    bool avx512vl;
    bool os_avx;      // XCR0 has the XMM and YMM state enabled
    bool os_avx512;   // ... and the opmask / ZMM state
    char brand[64];
};

CpuFeatures detect_cpu_features() noexcept;
// Features of this machine, detected once.
const CpuFeatures& host_cpu() noexcept;
bool cpu_supports(const CpuFeatures& cpu, const KernelTable& table) noexcept;

// The variants compiled into this build. kernels_baseline() always exists; the
// x86 variants return nullptr on other architectures and on compilers without
// the target pragma (MSVC).
const KernelTable* kernels_baseline() noexcept;
const KernelTable* kernels_avx2() noexcept;
const KernelTable* kernels_avx512() noexcept;

constexpr int kMaxKernelTables = 3;
// Fills `out` with the available tables in ascending level; returns the count.
int available_kernels(const KernelTable** out, int capacity) noexcept;
const KernelTable* find_kernels(const char* name) noexcept;

struct KernelSelection {
    const KernelTable* table;
    char message[256];  // empty, or why a request was not honoured / a build-time warning
};

// The highest-level variant `cpu` supports, or the one `request` names
// ("avx2", "baseline", ...; "force:<name>" skips the CPU check; null or empty
// for the automatic choice).
KernelSelection select_kernels(const CpuFeatures& cpu, const char* request) noexcept;

// Process-wide choice: select_kernels(host_cpu(), getenv("MONTLOK_ISA")) on
// first use, printing the selection message to stderr if there is one.
const KernelTable& active_kernels() noexcept;

// Switches the process-wide choice (A/B tests, benchmarks). Returns the table
// now in use, or nullptr with `message` filled when `name` is unknown or, without
// `force`, unsupported by this CPU.
const KernelTable* set_active_kernels(const char* name, bool force, char* message, size_t message_size) noexcept;

}  // namespace montlok
