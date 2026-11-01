// Copyright 2026 OpenAI.
// Apache-2.0. AVX-512 variant of the montlok kernels (x86-64, GCC/Clang).
//
// Same structure as montlok_isa_avx2.cpp. The recurrence is compiled with
// 64-byte float vectors (one zmm register per 16-lane headdim block, half the
// instructions of the AVX2 variant); the double-precision mHC and layer
// kernels keep their 4-lane accumulation so every variant sums in the same
// order. Requires AVX-512 F/VL/BW/DQ, i.e. every AVX-512 CPU since Skylake-X
// and Zen 4; the runtime also checks that the OS saves the ZMM state.

#define MONTLOK_VEC_BYTES 64

// The ISA macros come before every include: montlok_types.h turns
// MONTLOK_ISA_NAMESPACE into the inline-namespace wrappers when it is first read.
#define MONTLOK_ISA_NAMESPACE isa_avx512
#define MONTLOK_ISA_NAME "avx512"
#define MONTLOK_ISA_LEVEL 2
#define MONTLOK_ISA_TABLE kernels_avx512
#define MONTLOK_ISA_FMA true
#define MONTLOK_ISA_AVX2 true
#define MONTLOK_ISA_AVX512 true

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "montlok_isa.h"
#include "montlok_types.h"

#if defined(MONTLOK_X86) && defined(MONTLOK_VECTOR_EXT)

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,avx512dq,avx2,fma"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx512f,avx512vl,avx512bw,avx512dq,avx2,fma")
#endif

#include "montlok_kernel.h"
#include "montlok_layers.h"
#include "montlok_mhc.h"

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#include "montlok_isa_table.inc"

#else

namespace montlok {
const KernelTable* kernels_avx512() noexcept {
    return nullptr;
}
}  // namespace montlok

#endif
