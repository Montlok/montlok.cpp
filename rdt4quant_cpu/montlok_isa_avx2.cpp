// Copyright 2026 OpenAI.
// Apache-2.0. AVX2 + FMA variant of the montlok kernels (x86-64, GCC/Clang).
//
// The kernel headers are compiled under a target pragma instead of per-file
// compiler flags, so one flag set builds every variant with both the
// setuptools and the JIT path of torch.utils.cpp_extension. The standard
// headers and the ISA-independent montlok headers are included first, with the
// default target: a std:: template instantiated under the pragma would carry
// AVX code into a symbol that other translation units also define, and the
// linker could pick that copy for a machine without AVX.

// The ISA macros come before every include: montlok_types.h turns
// MONTLOK_ISA_NAMESPACE into the inline-namespace wrappers when it is first read.
#define MONTLOK_ISA_NAMESPACE isa_avx2
#define MONTLOK_ISA_NAME "avx2"
#define MONTLOK_ISA_LEVEL 1
#define MONTLOK_ISA_TABLE kernels_avx2
#define MONTLOK_ISA_FMA true
#define MONTLOK_ISA_AVX2 true
#define MONTLOK_ISA_AVX512 false

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "montlok_isa.h"
#include "montlok_types.h"

#if defined(MONTLOK_X86) && defined(MONTLOK_VECTOR_EXT)

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma")
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
const KernelTable* kernels_avx2() noexcept {
    return nullptr;
}
}  // namespace montlok

#endif
