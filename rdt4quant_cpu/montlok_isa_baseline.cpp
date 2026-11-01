// Copyright 2026 OpenAI.
// Apache-2.0. Baseline instruction-set variant of the montlok kernels.
//
// Compiled with the command-line target only: plain x86-64 (SSE2) unless
// MONTLOK_MARCH says otherwise, NEON on arm64, and the generic GCC/Clang vector
// extensions (or scalar code on other compilers) elsewhere. Always present;
// the runtime falls back to it when no better variant fits the CPU.

// The ISA macros come before every include: montlok_types.h turns
// MONTLOK_ISA_NAMESPACE into the inline-namespace wrappers when it is first read.
#define MONTLOK_ISA_NAMESPACE isa_baseline
#if defined(__aarch64__) || defined(_M_ARM64)
#define MONTLOK_ISA_NAME "neon"
#else
#define MONTLOK_ISA_NAME "baseline"
#endif
#define MONTLOK_ISA_LEVEL 0
#define MONTLOK_ISA_TABLE kernels_baseline
#if defined(__FMA__) || defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_FEATURE_FMA)
#define MONTLOK_ISA_FMA true
#else
#define MONTLOK_ISA_FMA false
#endif
#if defined(__AVX2__)
#define MONTLOK_ISA_AVX2 true
#else
#define MONTLOK_ISA_AVX2 false
#endif
#if defined(__AVX512F__)
#define MONTLOK_ISA_AVX512 true
#else
#define MONTLOK_ISA_AVX512 false
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "montlok_isa.h"
#include "montlok_types.h"


#include "montlok_kernel.h"
#include "montlok_layers.h"
#include "montlok_mhc.h"

#include "montlok_isa_table.inc"
