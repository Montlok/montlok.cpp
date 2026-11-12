#!/usr/bin/env bash
# Builds and runs the kernel self-test (montlok_selftest.cpp):
#   1. for the host CPU, with the default 32-byte float vectors;
#   2. again with 64-byte vectors, which exercises the AVX-512 recurrence path
#      (the compiler splits the vectors on narrower hardware);
#   3. on arm64 macOS with Rosetta 2, an x86-64 build with the baseline and
#      the AVX2 variant (forced: Rosetta executes AVX2 code but hides it from
#      cpuid; its AVX-512 coverage is incomplete, so that variant stays skipped).
# Each run exits non-zero on a tolerance violation and this script stops there.
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-c++}
FLAGS=(-O3 -std=c++17 -fno-math-errno -ffp-contract=fast ${MONTLOK_SELFTEST_FLAGS:-})
SOURCES=(montlok_selftest.cpp montlok_isa.cpp montlok_isa_baseline.cpp montlok_isa_avx2.cpp montlok_isa_avx512.cpp)
OUT=${MONTLOK_SELFTEST_DIR:-build/selftest}
mkdir -p "$OUT"

echo "### host build"
"$CXX" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT/montlok_selftest"
"$OUT/montlok_selftest"

echo
echo "### host build with 64-byte vectors (AVX-512 recurrence layout)"
"$CXX" "${FLAGS[@]}" -DMONTLOK_VEC_BYTES=64 "${SOURCES[@]}" -o "$OUT/montlok_selftest_vec64"
"$OUT/montlok_selftest_vec64"

if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] && arch -x86_64 /usr/bin/true 2>/dev/null; then
    echo
    echo "### x86-64 build under Rosetta 2, AVX2 variant forced"
    "$CXX" -target x86_64-apple-macos12 "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT/montlok_selftest_x86"
    MONTLOK_SELFTEST_FORCE=avx2 arch -x86_64 "$OUT/montlok_selftest_x86"
fi
echo
echo "all self-test builds passed"
