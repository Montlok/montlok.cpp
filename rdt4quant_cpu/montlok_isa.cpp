// Copyright 2026 OpenAI.
// Apache-2.0. CPU feature detection and kernel variant selection (montlok_isa.h).
// Compiled for the baseline target like the bindings; torch-free.

#include "montlok_isa.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(MONTLOK_X86)
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace montlok {

namespace {

#if defined(MONTLOK_X86)
void cpuid_leaf(unsigned leaf, unsigned subleaf, unsigned out[4]) noexcept {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    std::memcpy(out, regs, sizeof(regs));
#else
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
#endif
}

uint64_t xcr0_value() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0;
    unsigned hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}
#endif  // MONTLOK_X86

void copy_name(char* dst, size_t size, const char* src) noexcept {
    std::strncpy(dst, src, size - 1);
    dst[size - 1] = '\0';
}

std::atomic<const KernelTable*>& active_slot() noexcept {
    static std::atomic<const KernelTable*> slot{nullptr};
    return slot;
}

}  // namespace

CpuFeatures detect_cpu_features() noexcept {
    CpuFeatures f{};
#if defined(MONTLOK_X86)
    f.x86 = true;
    unsigned r[4] = {0, 0, 0, 0};
    cpuid_leaf(0, 0, r);
    const unsigned max_leaf = r[0];
    cpuid_leaf(0x80000000u, 0, r);
    if (r[0] >= 0x80000004u) {
        char brand[49] = {0};
        for (unsigned i = 0; i < 3; ++i) {
            cpuid_leaf(0x80000002u + i, 0, r);
            std::memcpy(brand + 16 * i, r, 16);
        }
        const char* start = brand;
        while (*start == ' ') {
            ++start;
        }
        copy_name(f.brand, sizeof(f.brand), start);
    } else {
        copy_name(f.brand, sizeof(f.brand), "x86-64");
    }
    if (max_leaf >= 1) {
        cpuid_leaf(1, 0, r);
        f.sse2 = (r[3] & (1u << 26)) != 0;
        f.sse4_2 = (r[2] & (1u << 20)) != 0;
        f.fma = (r[2] & (1u << 12)) != 0;
        f.avx = (r[2] & (1u << 28)) != 0;
        const bool osxsave = (r[2] & (1u << 27)) != 0;
        if (osxsave) {
            const uint64_t xcr0 = xcr0_value();
            f.os_avx = (xcr0 & 0x6u) == 0x6u;
            f.os_avx512 = f.os_avx && (xcr0 & 0xe0u) == 0xe0u;
        }
    }
    if (max_leaf >= 7) {
        cpuid_leaf(7, 0, r);
        f.avx2 = (r[1] & (1u << 5)) != 0;
        f.avx512f = (r[1] & (1u << 16)) != 0;
        f.avx512dq = (r[1] & (1u << 17)) != 0;
        f.avx512bw = (r[1] & (1u << 30)) != 0;
        f.avx512vl = (r[1] & (1u << 31)) != 0;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    f.arm64 = true;
    copy_name(f.brand, sizeof(f.brand), "arm64 (NEON)");
#else
    copy_name(f.brand, sizeof(f.brand), "generic");
#endif
    return f;
}

const CpuFeatures& host_cpu() noexcept {
    static const CpuFeatures cpu = detect_cpu_features();
    return cpu;
}

bool cpu_supports(const CpuFeatures& cpu, const KernelTable& table) noexcept {
    switch (table.level) {
        case 0:
            return true;
        case 1:
            return cpu.x86 && cpu.avx && cpu.avx2 && cpu.fma && cpu.os_avx;
        case 2:
            return cpu.x86 && cpu.avx2 && cpu.fma && cpu.avx512f && cpu.avx512dq && cpu.avx512bw && cpu.avx512vl &&
                   cpu.os_avx512;
        default:
            return false;
    }
}

int available_kernels(const KernelTable** out, int capacity) noexcept {
    const KernelTable* all[kMaxKernelTables] = {kernels_baseline(), kernels_avx2(), kernels_avx512()};
    int count = 0;
    for (const KernelTable* table : all) {
        if (table != nullptr && count < capacity) {
            out[count++] = table;
        }
    }
    return count;
}

const KernelTable* find_kernels(const char* name) noexcept {
    const KernelTable* tables[kMaxKernelTables];
    const int count = available_kernels(tables, kMaxKernelTables);
    for (int i = 0; i < count; ++i) {
        if (std::strcmp(tables[i]->name, name) == 0) {
            return tables[i];
        }
    }
    return nullptr;
}

KernelSelection select_kernels(const CpuFeatures& cpu, const char* request) noexcept {
    KernelSelection selection{};
    const KernelTable* tables[kMaxKernelTables];
    const int count = available_kernels(tables, kMaxKernelTables);
    selection.table = tables[0];
    for (int i = 0; i < count; ++i) {
        if (cpu_supports(cpu, *tables[i])) {
            selection.table = tables[i];  // ascending level: the last supported one wins
        }
    }
    if (request != nullptr && request[0] != '\0') {
        bool force = false;
        const char* name = request;
        if (std::strncmp(request, "force:", 6) == 0) {
            force = true;
            name += 6;
        }
        const KernelTable* wanted = find_kernels(name);
        if (wanted == nullptr) {
            std::snprintf(selection.message, sizeof(selection.message),
                          "MONTLOK_ISA=%s names no variant of this build; using %s", request, selection.table->name);
        } else if (!force && !cpu_supports(cpu, *wanted)) {
            std::snprintf(selection.message, sizeof(selection.message),
                          "MONTLOK_ISA=%s is not supported by this CPU (%s); using %s", request, cpu.brand,
                          selection.table->name);
        } else {
            selection.table = wanted;
        }
    }
    if (selection.table->level == 0 && cpu.x86 &&
        ((selection.table->avx512 && !cpu.avx512f) || (selection.table->avx2 && !cpu.avx2))) {
        std::snprintf(selection.message, sizeof(selection.message),
                      "the baseline kernels were compiled for %s but this CPU (%s) lacks it; rebuild without "
                      "MONTLOK_MARCH",
                      selection.table->avx512 ? "AVX-512" : "AVX2", cpu.brand);
    }
    return selection;
}

const KernelTable& active_kernels() noexcept {
    std::atomic<const KernelTable*>& slot = active_slot();
    const KernelTable* table = slot.load(std::memory_order_acquire);
    if (table == nullptr) {
        const KernelSelection selection = select_kernels(host_cpu(), std::getenv("MONTLOK_ISA"));
        if (selection.message[0] != '\0') {
            std::fprintf(stderr, "montlok: %s\n", selection.message);
        }
        const KernelTable* expected = nullptr;
        table = slot.compare_exchange_strong(expected, selection.table, std::memory_order_acq_rel) ? selection.table
                                                                                                   : expected;
    }
    return *table;
}

const KernelTable* set_active_kernels(const char* name, bool force, char* message, size_t message_size) noexcept {
    if (message != nullptr && message_size > 0) {
        message[0] = '\0';
    }
    const KernelTable* table = name != nullptr ? find_kernels(name) : nullptr;
    if (table == nullptr) {
        if (message != nullptr) {
            std::snprintf(message, message_size, "'%s' names no kernel variant of this build", name ? name : "");
        }
        return nullptr;
    }
    if (!force && !cpu_supports(host_cpu(), *table)) {
        if (message != nullptr) {
            std::snprintf(message, message_size, "kernel variant '%s' is not supported by this CPU (%s)", name,
                          host_cpu().brand);
        }
        return nullptr;
    }
    active_slot().store(table, std::memory_order_release);
    return table;
}

}  // namespace montlok
