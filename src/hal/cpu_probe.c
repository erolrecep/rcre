/**
 * @file cpu_probe.c
 * @brief Dynamic Runtime CPU Capability & SIMD ISA Probing
 * 
 * Satisfies HAL-REQ-01.
 */

#include "rcre.h"
#include <string.h>
#include <stdatomic.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#include <cpuid.h>
#endif

static atomic_bool s_probed = false;
static uint32_t s_features = 0;
static char s_brand[128] = "Unknown CPU";

static void probe_cpu(void) {
    uint32_t feats = RCRE_CPU_FEAT_SCALAR;

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    feats |= RCRE_CPU_FEAT_NEON;
#endif

#if defined(__wasm_simd128__)
    feats |= RCRE_CPU_FEAT_WASM_SIMD;
#endif

#if defined(__APPLE__)
    size_t len = sizeof(s_brand);
    if (sysctlbyname("machdep.cpu.brand_string", s_brand, &len, NULL, 0) != 0) {
        strncpy(s_brand, "Apple Silicon", sizeof(s_brand) - 1);
    }
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && !defined(_MSC_VER)
    /* x86 CPUID feature detection */
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1 << 12)) {
            feats |= RCRE_CPU_FEAT_FMA;
        }
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 5)) {
            feats |= RCRE_CPU_FEAT_AVX2;
        }
        if (ebx & (1 << 16)) {
            feats |= RCRE_CPU_FEAT_AVX512F;
        }
    }

    /* Probe brand string via extended cpuid leaves */
    unsigned int brand_regs[12];
    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) && eax >= 0x80000004) {
        __get_cpuid(0x80000002, &brand_regs[0], &brand_regs[1], &brand_regs[2], &brand_regs[3]);
        __get_cpuid(0x80000003, &brand_regs[4], &brand_regs[5], &brand_regs[6], &brand_regs[7]);
        __get_cpuid(0x80000004, &brand_regs[8], &brand_regs[9], &brand_regs[10], &brand_regs[11]);
        memcpy(s_brand, brand_regs, sizeof(brand_regs));
        s_brand[sizeof(s_brand) - 1] = '\0';
    } else {
        strncpy(s_brand, "Generic x86 CPU", sizeof(s_brand) - 1);
    }
#elif defined(__EMSCRIPTEN__)
    strncpy(s_brand, "WebAssembly Runtime", sizeof(s_brand) - 1);
#else
    strncpy(s_brand, "Generic Host CPU", sizeof(s_brand) - 1);
#endif

    s_features = feats;
    atomic_store(&s_probed, true);
}

uint32_t rcre_cpu_get_features(void) {
    if (!atomic_load(&s_probed)) {
        probe_cpu();
    }
    return s_features;
}

const char* rcre_cpu_get_brand(void) {
    if (!atomic_load(&s_probed)) {
        probe_cpu();
    }
    return s_brand;
}
