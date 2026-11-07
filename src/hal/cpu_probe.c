/**
 * @file cpu_probe.c
 * @brief Dynamic Runtime CPU Capability & SIMD ISA Probing
 * 
 * Satisfies HAL-REQ-01.
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#include <cpuid.h>
#endif

static atomic_bool s_probed = false;
static uint32_t s_features = 0;
static char s_brand[128] = "Unknown CPU";
static RCRE_CpuTopology s_topology = { 1, 1, 0, 1, 0 };

static uint32_t probe_env_positive_u32(const char* name) {
    const char* v = getenv(name);
    if (!v || v[0] == '\0') return 0;
    char* end = nullptr;
    unsigned long parsed = strtoul(v, &end, 10);
    if (end == v || parsed == 0 || parsed > 65535UL) return 0;
    return (uint32_t)parsed;
}

#if defined(__linux__)
static uint32_t probe_linux_cgroup_quota_cores(uint32_t host_cores) {
    /* cgroup v2: /sys/fs/cgroup/cpu.max format: "<quota_us> <period_us>" or "max 100000" */
    FILE* f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char quota_tok[32] = {0};
        unsigned long long period = 0;
        if (fscanf(f, "%31s %llu", quota_tok, &period) == 2 && period > 0) {
            if (strcmp(quota_tok, "max") != 0) {
                unsigned long long quota = strtoull(quota_tok, nullptr, 10);
                if (quota > 0) {
                    fclose(f);
                    uint32_t c = (uint32_t)((quota + period - 1ULL) / period);
                    if (c == 0) c = 1;
                    return (c < host_cores) ? c : host_cores;
                }
            }
        }
        fclose(f);
    }
    return host_cores;
}
#endif

static void probe_topology(void) {
    uint32_t logical = 1;
#if defined(_SC_NPROCESSORS_ONLN)
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0) logical = (uint32_t)nprocs;
#endif

    uint32_t perf = logical;
    uint32_t eff  = 0;
    uint32_t quota = logical;
    uint64_t ram_bytes = 0;

#if defined(__APPLE__)
    int p_val = 0, e_val = 0;
    size_t sz_p = sizeof(p_val), sz_e = sizeof(e_val);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &p_val, &sz_p, nullptr, 0) == 0 && p_val > 0) {
        perf = (uint32_t)p_val;
        if (sysctlbyname("hw.perflevel1.physicalcpu", &e_val, &sz_e, nullptr, 0) == 0 && e_val >= 0) {
            eff = (uint32_t)e_val;
        }
    }
    uint64_t mem64 = 0;
    size_t sz_mem = sizeof(mem64);
    if (sysctlbyname("hw.memsize", &mem64, &sz_mem, nullptr, 0) == 0) {
        ram_bytes = mem64;
    }
#elif defined(__linux__)
    quota = probe_linux_cgroup_quota_cores(logical);
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_sz = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_sz > 0) {
        ram_bytes = (uint64_t)pages * (uint64_t)page_sz;
    }
#endif

    /* Respect HPC scheduler per-allocation limits if narrower than cgroup/host */
    uint32_t slurm_cpus = probe_env_positive_u32("SLURM_CPUS_ON_NODE");
    if (slurm_cpus == 0) {
        slurm_cpus = probe_env_positive_u32("SLURM_CPUS_PER_TASK");
    }
    if (slurm_cpus > 0 && slurm_cpus < quota) {
        quota = slurm_cpus;
    }

    s_topology.logical_cores         = logical;
    s_topology.perf_cores            = perf;
    s_topology.eff_cores             = eff;
    s_topology.effective_quota_cores = (quota > 0) ? quota : 1;
    s_topology.total_ram_bytes       = ram_bytes;
}

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

    probe_topology();
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

RCRE_CpuTopology rcre_cpu_get_topology(void) {
    if (!atomic_load(&s_probed)) {
        probe_cpu();
    }
    return s_topology;
}

int rcre_cpu_get_memory_pressure(void) {
#if defined(__APPLE__)
    /* XNU memorystatus levels: 1 = NORMAL, 2 = WARN, 4 = CRITICAL */
    int level = 1;
    size_t sz = sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &sz, nullptr, 0) == 0) {
        if (level >= 4) return 2;
        if (level >= 2) return 1;
    }
    return 0;
#elif defined(__linux__)
    /* Check Linux PSI (/proc/pressure/memory) first */
    FILE* psi = fopen("/proc/pressure/memory", "r");
    if (psi) {
        char line[128];
        double avg10 = 0.0;
        while (fgets(line, sizeof(line), psi)) {
            if (strncmp(line, "full ", 5) == 0 && sscanf(line, "full avg10=%lf", &avg10) == 1) {
                if (avg10 >= 20.0) { fclose(psi); return 2; }
                if (avg10 >= 5.0)  { fclose(psi); return 1; }
            }
        }
        fclose(psi);
    }
    /* Fallback to MemAvailable / MemTotal ratio in /proc/meminfo */
    FILE* mi = fopen("/proc/meminfo", "r");
    if (mi) {
        char key[64];
        unsigned long long val = 0;
        unsigned long long mem_total = 0, mem_avail = 0;
        char unit[32];
        while (fscanf(mi, "%63s %llu %31s", key, &val, unit) >= 2) {
            if (strcmp(key, "MemTotal:") == 0) mem_total = val;
            else if (strcmp(key, "MemAvailable:") == 0) mem_avail = val;
            if (mem_total > 0 && mem_avail > 0) break;
        }
        fclose(mi);
        if (mem_total > 0 && mem_avail > 0) {
            if (mem_avail * 25ULL < mem_total) return 2; /* < 4% free */
            if (mem_avail * 10ULL < mem_total) return 1; /* < 10% free */
        }
    }
    return 0;
#else
    return 0;
#endif
}
