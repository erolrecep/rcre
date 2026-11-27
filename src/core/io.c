/**
 * @file io.c
 * @brief RCRE High-Performance Scientific Data I/O Engine (ISO C23)
 *
 * Implements IO-REQ-01 through IO-REQ-06:
 *  - Cross-platform virtual memory file mapping (mmap / madvise / F_NOCACHE)
 *  - Atomic file persistence with telemetry hooks into rcre_prof
 *  - Clinger / Eisel-Lemire 0-ULP fast float32 slice parser
 *  - SIMD-accelerated & threadpool-parallel CSV/TSV tabular loader
 *  - NumPy (.npy v1.0 & v2.0) 64-byte aligned writer, loader & zero-copy viewer
 *  - Hugging Face (.safetensors) zero-allocation parser, zero-copy viewer & 64B writer
 *  - SIMD Byte-Shuffle filter (NEON / AVX2 / WASM SIMD128 / Scalar) & LZ4-class block codec
 *  - Out-of-core ping-pong arena double-buffered .npy streamer
 *  - Zero-copy DLPack v1.0 & Apache Arrow C Data Interface bridges
 */

#include "rcre_io.h"
#include "rcre_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdatomic.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#define RCRE_IO_HAVE_POSIX 1
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <pthread.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(__linux__)
#include <sys/syscall.h>
#ifndef __NR_io_uring_setup
#if defined(__x86_64__)
#define __NR_io_uring_setup 425
#define __NR_io_uring_enter 426
#define __NR_io_uring_register 427
#elif defined(__aarch64__)
#define __NR_io_uring_setup 425
#define __NR_io_uring_enter 426
#define __NR_io_uring_register 427
#endif
#endif
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

/* ========================================================================= */
/* Internal Checked Arithmetic, Endianness & Telemetry Helpers               */
/* ========================================================================= */

static inline bool rcre_io_mul_overflow(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return false;
    }
    if (a > SIZE_MAX / b) {
        return true;
    }
    *out = a * b;
    return false;
}

static inline bool rcre_io_add_overflow(size_t a, size_t b, size_t* out) {
    if (b > SIZE_MAX - a) {
        return true;
    }
    *out = a + b;
    return false;
}

static inline uint16_t rcre_io_read_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t rcre_io_read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static inline uint64_t rcre_io_read_u64_le(const uint8_t* p) {
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static inline void rcre_io_write_u16_le(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFU);
    p[1] = (uint8_t)((v >> 8) & 0xFFU);
}

static inline void rcre_io_write_u32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFU);
    p[1] = (uint8_t)((v >> 8) & 0xFFU);
    p[2] = (uint8_t)((v >> 16) & 0xFFU);
    p[3] = (uint8_t)((v >> 24) & 0xFFU);
}

static inline void rcre_io_write_u64_le(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        p[i] = (uint8_t)((v >> (i * 8)) & 0xFFULL);
    }
}

static inline void rcre_io_record_telemetry(bool is_write, uint64_t bytes) {
    RCRE_ProfContext* prof = rcre_prof_global();
    if (prof) {
        rcre_prof_io_record(prof, is_write, bytes);
    }
}

static inline bool rcre_io_is_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/* ========================================================================= */
/* 1. Virtual Memory Mapping & OS Page-Cache Control (IO-REQ-01)             */
/* ========================================================================= */

RCRE_NODISCARD int rcre_io_mmap_open(
    const char* filepath,
    RCRE_IOHint hint,
    RCRE_MappedFile* out_map
) {
    if (!filepath || !out_map) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_map, 0, sizeof(RCRE_MappedFile));
    out_map->os_handle  = -1;
    out_map->map_handle = -1;

#if defined(RCRE_IO_HAVE_POSIX)
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        return RCRE_ERR_IO;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        close(fd);
        return RCRE_ERR_IO;
    }

    size_t file_size = (size_t)st.st_size;

#if defined(__APPLE__) && defined(F_NOCACHE)
    if (hint == RCRE_IO_HINT_STREAMING) {
        (void)fcntl(fd, F_NOCACHE, 1);
    }
#endif

    void* addr = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return RCRE_ERR_IO;
    }

    out_map->data        = (const uint8_t*)addr;
    out_map->size        = file_size;
    out_map->os_handle   = (intptr_t)fd;
    out_map->map_handle  = -1;
    out_map->hint        = hint;
    out_map->is_mapped   = true;
    out_map->is_heap_buf = false;

    (void)rcre_io_madvise_range(out_map, 0, file_size, hint);
    rcre_io_record_telemetry(false, (uint64_t)file_size);
    return RCRE_OK;
#elif defined(_WIN32)
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (hint == RCRE_IO_HINT_SEQUENTIAL) {
        flags |= FILE_FLAG_SEQUENTIAL_SCAN;
    } else if (hint == RCRE_IO_HINT_RANDOM) {
        flags |= FILE_FLAG_RANDOM_ACCESS;
    }
    HANDLE hFile = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, flags, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        return RCRE_ERR_IO;
    }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(hFile, &fsz) || fsz.QuadPart <= 0) {
        CloseHandle(hFile);
        return RCRE_ERR_IO;
    }
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        CloseHandle(hFile);
        return RCRE_ERR_IO;
    }
    void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        return RCRE_ERR_IO;
    }
    out_map->data        = (const uint8_t*)view;
    out_map->size        = (size_t)fsz.QuadPart;
    out_map->os_handle   = (intptr_t)hFile;
    out_map->map_handle  = (intptr_t)hMap;
    out_map->hint        = hint;
    out_map->is_mapped   = true;
    out_map->is_heap_buf = false;
    rcre_io_record_telemetry(false, (uint64_t)out_map->size);
    return RCRE_OK;
#else
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return RCRE_ERR_IO;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return RCRE_ERR_IO;
    }
    long fsize = ftell(f);
    if (fsize <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return RCRE_ERR_IO;
    }
    size_t file_size = (size_t)fsize;
    size_t aligned_cap = (file_size + 63) & ~((size_t)63);
    uint8_t* buf = (uint8_t*)aligned_alloc(64, aligned_cap);
    if (!buf) {
        fclose(f);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    if (fread(buf, 1, file_size, f) != file_size) {
        free(buf);
        fclose(f);
        return RCRE_ERR_IO;
    }
    fclose(f);

    out_map->data        = buf;
    out_map->size        = file_size;
    out_map->os_handle   = -1;
    out_map->hint        = hint;
    out_map->is_mapped   = false;
    out_map->is_heap_buf = true;
    rcre_io_record_telemetry(false, (uint64_t)file_size);
    return RCRE_OK;
#endif
}

RCRE_NODISCARD int rcre_io_mmap_from_buffer(
    const void* buf,
    size_t size,
    RCRE_MappedFile* out_map
) {
    if (!buf || size == 0 || !out_map) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_map, 0, sizeof(RCRE_MappedFile));
    out_map->data        = (const uint8_t*)buf;
    out_map->size        = size;
    out_map->os_handle   = -1;
    out_map->hint        = RCRE_IO_HINT_DEFAULT;
    out_map->is_mapped   = false;
    out_map->is_heap_buf = false;
    return RCRE_OK;
}

void rcre_io_mmap_close(RCRE_MappedFile* map) {
    if (!map) {
        return;
    }
#if defined(RCRE_IO_HAVE_POSIX)
    if (map->is_mapped && map->data && map->size > 0) {
        (void)munmap((void*)(uintptr_t)map->data, map->size);
    }
    if (map->os_handle >= 0) {
        (void)close((int)map->os_handle);
    }
#elif defined(_WIN32)
    if (map->is_mapped && map->data) {
        (void)UnmapViewOfFile((LPCVOID)map->data);
    }
    if (map->map_handle != -1 && map->map_handle != 0) {
        (void)CloseHandle((HANDLE)map->map_handle);
    }
    if (map->os_handle != -1 && map->os_handle != 0) {
        (void)CloseHandle((HANDLE)map->os_handle);
    }
#endif
    if (map->is_heap_buf && map->data) {
        free((void*)(uintptr_t)map->data);
    }
    memset(map, 0, sizeof(RCRE_MappedFile));
    map->os_handle  = -1;
    map->map_handle = -1;
}

RCRE_NODISCARD RCRE_Slice rcre_io_mmap_slice(const RCRE_MappedFile* map) {
    if (!map || !map->data || map->size == 0) {
        return rcre_slice_empty();
    }
    return rcre_slice_from_parts(map->data, map->size);
}

RCRE_NODISCARD int rcre_io_madvise_range(
    const RCRE_MappedFile* map,
    size_t offset,
    size_t length,
    RCRE_IOHint hint
) {
    if (!map || !map->data || map->size == 0 || offset >= map->size) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (!map->is_mapped) {
        return RCRE_OK;
    }
    if (length == 0 || offset + length > map->size) {
        length = map->size - offset;
    }

#if defined(RCRE_IO_HAVE_POSIX) && defined(POSIX_MADV_NORMAL)
    long page_sz = sysconf(_SC_PAGESIZE);
    size_t page_mask = (page_sz > 0) ? ((size_t)page_sz - 1) : 4095;
    size_t aligned_off = offset & ~page_mask;
    size_t aligned_len = length + (offset - aligned_off);

    int advice = POSIX_MADV_NORMAL;
    switch (hint) {
        case RCRE_IO_HINT_SEQUENTIAL:
            advice = POSIX_MADV_SEQUENTIAL;
            break;
        case RCRE_IO_HINT_PREFETCH:
            advice = POSIX_MADV_SEQUENTIAL;
            (void)posix_madvise((void*)(uintptr_t)(map->data + aligned_off), aligned_len, POSIX_MADV_WILLNEED);
            break;
        case RCRE_IO_HINT_RANDOM:
            advice = POSIX_MADV_RANDOM;
            break;
        case RCRE_IO_HINT_STREAMING:
            advice = POSIX_MADV_DONTNEED;
            break;
        default:
            advice = POSIX_MADV_NORMAL;
            break;
    }
    (void)posix_madvise((void*)(uintptr_t)(map->data + aligned_off), aligned_len, advice);
#else
    (void)length;
    (void)hint;
#endif
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_write_atomic(
    const char* filepath,
    const void* data,
    size_t size
) {
    if (!filepath || (!data && size > 0)) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t path_len = strlen(filepath);
    if (path_len == 0 || path_len > 4000) {
        return RCRE_ERR_INVALID_ARG;
    }

    char tmp_path[4096];
#if defined(RCRE_IO_HAVE_POSIX)
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", filepath, (int)getpid());
#else
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", filepath);
#endif
    if (n <= 0 || (size_t)n >= sizeof(tmp_path)) {
        return RCRE_ERR_INVALID_ARG;
    }

    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        return RCRE_ERR_IO;
    }

    if (size > 0) {
        if (fwrite(data, 1, size, f) != size) {
            fclose(f);
            (void)remove(tmp_path);
            return RCRE_ERR_IO;
        }
    }

    if (fflush(f) != 0) {
        fclose(f);
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }

#if defined(RCRE_IO_HAVE_POSIX)
    (void)fsync(fileno(f));
#endif
    fclose(f);

#if defined(_WIN32)
    if (!MoveFileExA(tmp_path, filepath, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }
#else
    if (rename(tmp_path, filepath) != 0) {
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }
#endif

    rcre_io_record_telemetry(true, (uint64_t)size);
    return RCRE_OK;
}

/* ========================================================================= */
/* 2. Clinger / Eisel-Lemire Fast Float32 Slice Parser (IO-REQ-02)           */
/* ========================================================================= */

static const double rcre_io_pow10_exact[23] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
    1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};

static inline uint8_t rcre_io_ascii_lower(uint8_t c) {
    return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + ('a' - 'A')) : c;
}

static bool rcre_io_slice_eq_ci(RCRE_Slice s, const char* lower_lit, size_t lit_len) {
    if (s.len != lit_len || !s.data) {
        return false;
    }
    for (size_t i = 0; i < lit_len; ++i) {
        if (rcre_io_ascii_lower(s.data[i]) != (uint8_t)lower_lit[i]) {
            return false;
        }
    }
    return true;
}

RCRE_NODISCARD bool rcre_slice_parse_f32(RCRE_Slice s, float* out_val) {
    if (!out_val || !s.data || s.len == 0) {
        return false;
    }

    s = rcre_slice_trim_ascii(s);
    if (s.len == 0) {
        return false;
    }

    size_t idx = 0;
    bool neg = false;
    if (s.data[idx] == '-') {
        neg = true;
        idx++;
    } else if (s.data[idx] == '+') {
        idx++;
    }

    if (idx >= s.len) {
        return false;
    }

    /* Check special IEEE-754 tokens: nan, inf, infinity */
    uint8_t first_lower = rcre_io_ascii_lower(s.data[idx]);
    if (first_lower == 'n' || first_lower == 'i') {
        RCRE_Slice tail = rcre_slice_sub(s, idx, s.len);
        if (rcre_io_slice_eq_ci(tail, "nan", 3)) {
            *out_val = NAN;
            return true;
        }
        if (rcre_io_slice_eq_ci(tail, "inf", 3) || rcre_io_slice_eq_ci(tail, "infinity", 8)) {
            *out_val = neg ? -INFINITY : INFINITY;
            return true;
        }
        return false;
    }

    uint64_t mantissa = 0;
    int32_t exp10 = 0;
    size_t total_digits = 0;
    size_t sig_digits = 0;
    bool seen_dot = false;
    bool truncated_nonzero = false;

    while (idx < s.len) {
        uint8_t c = s.data[idx];
        if (c >= '0' && c <= '9') {
            total_digits++;
            uint8_t digit = (uint8_t)(c - '0');
            if (mantissa == 0 && digit == 0) {
                if (seen_dot) {
                    exp10--;
                }
            } else {
                if (sig_digits < 15) {
                    mantissa = mantissa * 10ULL + (uint64_t)digit;
                    sig_digits++;
                    if (seen_dot) {
                        exp10--;
                    }
                } else {
                    if (!seen_dot) {
                        exp10++;
                    }
                    if (digit != 0) {
                        truncated_nonzero = true;
                    }
                    sig_digits++;
                }
            }
            idx++;
        } else if (c == '.' && !seen_dot) {
            seen_dot = true;
            idx++;
        } else {
            break;
        }
    }

    if (total_digits == 0) {
        return false;
    }

    /* Parse optional scientific exponent (e/E[+-]digits) */
    if (idx < s.len && (s.data[idx] == 'e' || s.data[idx] == 'E')) {
        idx++;
        bool exp_neg = false;
        if (idx < s.len && (s.data[idx] == '-' || s.data[idx] == '+')) {
            exp_neg = (s.data[idx] == '-');
            idx++;
        }
        if (idx >= s.len) {
            return false;
        }
        int32_t e_val = 0;
        size_t e_digits = 0;
        while (idx < s.len && s.data[idx] >= '0' && s.data[idx] <= '9') {
            if (e_val < 100000) {
                e_val = e_val * 10 + (int32_t)(s.data[idx] - '0');
            }
            e_digits++;
            idx++;
        }
        if (e_digits == 0) {
            return false;
        }
        exp10 += exp_neg ? -e_val : e_val;
    }

    /* Reject trailing non-numeric characters */
    if (idx != s.len) {
        return false;
    }

    if (mantissa == 0) {
        *out_val = neg ? -0.0f : 0.0f;
        return true;
    }

    /* Clinger / Eisel-Lemire Fast Exact Path:
     * In binary64 (53-bit significand), any 15-digit decimal significand
     * (< 10^15 < 2^53) and 10^|k| for |k| <= 22 are 100% exact.
     * Single binary64 mul/div has at most 0.5 ULP_f64 error; if the lower
     * 29 bits of the binary64 result do not lie on the float32 midpoint
     * (0x10000000), casting to float32 is provably 0-ULP exact. */
    if (!truncated_nonzero && sig_digits <= 15 && exp10 >= -22 && exp10 <= 22) {
        double d = (double)mantissa;
        if (exp10 > 0) {
            d *= rcre_io_pow10_exact[exp10];
        } else if (exp10 < 0) {
            d /= rcre_io_pow10_exact[-exp10];
        }
        uint64_t dbits = 0;
        memcpy(&dbits, &d, sizeof(uint64_t));
        /* Verify result is not within 1 f64 ULP of a float32 halfway tie */
        if (((dbits + 1ULL) & 0x1FFFFFFEULL) != 0x10000000ULL) {
            float res = (float)d;
            *out_val = neg ? -res : res;
            return true;
        }
    }

    /* Exact fallback for extreme exponents (|exp10| > 22), >15 significant
     * digits, or near-midpoint ties. */
    char stack_buf[128];
    if (s.len < sizeof(stack_buf)) {
        memcpy(stack_buf, s.data, s.len);
        stack_buf[s.len] = '\0';
        char* endp = nullptr;
        float v = strtof(stack_buf, &endp);
        if (endp == stack_buf + s.len) {
            *out_val = v;
            return true;
        }
    }

    /* Ultra-long token fallback (>127 chars of leading/trailing zeros) */
    double d_approx = (double)mantissa * pow(10.0, (double)exp10);
    float f_approx = (float)d_approx;
    *out_val = neg ? -f_approx : f_approx;
    return true;
}

/* ========================================================================= */
/* 3. SIMD-Accelerated & Threadpool-Parallel CSV/TSV Loader (IO-REQ-02)      */
/* ========================================================================= */

static inline bool rcre_io_is_csv_delim(uint8_t c, uint8_t delim) {
    if (delim == 0) {
        return c == ',' || c == '\t';
    }
    return c == delim;
}

/* Extracts the next non-EOF line from cursor, stripping trailing '\r'. */
static bool rcre_io_csv_next_line(RCRE_Slice* cursor, RCRE_Slice* out_line) {
    if (!cursor || !cursor->data || cursor->len == 0) {
        return false;
    }

    size_t nl = rcre_slice_find_byte(*cursor, '\n');
    RCRE_Slice line;
    if (nl == RCRE_NPOS) {
        line = *cursor;
        *cursor = rcre_slice_empty();
    } else {
        line = rcre_slice_sub(*cursor, 0, nl);
        *cursor = rcre_slice_sub(*cursor, nl + 1, cursor->len);
    }

    if (line.len > 0 && line.data[line.len - 1] == '\r') {
        line.len--;
    }

    *out_line = line;
    return true;
}

/* Counts fields in a CSV/TSV row using consecutive-delimiter skipping
 * matching strtok_r semantics while executing in zero-copy read-only memory.
 * When quote_char != 0, delimiters inside paired quotes are treated as field content. */
static size_t rcre_io_csv_count_cols(RCRE_Slice line, uint8_t delim, uint8_t quote_char) {
    size_t cols = 0;
    size_t i = 0;
    while (i < line.len) {
        while (i < line.len && rcre_io_is_csv_delim(line.data[i], delim)) {
            i++;
        }
        if (i >= line.len) {
            break;
        }
        cols++;
        if (quote_char != 0 && line.data[i] == quote_char) {
            i++;
            while (i < line.len) {
                if (line.data[i] == quote_char) {
                    if (i + 1 < line.len && line.data[i + 1] == quote_char) {
                        i += 2; /* RFC-4180 escaped quote "" */
                        continue;
                    }
                    i++;
                    break;
                }
                i++;
            }
            while (i < line.len && !rcre_io_is_csv_delim(line.data[i], delim)) {
                i++;
            }
        } else {
            while (i < line.len && !rcre_io_is_csv_delim(line.data[i], delim)) {
                i++;
            }
        }
    }
    return cols;
}

/* Parses a single CSV/TSV row into dst_row[0..expected_cols-1]. */
static bool rcre_io_csv_parse_row(
    RCRE_Slice line,
    uint8_t delim,
    uint8_t quote_char,
    bool strict_numbers,
    size_t expected_cols,
    float* dst_row
) {
    size_t c = 0;
    size_t i = 0;

    while (i < line.len) {
        while (i < line.len && rcre_io_is_csv_delim(line.data[i], delim)) {
            i++;
        }
        if (i >= line.len) {
            break;
        }
        size_t tok_start = i;
        size_t tok_end = i;
        if (quote_char != 0 && line.data[i] == quote_char) {
            i++;
            tok_start = i;
            bool closed = false;
            while (i < line.len) {
                if (line.data[i] == quote_char) {
                    if (i + 1 < line.len && line.data[i + 1] == quote_char) {
                        i += 2;
                        continue;
                    }
                    tok_end = i;
                    i++;
                    closed = true;
                    break;
                }
                i++;
            }
            if (!closed) {
                tok_end = i;
            }
            while (i < line.len && !rcre_io_is_csv_delim(line.data[i], delim)) {
                i++;
            }
        } else {
            if (delim != 0) {
                size_t rel = rcre_hal.find_byte(line.data + i, line.len - i, delim);
                i = (rel == RCRE_NPOS) ? line.len : (i + rel);
            } else {
                while (i < line.len && line.data[i] != ',' && line.data[i] != '\t') {
                    i++;
                }
            }
            tok_end = i;
        }

        if (c >= expected_cols) {
            return false; /* Too many columns (ragged row) */
        }

        RCRE_Slice tok = { .data = line.data + tok_start, .len = tok_end - tok_start };
        float val = 0.0f;
        if (!rcre_slice_parse_f32(tok, &val)) {
            if (strict_numbers) {
                return false;
            }
            val = 0.0f;
        }
        dst_row[c++] = val;
    }

    return c == expected_cols;
}

typedef struct {
    const RCRE_Slice* lines;
    size_t            total_rows;
    size_t            cols;
    size_t            rows_per_task;
    uint8_t           delim;
    uint8_t           quote_char;
    bool              strict_numbers;
    float*            dst_data;
    atomic_int        failed;
} RCRE_IOCSVParallelCtx;

static void rcre_io_csv_parallel_worker(void* arg, size_t task_id) {
    RCRE_IOCSVParallelCtx* ctx = (RCRE_IOCSVParallelCtx*)arg;
    if (atomic_load_explicit(&ctx->failed, memory_order_relaxed) != 0) {
        return;
    }

    size_t r_start = task_id * ctx->rows_per_task;
    if (r_start >= ctx->total_rows) {
        return;
    }
    size_t r_end = r_start + ctx->rows_per_task;
    if (r_end > ctx->total_rows) {
        r_end = ctx->total_rows;
    }

    for (size_t r = r_start; r < r_end; ++r) {
        if (!rcre_io_csv_parse_row(
                ctx->lines[r],
                ctx->delim,
                ctx->quote_char,
                ctx->strict_numbers,
                ctx->cols,
                ctx->dst_data + r * ctx->cols)) {
            atomic_store_explicit(&ctx->failed, 1, memory_order_relaxed);
            return;
        }
    }
}

RCRE_NODISCARD int rcre_io_parse_csv_slice(
    RCRE_Slice csv_slice,
    const RCRE_CSVOptions* opts,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
) {
    if (!arena || !out_tensor || !csv_slice.data || csv_slice.len == 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_tensor, 0, sizeof(RCRE_Tensor));

    RCRE_CSVOptions default_opts = {
        .delimiter      = 0,
        .skip_header    = false,
        .strict_numbers = false,
        .max_rows       = 0,
        .pool           = nullptr,
        .quote_char     = 0
    };
    const RCRE_CSVOptions* cfg = opts ? opts : &default_opts;

    RCRE_Slice cursor = csv_slice;
    RCRE_Slice line = rcre_slice_empty();

    if (cfg->skip_header) {
        if (!rcre_io_csv_next_line(&cursor, &line)) {
            return RCRE_ERR_IO;
        }
    }

    RCRE_Slice data_start = cursor;
    if (!rcre_io_csv_next_line(&cursor, &line)) {
        return RCRE_ERR_IO;
    }

    size_t cols = rcre_io_csv_count_cols(line, cfg->delimiter, cfg->quote_char);
    if (cols == 0) {
        return RCRE_ERR_IO;
    }

    size_t rows = 1;
    if (cfg->max_rows == 0 || rows < cfg->max_rows) {
        while (rcre_io_csv_next_line(&cursor, &line)) {
            if (line.len == 0 && cursor.len == 0) {
                break;
            }
            rows++;
            if (cfg->max_rows > 0 && rows >= cfg->max_rows) {
                break;
            }
        }
    }

    size_t shape[2] = { rows, cols };
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!t.data) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    float* td = rcre_tensor_f32_data(&t);

    /* Fast Parallel Path if ThreadPool provided and batch is nontrivial */
    if (cfg->pool && rows >= 64) {
        RCRE_ArenaTemp temp = rcre_arena_temp_begin(arena);
        RCRE_Slice* line_index = (RCRE_Slice*)rcre_arena_alloc(arena, rows * sizeof(RCRE_Slice), alignof(RCRE_Slice));
        if (line_index) {
            RCRE_Slice p_cur = data_start;
            size_t indexed = 0;
            while (indexed < rows && rcre_io_csv_next_line(&p_cur, &line_index[indexed])) {
                indexed++;
            }
            if (indexed == rows) {
                size_t num_tasks = 16;
                size_t rows_per_task = (rows + num_tasks - 1) / num_tasks;
                RCRE_IOCSVParallelCtx pctx = {
                    .lines          = line_index,
                    .total_rows     = rows,
                    .cols           = cols,
                    .rows_per_task  = rows_per_task,
                    .delim          = cfg->delimiter,
                    .quote_char     = cfg->quote_char,
                    .strict_numbers = cfg->strict_numbers,
                    .dst_data       = td,
                    .failed         = 0
                };
                rcre_threadpool_parallel_for(cfg->pool, num_tasks, rcre_io_csv_parallel_worker, &pctx);
                rcre_arena_temp_end(temp);

                if (atomic_load(&pctx.failed) != 0) {
                    return RCRE_ERR_IO;
                }
                *out_tensor = t;
                return RCRE_OK;
            }
        }
        rcre_arena_temp_end(temp);
    }

    /* Single-Threaded Zero-Allocation Stream Parse */
    RCRE_Slice s_cur = data_start;
    for (size_t r = 0; r < rows; ++r) {
        if (!rcre_io_csv_next_line(&s_cur, &line)) {
            return RCRE_ERR_IO;
        }
        if (!rcre_io_csv_parse_row(line, cfg->delimiter, cfg->quote_char, cfg->strict_numbers, cols, td + r * cols)) {
            return RCRE_ERR_IO;
        }
    }

    *out_tensor = t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_load_csv_ex(
    const char* filepath,
    const RCRE_CSVOptions* opts,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
) {
    if (!filepath || !arena || !out_tensor) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_MappedFile map;
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) {
        return rc;
    }

    rc = rcre_io_parse_csv_slice(rcre_io_mmap_slice(&map), opts, arena, out_tensor);
    rcre_io_mmap_close(&map);
    return rc;
}

/* ========================================================================= */
/* 4. NumPy (.npy v1.0 / v2.0) Serialization & Zero-Copy Views (IO-REQ-03)   */
/* ========================================================================= */

static const uint8_t rcre_io_npy_magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };

static const char* rcre_io_dtype_to_npy_descr(RCRE_DType dtype) {
    switch (dtype) {
        case RCRE_DTYPE_U8:   return "|u1";
        case RCRE_DTYPE_I8:   return "|i1";
        case RCRE_DTYPE_F16:  return "<f2";
        case RCRE_DTYPE_BF16: return "<b2";
        case RCRE_DTYPE_I32:  return "<i4";
        case RCRE_DTYPE_I64:  return "<i8";
        case RCRE_DTYPE_F32:  return "<f4";
        case RCRE_DTYPE_F64:  return "<f8";
        default:              return nullptr;
    }
}

static bool rcre_io_npy_descr_to_dtype(RCRE_Slice descr, RCRE_DType* out_dtype) {
    if (rcre_slice_eq_cstr(descr, "|u1") || rcre_slice_eq_cstr(descr, "<u1") || rcre_slice_eq_cstr(descr, "u1")) {
        *out_dtype = RCRE_DTYPE_U8;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "|i1") || rcre_slice_eq_cstr(descr, "<i1") || rcre_slice_eq_cstr(descr, "i1")) {
        *out_dtype = RCRE_DTYPE_I8;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "<f2") || rcre_slice_eq_cstr(descr, "=f2") || rcre_slice_eq_cstr(descr, "f2") || rcre_slice_eq_cstr(descr, "|f2")) {
        *out_dtype = RCRE_DTYPE_F16;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "<b2") || rcre_slice_eq_cstr(descr, "=b2") || rcre_slice_eq_cstr(descr, "b2") || rcre_slice_eq_cstr(descr, "|b2") ||
        rcre_slice_eq_cstr(descr, "<V2") || rcre_slice_eq_cstr(descr, "|V2")) {
        *out_dtype = RCRE_DTYPE_BF16;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "<i4") || rcre_slice_eq_cstr(descr, "=i4") || rcre_slice_eq_cstr(descr, "i4")) {
        *out_dtype = RCRE_DTYPE_I32;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "<i8") || rcre_slice_eq_cstr(descr, "=i8") || rcre_slice_eq_cstr(descr, "i8")) {
        *out_dtype = RCRE_DTYPE_I64;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "<f4") || rcre_slice_eq_cstr(descr, "=f4") || rcre_slice_eq_cstr(descr, "f4")) {
        *out_dtype = RCRE_DTYPE_F32;
        return true;
    }
    if (rcre_slice_eq_cstr(descr, "<f8") || rcre_slice_eq_cstr(descr, "=f8") || rcre_slice_eq_cstr(descr, "f8")) {
        *out_dtype = RCRE_DTYPE_F64;
        return true;
    }
    return false;
}

RCRE_NODISCARD int rcre_io_save_npy(
    const RCRE_Tensor* tensor,
    const char* filepath
) {
    if (!tensor || !tensor->data || !filepath || tensor->ndim == 0 || tensor->ndim > RCRE_MAX_DIMS) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (!rcre_tensor_is_contiguous(tensor)) {
        return RCRE_ERR_INVALID_ARG;
    }

    const char* descr = rcre_io_dtype_to_npy_descr(tensor->dtype);
    if (!descr) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    char dict_buf[256];
    int pos = snprintf(dict_buf, sizeof(dict_buf),
                       "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
    if (pos <= 0 || (size_t)pos >= sizeof(dict_buf)) {
        return RCRE_ERR_IO;
    }

    for (uint8_t i = 0; i < tensor->ndim; ++i) {
        int n = snprintf(dict_buf + pos, sizeof(dict_buf) - (size_t)pos,
                         (i == 0) ? "%zu" : ", %zu", tensor->shape[i]);
        if (n <= 0 || (size_t)(pos + n) >= sizeof(dict_buf)) {
            return RCRE_ERR_IO;
        }
        pos += n;
    }

    int tail_n = snprintf(dict_buf + pos, sizeof(dict_buf) - (size_t)pos,
                          (tensor->ndim == 1) ? ",), }" : "), }");
    if (tail_n <= 0 || (size_t)(pos + tail_n) >= sizeof(dict_buf)) {
        return RCRE_ERR_IO;
    }
    size_t dict_len = (size_t)(pos + tail_n);

    /* Pad v1.0 preamble (10 bytes) + dict_len + padding + '\n' to an exact
     * multiple of 64 bytes so the array payload starts at offset 64 * k. */
    size_t raw_total = 10 + dict_len + 1;
    size_t aligned_total = (raw_total + 63) & ~((size_t)63);
    size_t header_len = aligned_total - 10;

    uint8_t header_block[512];
    if (aligned_total > sizeof(header_block)) {
        return RCRE_ERR_IO;
    }

    memcpy(header_block, rcre_io_npy_magic, 6);
    header_block[6] = 1; /* major v1 */
    header_block[7] = 0; /* minor v0 */
    rcre_io_write_u16_le(&header_block[8], (uint16_t)header_len);
    memcpy(&header_block[10], dict_buf, dict_len);

    size_t pad_bytes = header_len - dict_len - 1;
    memset(&header_block[10 + dict_len], ' ', pad_bytes);
    header_block[aligned_total - 1] = '\n';

    size_t payload_bytes = rcre_tensor_nbytes(tensor);

    char tmp_path[4096];
#if defined(RCRE_IO_HAVE_POSIX)
    int tn = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", filepath, (int)getpid());
#else
    int tn = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", filepath);
#endif
    if (tn <= 0 || (size_t)tn >= sizeof(tmp_path)) {
        return RCRE_ERR_INVALID_ARG;
    }

    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        return RCRE_ERR_IO;
    }

    if (fwrite(header_block, 1, aligned_total, f) != aligned_total ||
        (payload_bytes > 0 && fwrite(tensor->data, 1, payload_bytes, f) != payload_bytes)) {
        fclose(f);
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }

    if (fflush(f) != 0) {
        fclose(f);
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }
#if defined(RCRE_IO_HAVE_POSIX)
    (void)fsync(fileno(f));
#endif
    fclose(f);

#if defined(_WIN32)
    if (!MoveFileExA(tmp_path, filepath, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }
#else
    if (rename(tmp_path, filepath) != 0) {
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }
#endif

    rcre_io_record_telemetry(true, (uint64_t)(aligned_total + payload_bytes));
    return RCRE_OK;
}

/* Internal zero-allocation .npy header parser */
static int rcre_io_parse_npy_header(
    RCRE_Slice file_slice,
    RCRE_DType* out_dtype,
    uint8_t* out_ndim,
    size_t out_shape[RCRE_MAX_DIMS],
    size_t* out_data_offset,
    size_t* out_data_nbytes
) {
    if (!file_slice.data || file_slice.len < 10) {
        return RCRE_ERR_IO;
    }
    if (memcmp(file_slice.data, rcre_io_npy_magic, 6) != 0) {
        return RCRE_ERR_IO;
    }

    uint8_t major = file_slice.data[6];
    uint8_t minor = file_slice.data[7];
    (void)minor;

    size_t preamble_len = 0;
    size_t header_len = 0;
    if (major == 1) {
        preamble_len = 10;
        header_len = (size_t)rcre_io_read_u16_le(&file_slice.data[8]);
    } else if (major == 2 || major == 3) {
        if (file_slice.len < 12) {
            return RCRE_ERR_IO;
        }
        preamble_len = 12;
        header_len = (size_t)rcre_io_read_u32_le(&file_slice.data[8]);
    } else {
        return RCRE_ERR_IO;
    }

    size_t data_offset = 0;
    if (header_len == 0 || rcre_io_add_overflow(preamble_len, header_len, &data_offset) ||
        data_offset > file_slice.len) {
        return RCRE_ERR_IO;
    }

    RCRE_Slice hdr = rcre_slice_sub(file_slice, preamble_len, data_offset);

    /* 1. Extract 'descr' */
    RCRE_SlicePair p_descr = rcre_slice_split_once(hdr, rcre_slice_from_cstr("'descr'"));
    if (!p_descr.found) {
        p_descr = rcre_slice_split_once(hdr, rcre_slice_from_cstr("\"descr\""));
        if (!p_descr.found) return RCRE_ERR_IO;
    }
    RCRE_SlicePair p_colon = rcre_slice_split_once_byte(p_descr.tail, ':');
    if (!p_colon.found) return RCRE_ERR_IO;
    RCRE_Slice descr_rest = rcre_slice_trim_ascii(p_colon.tail);
    if (descr_rest.len < 3 || (descr_rest.data[0] != '\'' && descr_rest.data[0] != '"')) {
        return RCRE_ERR_IO;
    }
    uint8_t quote_ch = descr_rest.data[0];
    RCRE_Slice descr_body = rcre_slice_sub(descr_rest, 1, descr_rest.len);
    size_t q_end = rcre_slice_find_byte(descr_body, quote_ch);
    if (q_end == RCRE_NPOS) return RCRE_ERR_IO;
    RCRE_Slice descr_val = rcre_slice_sub(descr_body, 0, q_end);
    if (!rcre_io_npy_descr_to_dtype(descr_val, out_dtype)) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    /* 2. Verify 'fortran_order': False */
    RCRE_SlicePair p_fort = rcre_slice_split_once(hdr, rcre_slice_from_cstr("'fortran_order'"));
    if (!p_fort.found) {
        p_fort = rcre_slice_split_once(hdr, rcre_slice_from_cstr("\"fortran_order\""));
        if (!p_fort.found) return RCRE_ERR_IO;
    }
    RCRE_SlicePair p_fcolon = rcre_slice_split_once_byte(p_fort.tail, ':');
    if (!p_fcolon.found) return RCRE_ERR_IO;
    RCRE_Slice fort_val = rcre_slice_trim_ascii(p_fcolon.tail);
    if (!rcre_slice_starts_with(fort_val, rcre_slice_from_cstr("False"))) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    /* 3. Extract 'shape': (d0, d1, ...) */
    RCRE_SlicePair p_shape = rcre_slice_split_once(hdr, rcre_slice_from_cstr("'shape'"));
    if (!p_shape.found) {
        p_shape = rcre_slice_split_once(hdr, rcre_slice_from_cstr("\"shape\""));
        if (!p_shape.found) return RCRE_ERR_IO;
    }
    size_t lparen = rcre_slice_find_byte(p_shape.tail, '(');
    size_t rparen = rcre_slice_find_byte(p_shape.tail, ')');
    if (lparen == RCRE_NPOS || rparen == RCRE_NPOS || lparen >= rparen) {
        return RCRE_ERR_IO;
    }

    RCRE_Slice shape_tuple = rcre_slice_sub(p_shape.tail, lparen + 1, rparen);
    uint8_t ndim = 0;
    size_t numel = 1;
    memset(out_shape, 0, RCRE_MAX_DIMS * sizeof(size_t));

    while (shape_tuple.len > 0) {
        RCRE_SlicePair sp = rcre_slice_split_once_byte(shape_tuple, ',');
        RCRE_Slice item = rcre_slice_trim_ascii(sp.head);
        shape_tuple = sp.tail;
        if (item.len == 0) {
            continue;
        }
        if (ndim >= RCRE_MAX_DIMS) {
            return RCRE_ERR_UNSUPPORTED_OP;
        }
        uint64_t dim_u64 = 0;
        if (!rcre_slice_parse_u64(item, &dim_u64) || dim_u64 == 0 || dim_u64 > (uint64_t)SIZE_MAX) {
            return RCRE_ERR_IO;
        }
        size_t dim = (size_t)dim_u64;
        if (rcre_io_mul_overflow(numel, dim, &numel)) {
            return RCRE_ERR_IO;
        }
        out_shape[ndim++] = dim;
    }

    if (ndim == 0) {
        return RCRE_ERR_IO;
    }

    size_t elem_size = rcre_dtype_size(*out_dtype);
    size_t data_nbytes = 0;
    if (rcre_io_mul_overflow(numel, elem_size, &data_nbytes)) {
        return RCRE_ERR_IO;
    }

    size_t expected_file_size = 0;
    if (rcre_io_add_overflow(data_offset, data_nbytes, &expected_file_size) ||
        expected_file_size != file_slice.len) {
        return RCRE_ERR_IO;
    }

    *out_ndim = ndim;
    *out_data_offset = data_offset;
    *out_data_nbytes = data_nbytes;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_view_npy(
    const RCRE_MappedFile* map,
    RCRE_Tensor* out_view
) {
    if (!map || !map->data || map->size == 0 || !out_view) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_view, 0, sizeof(RCRE_Tensor));

    RCRE_DType dtype = RCRE_DTYPE_F32;
    uint8_t ndim = 0;
    size_t shape[RCRE_MAX_DIMS] = {0};
    size_t data_offset = 0;
    size_t data_nbytes = 0;

    int rc = rcre_io_parse_npy_header(rcre_io_mmap_slice(map), &dtype, &ndim, shape, &data_offset, &data_nbytes);
    if (rc != RCRE_OK) {
        return rc;
    }

    const uint8_t* payload_ptr = map->data + data_offset;
    /* Enforce 64-byte SIMD alignment invariant on zero-copy views */
    if (((uintptr_t)payload_ptr & 63U) != 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_Layout layout = (ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;
    *out_view = rcre_tensor_from_buffer((void*)(uintptr_t)payload_ptr, shape, ndim, dtype, layout);
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_load_npy(
    const char* filepath,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
) {
    if (!filepath || !arena || !out_tensor) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_tensor, 0, sizeof(RCRE_Tensor));

    RCRE_MappedFile map;
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) {
        return rc;
    }

    RCRE_DType dtype = RCRE_DTYPE_F32;
    uint8_t ndim = 0;
    size_t shape[RCRE_MAX_DIMS] = {0};
    size_t data_offset = 0;
    size_t data_nbytes = 0;

    rc = rcre_io_parse_npy_header(rcre_io_mmap_slice(&map), &dtype, &ndim, shape, &data_offset, &data_nbytes);
    if (rc != RCRE_OK) {
        rcre_io_mmap_close(&map);
        return rc;
    }

    RCRE_Layout layout = (ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;
    RCRE_Tensor t = rcre_tensor_create(shape, ndim, dtype, layout, arena);
    if (!t.data) {
        rcre_io_mmap_close(&map);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    memcpy(t.data, map.data + data_offset, data_nbytes);
    rcre_io_mmap_close(&map);
    *out_tensor = t;
    return RCRE_OK;
}

/* ========================================================================= */
/* 5. Multi-Tensor Safetensors (.safetensors) Container Engine (IO-REQ-04)   */
/* ========================================================================= */

static const char* rcre_io_dtype_to_st_str(RCRE_DType dtype) {
    switch (dtype) {
        case RCRE_DTYPE_U8:   return "U8";
        case RCRE_DTYPE_I8:   return "I8";
        case RCRE_DTYPE_F16:  return "F16";
        case RCRE_DTYPE_BF16: return "BF16";
        case RCRE_DTYPE_I32:  return "I32";
        case RCRE_DTYPE_I64:  return "I64";
        case RCRE_DTYPE_F32:  return "F32";
        case RCRE_DTYPE_F64:  return "F64";
        default:              return nullptr;
    }
}

static bool rcre_io_st_str_to_dtype(RCRE_Slice s, RCRE_DType* out_dtype) {
    if (rcre_slice_eq_cstr(s, "F32")) {
        *out_dtype = RCRE_DTYPE_F32;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "F64")) {
        *out_dtype = RCRE_DTYPE_F64;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "F16")) {
        *out_dtype = RCRE_DTYPE_F16;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "BF16")) {
        *out_dtype = RCRE_DTYPE_BF16;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "I32")) {
        *out_dtype = RCRE_DTYPE_I32;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "I64")) {
        *out_dtype = RCRE_DTYPE_I64;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "I8")) {
        *out_dtype = RCRE_DTYPE_I8;
        return true;
    }
    if (rcre_slice_eq_cstr(s, "U8")) {
        *out_dtype = RCRE_DTYPE_U8;
        return true;
    }
    return false;
}

static void rcre_io_json_skip_ws(RCRE_Slice s, size_t* pos) {
    while (*pos < s.len && rcre_io_is_space(s.data[*pos])) {
        (*pos)++;
    }
}

static bool rcre_io_json_parse_string(RCRE_Slice s, size_t* pos, RCRE_Slice* out_str) {
    rcre_io_json_skip_ws(s, pos);
    if (*pos >= s.len || s.data[*pos] != '"') {
        return false;
    }
    (*pos)++;
    size_t start = *pos;
    while (*pos < s.len) {
        uint8_t c = s.data[*pos];
        if (c == '\\') {
            *pos += 2;
            if (*pos > s.len) return false;
            continue;
        }
        if (c == '"') {
            *out_str = rcre_slice_sub(s, start, *pos);
            (*pos)++;
            return true;
        }
        (*pos)++;
    }
    return false;
}

static bool rcre_io_json_skip_value(RCRE_Slice s, size_t* pos) {
    rcre_io_json_skip_ws(s, pos);
    if (*pos >= s.len) return false;
    uint8_t c = s.data[*pos];
    if (c == '"') {
        RCRE_Slice dummy;
        return rcre_io_json_parse_string(s, pos, &dummy);
    }
    if (c == '{' || c == '[') {
        uint8_t open_ch = c;
        uint8_t close_ch = (c == '{') ? '}' : ']';
        int depth = 1;
        (*pos)++;
        while (*pos < s.len && depth > 0) {
            if (s.data[*pos] == '"') {
                RCRE_Slice dummy;
                if (!rcre_io_json_parse_string(s, pos, &dummy)) return false;
                continue;
            }
            if (s.data[*pos] == open_ch) depth++;
            else if (s.data[*pos] == close_ch) depth--;
            (*pos)++;
        }
        return depth == 0;
    }
    while (*pos < s.len && s.data[*pos] != ',' && s.data[*pos] != '}' &&
           s.data[*pos] != ']' && !rcre_io_is_space(s.data[*pos])) {
        (*pos)++;
    }
    return true;
}

static bool rcre_io_json_parse_u64_array(
    RCRE_Slice s,
    size_t* pos,
    size_t* out_vals,
    size_t max_count,
    size_t* out_count
) {
    rcre_io_json_skip_ws(s, pos);
    if (*pos >= s.len || s.data[*pos] != '[') {
        return false;
    }
    (*pos)++;
    size_t count = 0;

    for (;;) {
        rcre_io_json_skip_ws(s, pos);
        if (*pos >= s.len) return false;
        if (s.data[*pos] == ']') {
            (*pos)++;
            *out_count = count;
            return true;
        }
        if (count > 0) {
            if (s.data[*pos] != ',') return false;
            (*pos)++;
            rcre_io_json_skip_ws(s, pos);
        }
        size_t num_start = *pos;
        while (*pos < s.len && s.data[*pos] >= '0' && s.data[*pos] <= '9') {
            (*pos)++;
        }
        if (*pos == num_start || count >= max_count) {
            return false;
        }
        uint64_t val = 0;
        if (!rcre_slice_parse_u64(rcre_slice_sub(s, num_start, *pos), &val) ||
            val > (uint64_t)SIZE_MAX) {
            return false;
        }
        out_vals[count++] = (size_t)val;
    }
}

static int rcre_io_safetensors_parse_archive(RCRE_SafeTensorsArchive* arch) {
    if (!arch || !arch->map.data || arch->map.size < 8) {
        return RCRE_ERR_IO;
    }

    uint64_t hdr_len_u64 = rcre_io_read_u64_le(arch->map.data);
    /* Reject empty or excessively large (> 100 MiB) headers */
    if (hdr_len_u64 == 0 || hdr_len_u64 > (100ULL * 1024ULL * 1024ULL)) {
        return RCRE_ERR_IO;
    }
    size_t hdr_len = (size_t)hdr_len_u64;
    size_t data_base = 0;
    if (rcre_io_add_overflow(8, hdr_len, &data_base) || data_base > arch->map.size) {
        return RCRE_ERR_IO;
    }

    size_t payload_limit = arch->map.size - data_base;
    arch->header_nbytes    = hdr_len;
    arch->data_base_offset = data_base;
    arch->num_tensors      = 0;

    RCRE_Slice json = rcre_slice_from_parts(arch->map.data + 8, hdr_len);
    size_t pos = 0;
    rcre_io_json_skip_ws(json, &pos);
    if (pos >= json.len || json.data[pos] != '{') {
        return RCRE_ERR_IO;
    }
    pos++;

    for (;;) {
        rcre_io_json_skip_ws(json, &pos);
        if (pos >= json.len) return RCRE_ERR_IO;
        if (json.data[pos] == '}') {
            pos++;
            break;
        }
        if (arch->num_tensors > 0 || pos > 1) {
            if (json.data[pos] == ',') {
                pos++;
                rcre_io_json_skip_ws(json, &pos);
            }
        }

        RCRE_Slice key_name;
        if (!rcre_io_json_parse_string(json, &pos, &key_name)) {
            return RCRE_ERR_IO;
        }
        rcre_io_json_skip_ws(json, &pos);
        if (pos >= json.len || json.data[pos] != ':') {
            return RCRE_ERR_IO;
        }
        pos++;

        if (rcre_slice_eq_cstr(key_name, "__metadata__")) {
            if (!rcre_io_json_skip_value(json, &pos)) {
                return RCRE_ERR_IO;
            }
            continue;
        }

        if (arch->num_tensors >= RCRE_SAFETENSORS_MAX_TENSORS || key_name.len == 0) {
            return RCRE_ERR_IO;
        }

        rcre_io_json_skip_ws(json, &pos);
        if (pos >= json.len || json.data[pos] != '{') {
            return RCRE_ERR_IO;
        }
        pos++;

        bool has_dtype = false;
        bool has_shape = false;
        bool has_offsets = false;
        RCRE_SafeTensorEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.name = key_name;

        for (;;) {
            rcre_io_json_skip_ws(json, &pos);
            if (pos >= json.len) return RCRE_ERR_IO;
            if (json.data[pos] == '}') {
                pos++;
                break;
            }
            if (json.data[pos] == ',') {
                pos++;
                rcre_io_json_skip_ws(json, &pos);
            }

            RCRE_Slice field_name;
            if (!rcre_io_json_parse_string(json, &pos, &field_name)) {
                return RCRE_ERR_IO;
            }
            rcre_io_json_skip_ws(json, &pos);
            if (pos >= json.len || json.data[pos] != ':') {
                return RCRE_ERR_IO;
            }
            pos++;

            if (rcre_slice_eq_cstr(field_name, "dtype")) {
                RCRE_Slice dtype_str;
                if (!rcre_io_json_parse_string(json, &pos, &dtype_str) ||
                    !rcre_io_st_str_to_dtype(dtype_str, &entry.dtype)) {
                    return RCRE_ERR_IO;
                }
                has_dtype = true;
            } else if (rcre_slice_eq_cstr(field_name, "shape")) {
                size_t dim_count = 0;
                if (!rcre_io_json_parse_u64_array(json, &pos, entry.shape, RCRE_MAX_DIMS, &dim_count) ||
                    dim_count == 0 || dim_count > RCRE_MAX_DIMS) {
                    return RCRE_ERR_IO;
                }
                entry.ndim = (uint8_t)dim_count;
                has_shape = true;
            } else if (rcre_slice_eq_cstr(field_name, "data_offsets")) {
                size_t offs[2] = {0, 0};
                size_t off_count = 0;
                if (!rcre_io_json_parse_u64_array(json, &pos, offs, 2, &off_count) ||
                    off_count != 2 || offs[0] > offs[1] || offs[1] > payload_limit) {
                    return RCRE_ERR_IO;
                }
                entry.data_offset = offs[0];
                entry.data_nbytes = offs[1] - offs[0];
                has_offsets = true;
            } else {
                if (!rcre_io_json_skip_value(json, &pos)) {
                    return RCRE_ERR_IO;
                }
            }
        }

        if (!has_dtype || !has_shape || !has_offsets) {
            return RCRE_ERR_IO;
        }

        /* Verify exact byte size matches prod(shape) * sizeof(dtype) */
        size_t numel = 1;
        for (uint8_t d = 0; d < entry.ndim; ++d) {
            if (entry.shape[d] == 0 || rcre_io_mul_overflow(numel, entry.shape[d], &numel)) {
                return RCRE_ERR_IO;
            }
        }
        size_t expected_bytes = 0;
        if (rcre_io_mul_overflow(numel, rcre_dtype_size(entry.dtype), &expected_bytes) ||
            expected_bytes != entry.data_nbytes) {
            return RCRE_ERR_IO;
        }

        /* Check for duplicate tensor names or overlapping byte intervals */
        size_t cur_begin = entry.data_offset;
        size_t cur_end   = entry.data_offset + entry.data_nbytes;
        for (size_t prev = 0; prev < arch->num_tensors; ++prev) {
            if (rcre_slice_eq(arch->entries[prev].name, entry.name)) {
                return RCRE_ERR_IO;
            }
            size_t p_begin = arch->entries[prev].data_offset;
            size_t p_end   = p_begin + arch->entries[prev].data_nbytes;
            if (cur_begin < p_end && p_begin < cur_end) {
                return RCRE_ERR_IO; /* Overlapping tensor slices */
            }
        }

        arch->entries[arch->num_tensors++] = entry;
    }

    rcre_io_json_skip_ws(json, &pos);
    if (pos != json.len || arch->num_tensors == 0) {
        return RCRE_ERR_IO;
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_safetensors_open(
    const char* filepath,
    RCRE_SafeTensorsArchive* out_archive
) {
    if (!filepath || !out_archive) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_archive, 0, sizeof(RCRE_SafeTensorsArchive));

    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_RANDOM, &out_archive->map);
    if (rc != RCRE_OK) {
        return rc;
    }

    rc = rcre_io_safetensors_parse_archive(out_archive);
    if (rc != RCRE_OK) {
        rcre_io_mmap_close(&out_archive->map);
        return rc;
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_safetensors_open_from_buffer(
    const void* buf,
    size_t size,
    RCRE_SafeTensorsArchive* out_archive
) {
    if (!buf || size == 0 || !out_archive) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_archive, 0, sizeof(RCRE_SafeTensorsArchive));

    int rc = rcre_io_mmap_from_buffer(buf, size, &out_archive->map);
    if (rc != RCRE_OK) {
        return rc;
    }

    rc = rcre_io_safetensors_parse_archive(out_archive);
    if (rc != RCRE_OK) {
        rcre_io_mmap_close(&out_archive->map);
        return rc;
    }
    return RCRE_OK;
}

void rcre_io_safetensors_close(RCRE_SafeTensorsArchive* archive) {
    if (!archive) {
        return;
    }
    rcre_io_mmap_close(&archive->map);
    memset(archive, 0, sizeof(RCRE_SafeTensorsArchive));
}

RCRE_NODISCARD int rcre_io_safetensors_get_tensor(
    const RCRE_SafeTensorsArchive* archive,
    const char* name,
    RCRE_Arena* fallback_arena,
    RCRE_Tensor* out_tensor
) {
    if (!archive || !archive->map.data || !name || !out_tensor) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_tensor, 0, sizeof(RCRE_Tensor));

    RCRE_Slice target_name = rcre_slice_from_cstr(name);
    const RCRE_SafeTensorEntry* found = nullptr;
    for (size_t i = 0; i < archive->num_tensors; ++i) {
        if (rcre_slice_eq(archive->entries[i].name, target_name)) {
            found = &archive->entries[i];
            break;
        }
    }
    if (!found) {
        return RCRE_ERR_INVALID_ARG;
    }

    const uint8_t* ptr = archive->map.data + archive->data_base_offset + found->data_offset;
    RCRE_Layout layout = (found->ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;

    if (((uintptr_t)ptr & 63U) == 0) {
        *out_tensor = rcre_tensor_from_buffer((void*)(uintptr_t)ptr, found->shape, found->ndim, found->dtype, layout);
        return RCRE_OK;
    }

    if (!fallback_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_Tensor copy_t = rcre_tensor_create(found->shape, found->ndim, found->dtype, layout, fallback_arena);
    if (!copy_t.data) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memcpy(copy_t.data, ptr, found->data_nbytes);
    *out_tensor = copy_t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_safetensors_save(
    const char* filepath,
    const char* const* names,
    const RCRE_Tensor* tensors,
    size_t num_tensors,
    RCRE_Arena* scratch
) {
    if (!filepath || !names || !tensors || !scratch ||
        num_tensors == 0 || num_tensors > RCRE_SAFETENSORS_MAX_TENSORS) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_ArenaTemp temp = rcre_arena_temp_begin(scratch);
    RCRE_StrBuilder sb = rcre_sb_init(scratch, 512);
    (void)rcre_sb_append_byte(&sb, '{');

    size_t current_offset = 0;
    for (size_t i = 0; i < num_tensors; ++i) {
        const char* tname = names[i];
        const RCRE_Tensor* t = &tensors[i];
        if (!tname || tname[0] == '\0' || !t->data || t->ndim == 0 || t->ndim > RCRE_MAX_DIMS ||
            !rcre_tensor_is_contiguous(t)) {
            rcre_arena_temp_end(temp);
            return RCRE_ERR_INVALID_ARG;
        }

        const char* dt_str = rcre_io_dtype_to_st_str(t->dtype);
        if (!dt_str) {
            rcre_arena_temp_end(temp);
            return RCRE_ERR_UNSUPPORTED_OP;
        }

        size_t nbytes = rcre_tensor_nbytes(t);
        size_t next_offset = 0;
        if (rcre_io_add_overflow(current_offset, nbytes, &next_offset)) {
            rcre_arena_temp_end(temp);
            return RCRE_ERR_INVALID_ARG;
        }

        if (i > 0) {
            (void)rcre_sb_append_byte(&sb, ',');
        }
        (void)rcre_sb_append_byte(&sb, '"');
        (void)rcre_sb_append_cstr(&sb, tname);
        (void)rcre_sb_append_cstr(&sb, "\":{\"dtype\":\"");
        (void)rcre_sb_append_cstr(&sb, dt_str);
        (void)rcre_sb_append_cstr(&sb, "\",\"shape\":[");
        for (uint8_t d = 0; d < t->ndim; ++d) {
            if (d > 0) (void)rcre_sb_append_byte(&sb, ',');
            (void)rcre_sb_append_u64(&sb, (uint64_t)t->shape[d]);
        }
        (void)rcre_sb_append_cstr(&sb, "],\"data_offsets\":[");
        (void)rcre_sb_append_u64(&sb, (uint64_t)current_offset);
        (void)rcre_sb_append_byte(&sb, ',');
        (void)rcre_sb_append_u64(&sb, (uint64_t)next_offset);
        (void)rcre_sb_append_cstr(&sb, "]}");

        current_offset = next_offset;
    }

    (void)rcre_sb_append_byte(&sb, '}');

    /* Pad JSON header with trailing spaces (' ') so that (8 + header_len)
     * lands on an exact 64-byte boundary for zero-copy SIMD mmap views. */
    size_t raw_preamble = 8 + sb.len;
    size_t aligned_preamble = (raw_preamble + 63) & ~((size_t)63);
    size_t pad_spaces = aligned_preamble - raw_preamble;
    for (size_t p = 0; p < pad_spaces; ++p) {
        (void)rcre_sb_append_byte(&sb, ' ');
    }

    RCRE_Slice hdr_slice = rcre_sb_to_slice(&sb);
    if (!hdr_slice.data || hdr_slice.len == 0) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    char tmp_path[4096];
#if defined(RCRE_IO_HAVE_POSIX)
    int tn = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", filepath, (int)getpid());
#else
    int tn = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", filepath);
#endif
    if (tn <= 0 || (size_t)tn >= sizeof(tmp_path)) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_INVALID_ARG;
    }

    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_IO;
    }

    uint8_t len_prefix[8];
    rcre_io_write_u64_le(len_prefix, (uint64_t)hdr_slice.len);

    if (fwrite(len_prefix, 1, 8, f) != 8 ||
        fwrite(hdr_slice.data, 1, hdr_slice.len, f) != hdr_slice.len) {
        fclose(f);
        (void)remove(tmp_path);
        rcre_arena_temp_end(temp);
        return RCRE_ERR_IO;
    }

    for (size_t i = 0; i < num_tensors; ++i) {
        size_t nb = rcre_tensor_nbytes(&tensors[i]);
        if (nb > 0 && fwrite(tensors[i].data, 1, nb, f) != nb) {
            fclose(f);
            (void)remove(tmp_path);
            rcre_arena_temp_end(temp);
            return RCRE_ERR_IO;
        }
    }

    if (fflush(f) != 0) {
        fclose(f);
        (void)remove(tmp_path);
        rcre_arena_temp_end(temp);
        return RCRE_ERR_IO;
    }
#if defined(RCRE_IO_HAVE_POSIX)
    (void)fsync(fileno(f));
#endif
    fclose(f);
    rcre_arena_temp_end(temp);

    if (rename(tmp_path, filepath) != 0) {
        (void)remove(tmp_path);
        return RCRE_ERR_IO;
    }

    rcre_io_record_telemetry(true, (uint64_t)(8 + hdr_slice.len + current_offset));
    return RCRE_OK;
}

/* ========================================================================= */
/* Half-Precision (IEEE FP16 & Google BF16) SIMD Conversion Engine           */
/* ========================================================================= */

static inline float rcre_f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000U) << 16;
    uint32_t exp  = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t u;
    if (exp == 31) {
        u = sign | 0x7F800000U | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int e = 0;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                e--;
            }
            e++;
            mant &= ~0x0400U;
            u = sign | ((uint32_t)(e + (127 - 15)) << 23) | (mant << 13);
        }
    } else {
        u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t rcre_f32_to_f16_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000U;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = u & 0x007FFFFFU;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000U) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00U | ((mant != 0) ? 0x0200U : 0));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static inline uint16_t rcre_f32_to_bf16_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t bias = 0x7FFFU + lsb;
    u += bias;
    return (uint16_t)(u >> 16);
}

void rcre_f16_to_f32(const void* src, float* dst, size_t numel) {
    if (!src || !dst || numel == 0) return;
    const uint16_t* s = (const uint16_t*)src;
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 4 <= numel; i += 4) {
        float16x4_t h = vld1_f16((const float16_t*)(s + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= numel; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(s + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
#endif
    for (; i < numel; ++i) {
        dst[i] = rcre_f16_to_f32_scalar(s[i]);
    }
}

void rcre_bf16_to_f32(const void* src, float* dst, size_t numel) {
    if (!src || !dst || numel == 0) return;
    const uint16_t* s = (const uint16_t*)src;
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 4 <= numel; i += 4) {
        uint16x4_t h = vld1_u16(s + i);
        uint32x4_t w = vshll_n_u16(h, 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(w));
    }
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    for (; i + 8 <= numel; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(s + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
#endif
    for (; i < numel; ++i) {
        uint32_t u = (uint32_t)s[i] << 16;
        memcpy(dst + i, &u, 4);
    }
}

void rcre_f32_to_f16(const float* src, void* dst, size_t numel) {
    if (!src || !dst || numel == 0) return;
    uint16_t* d = (uint16_t*)dst;
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 4 <= numel; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16((float16_t*)(d + i), h);
    }
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= numel; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(d + i), h);
    }
#endif
    for (; i < numel; ++i) {
        d[i] = rcre_f32_to_f16_scalar(src[i]);
    }
}

void rcre_f32_to_bf16(const float* src, void* dst, size_t numel) {
    if (!src || !dst || numel == 0) return;
    uint16_t* d = (uint16_t*)dst;
    for (size_t i = 0; i < numel; ++i) {
        d[i] = rcre_f32_to_bf16_scalar(src[i]);
    }
}

RCRE_NODISCARD int rcre_tensor_to_f32(
    const RCRE_Tensor* src,
    RCRE_Arena* arena,
    RCRE_Tensor* dst_f32
) {
    if (!src || !src->data || !dst_f32) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype == RCRE_DTYPE_F32) {
        *dst_f32 = *src;
        return RCRE_OK;
    }
    if (!arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t numel = rcre_tensor_numel(src);
    *dst_f32 = rcre_tensor_create(src->shape, src->ndim, RCRE_DTYPE_F32, src->layout, arena);
    if (!dst_f32->data) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    float* out = (float*)dst_f32->data;

    switch (src->dtype) {
        case RCRE_DTYPE_F16:
            rcre_f16_to_f32(src->data, out, numel);
            return RCRE_OK;
        case RCRE_DTYPE_BF16:
            rcre_bf16_to_f32(src->data, out, numel);
            return RCRE_OK;
        case RCRE_DTYPE_U8: {
            const uint8_t* u = (const uint8_t*)src->data;
            for (size_t i = 0; i < numel; ++i) out[i] = (float)u[i];
            return RCRE_OK;
        }
        case RCRE_DTYPE_I8: {
            const int8_t* iv = (const int8_t*)src->data;
            for (size_t i = 0; i < numel; ++i) out[i] = (float)iv[i];
            return RCRE_OK;
        }
        case RCRE_DTYPE_I32: {
            const int32_t* iv = (const int32_t*)src->data;
            for (size_t i = 0; i < numel; ++i) out[i] = (float)iv[i];
            return RCRE_OK;
        }
        case RCRE_DTYPE_I64: {
            const int64_t* iv = (const int64_t*)src->data;
            for (size_t i = 0; i < numel; ++i) out[i] = (float)iv[i];
            return RCRE_OK;
        }
        case RCRE_DTYPE_F64: {
            const double* dv = (const double*)src->data;
            for (size_t i = 0; i < numel; ++i) out[i] = (float)dv[i];
            return RCRE_OK;
        }
        default:
            return RCRE_ERR_UNSUPPORTED_OP;
    }
}

RCRE_NODISCARD int rcre_io_safetensors_get_tensor_f32(
    const RCRE_SafeTensorsArchive* archive,
    const char* name,
    RCRE_Arena* arena,
    RCRE_Tensor* out_f32_tensor
) {
    if (!archive || !name || !out_f32_tensor) {
        return RCRE_ERR_INVALID_ARG;
    }
    RCRE_Tensor raw_tensor = {0};
    int rc = rcre_io_safetensors_get_tensor(archive, name, arena, &raw_tensor);
    if (rc != RCRE_OK) {
        return rc;
    }

    if (raw_tensor.dtype == RCRE_DTYPE_F32) {
        *out_f32_tensor = raw_tensor;
        return RCRE_OK;
    }

    return rcre_tensor_to_f32(&raw_tensor, arena, out_f32_tensor);
}

RCRE_NODISCARD int rcre_io_safetensors_get_tensor_slice_f32(
    const RCRE_SafeTensorsArchive* archive,
    const char* name,
    size_t start_row,
    size_t num_rows,
    RCRE_Arena* arena,
    RCRE_Tensor* out_slice
) {
    if (!archive || !archive->map.data || !name || !arena || !out_slice || num_rows == 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_slice, 0, sizeof(RCRE_Tensor));

    RCRE_Slice target_name = rcre_slice_from_cstr(name);
    const RCRE_SafeTensorEntry* found = nullptr;
    for (size_t i = 0; i < archive->num_tensors; ++i) {
        if (rcre_slice_eq(archive->entries[i].name, target_name)) {
            found = &archive->entries[i];
            break;
        }
    }
    if (!found || found->ndim == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t total_rows = found->shape[0];
    if (start_row >= total_rows || (start_row + num_rows) > total_rows) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t row_elements = 1;
    for (uint8_t d = 1; d < found->ndim; ++d) {
        row_elements *= found->shape[d];
    }
    size_t elem_size = rcre_dtype_size(found->dtype);
    size_t row_bytes = row_elements * elem_size;
    size_t slice_bytes = num_rows * row_bytes;
    size_t slice_offset = archive->data_base_offset + found->data_offset + start_row * row_bytes;

    if (slice_offset + slice_bytes > archive->map.size) {
        return RCRE_ERR_IO;
    }

    const uint8_t* src_ptr = archive->map.data + slice_offset;
    size_t slice_shape[4] = {0};
    memcpy(slice_shape, found->shape, found->ndim * sizeof(size_t));
    slice_shape[0] = num_rows;

    RCRE_Layout layout = (found->ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;

    if (found->dtype == RCRE_DTYPE_F32) {
        if (((uintptr_t)src_ptr & 63U) == 0) {
            *out_slice = rcre_tensor_from_buffer((void*)(uintptr_t)src_ptr, slice_shape, found->ndim, RCRE_DTYPE_F32, layout);
            return RCRE_OK;
        }
        *out_slice = rcre_tensor_create(slice_shape, found->ndim, RCRE_DTYPE_F32, layout, arena);
        if (!out_slice->data) return RCRE_ERR_OUT_OF_MEMORY;
        memcpy(out_slice->data, src_ptr, slice_bytes);
        return RCRE_OK;
    }

    *out_slice = rcre_tensor_create(slice_shape, found->ndim, RCRE_DTYPE_F32, layout, arena);
    if (!out_slice->data) return RCRE_ERR_OUT_OF_MEMORY;
    float* dst = (float*)out_slice->data;
    size_t slice_numel = num_rows * row_elements;

    if (found->dtype == RCRE_DTYPE_F16) {
        rcre_f16_to_f32(src_ptr, dst, slice_numel);
    } else if (found->dtype == RCRE_DTYPE_BF16) {
        rcre_bf16_to_f32(src_ptr, dst, slice_numel);
    } else if (found->dtype == RCRE_DTYPE_U8) {
        for (size_t i = 0; i < slice_numel; ++i) dst[i] = (float)src_ptr[i];
    } else if (found->dtype == RCRE_DTYPE_I32) {
        const int32_t* iv = (const int32_t*)src_ptr;
        for (size_t i = 0; i < slice_numel; ++i) dst[i] = (float)iv[i];
    } else {
        return RCRE_ERR_UNSUPPORTED_OP;
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* 6. SIMD Byte-Shuffle Codec & Out-of-Core Ping-Pong Streamer (IO-REQ-05)   */
/* ========================================================================= */

void rcre_io_byte_shuffle_16(const void* src, void* dst, size_t num_elements) {
    if (!src || !dst || num_elements == 0) {
        return;
    }

    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d0 = (uint8_t*)dst;
    uint8_t* d1 = d0 + num_elements;

    size_t i = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 15 < num_elements; i += 16) {
        uint8x16x2_t v = vld2q_u8(s + i * 2);
        vst1q_u8(d0 + i, v.val[0]);
        vst1q_u8(d1 + i, v.val[1]);
    }
#endif

    for (; i < num_elements; ++i) {
        d0[i] = s[i * 2 + 0];
        d1[i] = s[i * 2 + 1];
    }
}

void rcre_io_byte_unshuffle_16(const void* src, void* dst, size_t num_elements) {
    if (!src || !dst || num_elements == 0) {
        return;
    }

    const uint8_t* s0 = (const uint8_t*)src;
    const uint8_t* s1 = s0 + num_elements;
    uint8_t* d = (uint8_t*)dst;

    size_t i = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 15 < num_elements; i += 16) {
        uint8x16x2_t v;
        v.val[0] = vld1q_u8(s0 + i);
        v.val[1] = vld1q_u8(s1 + i);
        vst2q_u8(d + i * 2, v);
    }
#endif

    for (; i < num_elements; ++i) {
        d[i * 2 + 0] = s0[i];
        d[i * 2 + 1] = s1[i];
    }
}

void rcre_io_byte_shuffle_32(const void* src, void* dst, size_t num_elements) {
    if (!src || !dst || num_elements == 0) {
        return;
    }

    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d0 = (uint8_t*)dst;
    uint8_t* d1 = d0 + num_elements;
    uint8_t* d2 = d1 + num_elements;
    uint8_t* d3 = d2 + num_elements;

    size_t i = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 15 < num_elements; i += 16) {
        uint8x16x4_t v = vld4q_u8(s + i * 4);
        vst1q_u8(d0 + i, v.val[0]);
        vst1q_u8(d1 + i, v.val[1]);
        vst1q_u8(d2 + i, v.val[2]);
        vst1q_u8(d3 + i, v.val[3]);
    }
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    const __m128i shuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    for (; i + 3 < num_elements; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(s + i * 4));
        __m128i t = _mm_shuffle_epi8(v, shuf_mask);
        uint32_t lane0 = (uint32_t)_mm_extract_epi32(t, 0);
        uint32_t lane1 = (uint32_t)_mm_extract_epi32(t, 1);
        uint32_t lane2 = (uint32_t)_mm_extract_epi32(t, 2);
        uint32_t lane3 = (uint32_t)_mm_extract_epi32(t, 3);
        memcpy(d0 + i, &lane0, 4);
        memcpy(d1 + i, &lane1, 4);
        memcpy(d2 + i, &lane2, 4);
        memcpy(d3 + i, &lane3, 4);
    }
#elif defined(__wasm_simd128__)
    for (; i + 3 < num_elements; i += 4) {
        v128_t v = wasm_v128_load(s + i * 4);
        v128_t t = wasm_i8x16_shuffle(v, v, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        uint32_t lane0 = (uint32_t)wasm_i32x4_extract_lane(t, 0);
        uint32_t lane1 = (uint32_t)wasm_i32x4_extract_lane(t, 1);
        uint32_t lane2 = (uint32_t)wasm_i32x4_extract_lane(t, 2);
        uint32_t lane3 = (uint32_t)wasm_i32x4_extract_lane(t, 3);
        memcpy(d0 + i, &lane0, 4);
        memcpy(d1 + i, &lane1, 4);
        memcpy(d2 + i, &lane2, 4);
        memcpy(d3 + i, &lane3, 4);
    }
#endif

    for (; i < num_elements; ++i) {
        d0[i] = s[i * 4 + 0];
        d1[i] = s[i * 4 + 1];
        d2[i] = s[i * 4 + 2];
        d3[i] = s[i * 4 + 3];
    }
}

void rcre_io_byte_unshuffle_32(const void* src, void* dst, size_t num_elements) {
    if (!src || !dst || num_elements == 0) {
        return;
    }

    const uint8_t* s0 = (const uint8_t*)src;
    const uint8_t* s1 = s0 + num_elements;
    const uint8_t* s2 = s1 + num_elements;
    const uint8_t* s3 = s2 + num_elements;
    uint8_t* d = (uint8_t*)dst;

    size_t i = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 15 < num_elements; i += 16) {
        uint8x16x4_t v;
        v.val[0] = vld1q_u8(s0 + i);
        v.val[1] = vld1q_u8(s1 + i);
        v.val[2] = vld1q_u8(s2 + i);
        v.val[3] = vld1q_u8(s3 + i);
        vst4q_u8(d + i * 4, v);
    }
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    const __m128i unshuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    for (; i + 3 < num_elements; i += 4) {
        uint32_t lane0, lane1, lane2, lane3;
        memcpy(&lane0, s0 + i, 4);
        memcpy(&lane1, s1 + i, 4);
        memcpy(&lane2, s2 + i, 4);
        memcpy(&lane3, s3 + i, 4);
        __m128i v = _mm_setr_epi32((int)lane0, (int)lane1, (int)lane2, (int)lane3);
        __m128i t = _mm_shuffle_epi8(v, unshuf_mask);
        _mm_storeu_si128((__m128i*)(d + i * 4), t);
    }
#elif defined(__wasm_simd128__)
    for (; i + 3 < num_elements; i += 4) {
        int32_t lane0, lane1, lane2, lane3;
        memcpy(&lane0, s0 + i, 4);
        memcpy(&lane1, s1 + i, 4);
        memcpy(&lane2, s2 + i, 4);
        memcpy(&lane3, s3 + i, 4);
        v128_t v = wasm_i32x4_make(lane0, lane1, lane2, lane3);
        v128_t t = wasm_i8x16_shuffle(v, v, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        wasm_v128_store(d + i * 4, t);
    }
#endif

    for (; i < num_elements; ++i) {
        d[i * 4 + 0] = s0[i];
        d[i * 4 + 1] = s1[i];
        d[i * 4 + 2] = s2[i];
        d[i * 4 + 3] = s3[i];
    }
}

/*
 * RCRE Compressed Tensor Frame Format ("RCRZ", 64-byte header):
 *  - bytes 0..3:   Magic "RCRZ"
 *  - bytes 4..7:   Version (1)
 *  - bytes 8..11:  dtype (RCRE_DType)
 *  - bytes 12..15: ndim (1..4)
 *  - bytes 16..47: shape[4] as uint64_le
 *  - bytes 48..55: uncompressed_bytes as uint64_le
 *  - bytes 56..63: compressed_payload_bytes as uint64_le
 *
 * Payload encoding on byte-shuffled buffer:
 *  Zero-dependency LZ4-architecture token + literal + 16-bit offset match stream.
 */
static size_t rcre_io_lz_compress(const uint8_t* src, size_t src_len, uint8_t* dst) {
    if (src_len == 0) {
        return 0;
    }

    uint32_t hash_table[1024];
    memset(hash_table, 0, sizeof(hash_table));

    size_t anchor = 0;
    size_t cur = 0;
    size_t d_pos = 0;
    const size_t match_limit = (src_len > 12) ? (src_len - 5) : 0;

    while (cur < match_limit) {
        uint32_t v = rcre_io_read_u32_le(src + cur);
        uint32_t h = (v * 2654435761U) >> 22;
        size_t ref = (size_t)hash_table[h];
        hash_table[h] = (uint32_t)cur;

        size_t dist = cur - ref;
        if (dist > 0 && dist <= 65535 && rcre_io_read_u32_le(src + ref) == v) {
            /* Found >= 4-byte match */
            size_t lit_len = cur - anchor;
            size_t m_len = 4;
            while (cur + m_len < src_len && src[ref + m_len] == src[cur + m_len]) {
                m_len++;
            }

            size_t m_code = m_len - 4;
            uint8_t lit_nib = (lit_len >= 15) ? 15 : (uint8_t)lit_len;
            uint8_t mat_nib = (m_code >= 15)  ? 15 : (uint8_t)m_code;
            dst[d_pos++] = (uint8_t)((lit_nib << 4) | mat_nib);

            if (lit_len >= 15) {
                size_t rem = lit_len - 15;
                while (rem >= 255) {
                    dst[d_pos++] = 255;
                    rem -= 255;
                }
                dst[d_pos++] = (uint8_t)rem;
            }

            if (lit_len > 0) {
                memcpy(dst + d_pos, src + anchor, lit_len);
                d_pos += lit_len;
            }

            rcre_io_write_u16_le(dst + d_pos, (uint16_t)dist);
            d_pos += 2;

            if (m_code >= 15) {
                size_t rem = m_code - 15;
                while (rem >= 255) {
                    dst[d_pos++] = 255;
                    rem -= 255;
                }
                dst[d_pos++] = (uint8_t)rem;
            }

            cur += m_len;
            anchor = cur;
            continue;
        }
        cur++;
    }

    /* Emit final trailing literal sequence */
    size_t tail_lit = src_len - anchor;
    uint8_t lit_nib = (tail_lit >= 15) ? 15 : (uint8_t)tail_lit;
    dst[d_pos++] = (uint8_t)(lit_nib << 4);
    if (tail_lit >= 15) {
        size_t rem = tail_lit - 15;
        while (rem >= 255) {
            dst[d_pos++] = 255;
            rem -= 255;
        }
        dst[d_pos++] = (uint8_t)rem;
    }
    if (tail_lit > 0) {
        memcpy(dst + d_pos, src + anchor, tail_lit);
        d_pos += tail_lit;
    }
    return d_pos;
}

static bool rcre_io_lz_decompress(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_cap) {
    size_t s_pos = 0;
    size_t d_pos = 0;

    while (s_pos < src_len) {
        uint8_t token = src[s_pos++];
        size_t lit_len = (size_t)(token >> 4);
        if (lit_len == 15) {
            uint8_t s_byte = 0;
            do {
                if (s_pos >= src_len) return false;
                s_byte = src[s_pos++];
                if (rcre_io_add_overflow(lit_len, (size_t)s_byte, &lit_len)) return false;
            } while (s_byte == 255);
        }

        if (s_pos + lit_len > src_len || d_pos + lit_len > dst_cap) {
            return false;
        }
        if (lit_len > 0) {
            memcpy(dst + d_pos, src + s_pos, lit_len);
            s_pos += lit_len;
            d_pos += lit_len;
        }

        /* Last sequence in block ends immediately after literals */
        if (s_pos == src_len) {
            break;
        }

        if (s_pos + 2 > src_len) {
            return false;
        }
        size_t offset = (size_t)rcre_io_read_u16_le(src + s_pos);
        s_pos += 2;
        if (offset == 0 || offset > d_pos) {
            return false;
        }

        size_t m_code = (size_t)(token & 0x0FU);
        if (m_code == 15) {
            uint8_t s_byte = 0;
            do {
                if (s_pos >= src_len) return false;
                s_byte = src[s_pos++];
                if (rcre_io_add_overflow(m_code, (size_t)s_byte, &m_code)) return false;
            } while (s_byte == 255);
        }
        size_t match_len = m_code + 4;
        if (d_pos + match_len > dst_cap) {
            return false;
        }

        size_t ref = d_pos - offset;
        if (offset >= match_len) {
            memcpy(dst + d_pos, dst + ref, match_len);
            d_pos += match_len;
        } else {
            for (size_t i = 0; i < match_len; ++i) {
                dst[d_pos++] = dst[ref + i];
            }
        }
    }

    return d_pos == dst_cap;
}

RCRE_NODISCARD int rcre_io_compress_tensor(
    const RCRE_Tensor* tensor,
    RCRE_Arena* arena,
    RCRE_Slice* out_slice
) {
    if (!tensor || !tensor->data || !arena || !out_slice ||
        tensor->ndim == 0 || tensor->ndim > 4 || !rcre_tensor_is_contiguous(tensor)) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (tensor->dtype != RCRE_DTYPE_F32 && tensor->dtype != RCRE_DTYPE_I32 &&
        tensor->dtype != RCRE_DTYPE_F16 && tensor->dtype != RCRE_DTYPE_BF16) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t numel = rcre_tensor_numel(tensor);
    size_t elem_size = rcre_dtype_size(tensor->dtype);
    size_t raw_bytes = numel * elem_size;
    if (numel == 0 || raw_bytes == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* Worst-case RLE size is raw_bytes + ceil(raw_bytes / 128) + 64 header */
    size_t max_comp = 64 + raw_bytes + (raw_bytes / 128) + 16;
    uint8_t* out_buf = (uint8_t*)rcre_arena_alloc(arena, max_comp, 64);
    if (!out_buf) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    RCRE_ArenaTemp temp = rcre_arena_temp_begin(arena);
    uint8_t* shuffled = (uint8_t*)rcre_arena_alloc(arena, raw_bytes, 64);
    if (!shuffled) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    if (elem_size == 4) {
        rcre_io_byte_shuffle_32(tensor->data, shuffled, numel);
    } else if (elem_size == 2) {
        rcre_io_byte_shuffle_16(tensor->data, shuffled, numel);
    }
    size_t comp_payload = rcre_io_lz_compress(shuffled, raw_bytes, out_buf + 64);
    rcre_arena_temp_end(temp);

    memset(out_buf, 0, 64);
    memcpy(out_buf, "RCRZ", 4);
    rcre_io_write_u32_le(out_buf + 4, 1U);
    rcre_io_write_u32_le(out_buf + 8, (uint32_t)tensor->dtype);
    rcre_io_write_u32_le(out_buf + 12, (uint32_t)tensor->ndim);
    for (uint8_t d = 0; d < tensor->ndim; ++d) {
        rcre_io_write_u64_le(out_buf + 16 + d * 8, (uint64_t)tensor->shape[d]);
    }
    rcre_io_write_u64_le(out_buf + 48, (uint64_t)raw_bytes);
    rcre_io_write_u64_le(out_buf + 56, (uint64_t)comp_payload);

    *out_slice = rcre_slice_from_parts(out_buf, 64 + comp_payload);
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_decompress_tensor(
    RCRE_Slice compressed_frame,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
) {
    if (!compressed_frame.data || compressed_frame.len < 64 || !arena || !out_tensor) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_tensor, 0, sizeof(RCRE_Tensor));

    const uint8_t* hdr = compressed_frame.data;
    if (memcmp(hdr, "RCRZ", 4) != 0 || rcre_io_read_u32_le(hdr + 4) != 1U) {
        return RCRE_ERR_IO;
    }

    RCRE_DType dtype = (RCRE_DType)rcre_io_read_u32_le(hdr + 8);
    uint32_t ndim_u32 = rcre_io_read_u32_le(hdr + 12);
    if ((dtype != RCRE_DTYPE_F32 && dtype != RCRE_DTYPE_I32 &&
         dtype != RCRE_DTYPE_F16 && dtype != RCRE_DTYPE_BF16) || ndim_u32 == 0 || ndim_u32 > 4) {
        return RCRE_ERR_IO;
    }
    uint8_t ndim = (uint8_t)ndim_u32;
    size_t elem_size = rcre_dtype_size(dtype);

    size_t shape[4] = {0};
    size_t numel = 1;
    for (uint8_t d = 0; d < ndim; ++d) {
        uint64_t dim_u64 = rcre_io_read_u64_le(hdr + 16 + d * 8);
        if (dim_u64 == 0 || dim_u64 > (uint64_t)SIZE_MAX ||
            rcre_io_mul_overflow(numel, (size_t)dim_u64, &numel)) {
            return RCRE_ERR_IO;
        }
        shape[d] = (size_t)dim_u64;
    }

    uint64_t raw_bytes_u64  = rcre_io_read_u64_le(hdr + 48);
    uint64_t comp_bytes_u64 = rcre_io_read_u64_le(hdr + 56);
    size_t expected_raw = 0;
    if (rcre_io_mul_overflow(numel, elem_size, &expected_raw) ||
        raw_bytes_u64 != (uint64_t)expected_raw ||
        64ULL + comp_bytes_u64 != (uint64_t)compressed_frame.len) {
        return RCRE_ERR_IO;
    }

    RCRE_Layout layout = (ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;
    RCRE_Tensor t = rcre_tensor_create(shape, ndim, dtype, layout, arena);
    if (!t.data) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    RCRE_ArenaTemp temp = rcre_arena_temp_begin(arena);
    uint8_t* shuffled = (uint8_t*)rcre_arena_alloc(arena, expected_raw, 64);
    if (!shuffled) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    if (!rcre_io_lz_decompress(hdr + 64, (size_t)comp_bytes_u64, shuffled, expected_raw)) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_IO;
    }

    if (elem_size == 4) {
        rcre_io_byte_unshuffle_32(shuffled, t.data, numel);
    } else if (elem_size == 2) {
        rcre_io_byte_unshuffle_16(shuffled, t.data, numel);
    }
    rcre_arena_temp_end(temp);

    *out_tensor = t;
    return RCRE_OK;
}

/* --- Out-of-Core Ping-Pong Arena Double-Buffered Streamer --- */

#if defined(RCRE_IO_HAVE_POSIX)
typedef struct RCRE_IOAsyncPrefetch {
    RCRE_TensorStream* stream;
    pthread_mutex_t    mu;
    pthread_cond_t     cv_req;
    pthread_cond_t     cv_done;
    pthread_t          thr;
    bool               started;
    bool               stop;
    bool               req_pending;
    size_t             req_row;
    uint8_t            req_slot;
    bool               has_ready;
    size_t             ready_row;
    size_t             ready_rows;
    uint8_t            ready_slot;
} RCRE_IOAsyncPrefetch;

static void* rcre_io_prefetch_worker(void* arg) {
    RCRE_IOAsyncPrefetch* pf = (RCRE_IOAsyncPrefetch*)arg;
    pthread_mutex_lock(&pf->mu);
    for (;;) {
        while (!pf->stop && !pf->req_pending) {
            pthread_cond_wait(&pf->cv_req, &pf->mu);
        }
        if (pf->stop) {
            break;
        }
        size_t row = pf->req_row;
        uint8_t slot = pf->req_slot;
        RCRE_TensorStream* s = pf->stream;
        size_t rem_rows = (row < s->total_rows) ? (s->total_rows - row) : 0;
        size_t cur_rows = (rem_rows < s->batch_rows) ? rem_rows : s->batch_rows;
        size_t copy_bytes = cur_rows * s->row_bytes;
        size_t file_off = s->data_offset + row * s->row_bytes;
        const uint8_t* src_ptr = s->map.data + file_off;
        uint8_t* dst_ptr = s->ping_pong_buf[slot];
        pthread_mutex_unlock(&pf->mu);

        if (copy_bytes > 0) {
            (void)rcre_io_madvise_range(&s->map, file_off, copy_bytes, RCRE_IO_HINT_SEQUENTIAL);
            memcpy(dst_ptr, src_ptr, copy_bytes);
        }

        pthread_mutex_lock(&pf->mu);
        pf->req_pending = false;
        pf->has_ready   = (cur_rows > 0);
        pf->ready_row   = row;
        pf->ready_rows  = cur_rows;
        pf->ready_slot  = slot;
        pthread_cond_broadcast(&pf->cv_done);
    }
    pthread_mutex_unlock(&pf->mu);
    return nullptr;
}
#endif

RCRE_NODISCARD int rcre_io_stream_open_npy(
    const char* filepath,
    size_t batch_rows,
    RCRE_Arena* arena,
    RCRE_TensorStream* out_stream
) {
    if (!filepath || batch_rows == 0 || !arena || !out_stream) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_stream, 0, sizeof(RCRE_TensorStream));

    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &out_stream->map);
    if (rc != RCRE_OK) {
        return rc;
    }

    size_t data_nbytes = 0;
    rc = rcre_io_parse_npy_header(
        rcre_io_mmap_slice(&out_stream->map),
        &out_stream->dtype,
        &out_stream->ndim,
        out_stream->full_shape,
        &out_stream->data_offset,
        &data_nbytes
    );
    if (rc != RCRE_OK) {
        rcre_io_mmap_close(&out_stream->map);
        return rc;
    }

    size_t row_elems = 1;
    for (uint8_t d = 1; d < out_stream->ndim; ++d) {
        row_elems *= out_stream->full_shape[d];
    }
    size_t row_bytes = row_elems * rcre_dtype_size(out_stream->dtype);
    size_t buf_bytes = 0;
    if (rcre_io_mul_overflow(batch_rows, row_bytes, &buf_bytes)) {
        rcre_io_mmap_close(&out_stream->map);
        return RCRE_ERR_INVALID_ARG;
    }

    out_stream->ping_pong_buf[0] = (uint8_t*)rcre_arena_alloc(arena, buf_bytes, 64);
    out_stream->ping_pong_buf[1] = (uint8_t*)rcre_arena_alloc(arena, buf_bytes, 64);
    if (!out_stream->ping_pong_buf[0] || !out_stream->ping_pong_buf[1]) {
        rcre_io_mmap_close(&out_stream->map);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    out_stream->layout       = (out_stream->ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;
    out_stream->row_elements = row_elems;
    out_stream->row_bytes    = row_bytes;
    out_stream->total_rows   = out_stream->full_shape[0];
    out_stream->batch_rows   = batch_rows;
    out_stream->current_row  = 0;
    out_stream->active_slot  = 0;
    out_stream->async_enabled = false;
    out_stream->use_io_uring = false;
    out_stream->async_handle = 0;

    return RCRE_OK;
}

RCRE_NODISCARD bool rcre_io_is_io_uring_available(void) {
#if defined(__linux__) && defined(__NR_io_uring_setup)
    static int s_cached = -1;
    if (s_cached != -1) {
        return s_cached == 1;
    }
    struct {
        uint32_t sq_entries;
        uint32_t cq_entries;
        uint32_t flags;
        uint32_t sq_thread_cpu;
        uint32_t sq_thread_idle;
        uint32_t features;
        uint32_t wq_fd;
        uint32_t resv[3];
        struct { uint32_t head, tail, ring_mask, ring_entries, flags, dropped, array, resv[3]; } sq_off;
        struct { uint32_t head, tail, ring_mask, ring_entries, overflow, cqes, flags, resv[3]; } cq_off;
    } p;
    memset(&p, 0, sizeof(p));
    long fd = syscall(__NR_io_uring_setup, 1, &p);
    if (fd >= 0) {
        close((int)fd);
        s_cached = 1;
        return true;
    }
    s_cached = 0;
    return false;
#else
    return false;
#endif
}

RCRE_NODISCARD int rcre_io_stream_enable_async(RCRE_TensorStream* stream) {
    if (!stream || !stream->map.data) {
        return RCRE_ERR_INVALID_ARG;
    }
    stream->async_enabled = true;
    if (rcre_io_is_io_uring_available()) {
        stream->use_io_uring = true;
    } else {
        stream->use_io_uring = false;
        (void)rcre_io_madvise_range(&stream->map, 0, stream->map.size, RCRE_IO_HINT_SEQUENTIAL);
    }
#if defined(RCRE_IO_HAVE_POSIX)
    if (!stream->prefetch_ctx && stream->total_rows > stream->batch_rows) {
        RCRE_IOAsyncPrefetch* pf = (RCRE_IOAsyncPrefetch*)calloc(1, sizeof(RCRE_IOAsyncPrefetch));
        if (pf) {
            pf->stream = stream;
            pthread_mutex_init(&pf->mu, nullptr);
            pthread_cond_init(&pf->cv_req, nullptr);
            pthread_cond_init(&pf->cv_done, nullptr);
            if (pthread_create(&pf->thr, nullptr, rcre_io_prefetch_worker, pf) == 0) {
                pf->started = true;
                stream->prefetch_ctx = pf;
                /* Kick off slot 0 prefetch immediately */
                pthread_mutex_lock(&pf->mu);
                pf->req_row     = 0;
                pf->req_slot    = 0;
                pf->req_pending = true;
                pthread_cond_signal(&pf->cv_req);
                pthread_mutex_unlock(&pf->mu);
            } else {
                pthread_cond_destroy(&pf->cv_done);
                pthread_cond_destroy(&pf->cv_req);
                pthread_mutex_destroy(&pf->mu);
                free(pf);
            }
        }
    }
#endif
    return RCRE_OK;
}

RCRE_NODISCARD bool rcre_io_stream_next(
    RCRE_TensorStream* stream,
    RCRE_Tensor* out_batch
) {
    if (!stream || !stream->map.data || !out_batch || stream->current_row >= stream->total_rows) {
        return false;
    }

    size_t cur_rows = 0;
    uint8_t slot = stream->active_slot;
    uint8_t* dst = stream->ping_pong_buf[slot];

#if defined(RCRE_IO_HAVE_POSIX)
    RCRE_IOAsyncPrefetch* pf = (RCRE_IOAsyncPrefetch*)stream->prefetch_ctx;
    if (pf && pf->started) {
        pthread_mutex_lock(&pf->mu);
        while (pf->req_pending) {
            pthread_cond_wait(&pf->cv_done, &pf->mu);
        }
        if (pf->has_ready && pf->ready_row == stream->current_row) {
            slot = pf->ready_slot;
            dst = stream->ping_pong_buf[slot];
            cur_rows = pf->ready_rows;
            pf->has_ready = false;
        } else {
            size_t rem_rows = stream->total_rows - stream->current_row;
            cur_rows = (rem_rows < stream->batch_rows) ? rem_rows : stream->batch_rows;
            size_t copy_bytes = cur_rows * stream->row_bytes;
            size_t file_off = stream->data_offset + stream->current_row * stream->row_bytes;
            memcpy(dst, stream->map.data + file_off, copy_bytes);
            pf->has_ready = false;
        }
        size_t next_row = stream->current_row + cur_rows;
        if (next_row < stream->total_rows) {
            pf->req_row     = next_row;
            pf->req_slot    = slot ^ 1U;
            pf->req_pending = true;
            pthread_cond_signal(&pf->cv_req);
        }
        pthread_mutex_unlock(&pf->mu);
    } else
#endif
    {
        size_t rem_rows = stream->total_rows - stream->current_row;
        cur_rows = (rem_rows < stream->batch_rows) ? rem_rows : stream->batch_rows;
        size_t copy_bytes = cur_rows * stream->row_bytes;
        size_t file_off = stream->data_offset + stream->current_row * stream->row_bytes;
        memcpy(dst, stream->map.data + file_off, copy_bytes);

        size_t next_off = file_off + copy_bytes;
        if (next_off < stream->map.size) {
            size_t next_rem = stream->total_rows - (stream->current_row + cur_rows);
            size_t next_rows = (next_rem < stream->batch_rows) ? next_rem : stream->batch_rows;
            size_t next_bytes = next_rows * stream->row_bytes;
            (void)rcre_io_madvise_range(&stream->map, next_off, next_bytes, RCRE_IO_HINT_SEQUENTIAL);
        }
    }

    size_t batch_shape[RCRE_MAX_DIMS] = {0};
    memcpy(batch_shape, stream->full_shape, stream->ndim * sizeof(size_t));
    batch_shape[0] = cur_rows;

    *out_batch = rcre_tensor_from_buffer(dst, batch_shape, stream->ndim, stream->dtype, stream->layout);

    stream->current_row += cur_rows;
    stream->active_slot = slot ^ 1U;
    return true;
}

void rcre_io_stream_reset(RCRE_TensorStream* stream) {
    if (stream) {
#if defined(RCRE_IO_HAVE_POSIX)
        RCRE_IOAsyncPrefetch* pf = (RCRE_IOAsyncPrefetch*)stream->prefetch_ctx;
        if (pf && pf->started) {
            pthread_mutex_lock(&pf->mu);
            while (pf->req_pending) {
                pthread_cond_wait(&pf->cv_done, &pf->mu);
            }
            pf->has_ready = false;
            pthread_mutex_unlock(&pf->mu);
        }
#endif
        stream->current_row = 0;
        stream->active_slot = 0;
    }
}

void rcre_io_stream_close(RCRE_TensorStream* stream) {
    if (!stream) {
        return;
    }
#if defined(RCRE_IO_HAVE_POSIX)
    RCRE_IOAsyncPrefetch* pf = (RCRE_IOAsyncPrefetch*)stream->prefetch_ctx;
    if (pf) {
        if (pf->started) {
            pthread_mutex_lock(&pf->mu);
            while (pf->req_pending) {
                pthread_cond_wait(&pf->cv_done, &pf->mu);
            }
            pf->stop = true;
            pthread_cond_signal(&pf->cv_req);
            pthread_mutex_unlock(&pf->mu);
            pthread_join(pf->thr, nullptr);
        }
        pthread_cond_destroy(&pf->cv_done);
        pthread_cond_destroy(&pf->cv_req);
        pthread_mutex_destroy(&pf->mu);
        free(pf);
        stream->prefetch_ctx = nullptr;
    }
#endif
#if defined(__linux__)
    if (stream->use_io_uring && stream->async_handle > 0) {
        close((int)stream->async_handle);
        stream->async_handle = 0;
    }
#endif
    rcre_io_mmap_close(&stream->map);
    memset(stream, 0, sizeof(RCRE_TensorStream));
}

/* ========================================================================= */
/* 7. Zero-Copy Cross-Framework C ABI Bridges: DLPack & Arrow (IO-REQ-06)    */
/* ========================================================================= */

RCRE_NODISCARD int rcre_io_to_dlpack(
    const RCRE_Tensor* src,
    RCRE_Arena* meta_arena,
    RCRE_DLTensor* out_dl
) {
    if (!src || !src->data || !meta_arena || !out_dl || src->ndim == 0 || src->ndim > RCRE_MAX_DIMS) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_dl, 0, sizeof(RCRE_DLTensor));

    uint8_t code = 0;
    uint8_t bits = 0;
    switch (src->dtype) {
        case RCRE_DTYPE_U8:   code = (uint8_t)RCRE_DL_UINT;  bits = 8;  break;
        case RCRE_DTYPE_I8:   code = (uint8_t)RCRE_DL_INT;   bits = 8;  break;
        case RCRE_DTYPE_F16:  code = (uint8_t)RCRE_DL_FLOAT; bits = 16; break;
        case RCRE_DTYPE_BF16: code = 4;                       bits = 16; break; /* kDLBfloat = 4 */
        case RCRE_DTYPE_I32:  code = (uint8_t)RCRE_DL_INT;   bits = 32; break;
        case RCRE_DTYPE_I64:  code = (uint8_t)RCRE_DL_INT;   bits = 64; break;
        case RCRE_DTYPE_F32:  code = (uint8_t)RCRE_DL_FLOAT; bits = 32; break;
        case RCRE_DTYPE_F64:  code = (uint8_t)RCRE_DL_FLOAT; bits = 64; break;
        default: return RCRE_ERR_UNSUPPORTED_OP;
    }

    int64_t* dl_shape   = (int64_t*)rcre_arena_alloc(meta_arena, src->ndim * sizeof(int64_t), alignof(int64_t));
    int64_t* dl_strides = (int64_t*)rcre_arena_alloc(meta_arena, src->ndim * sizeof(int64_t), alignof(int64_t));
    if (!dl_shape || !dl_strides) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (uint8_t i = 0; i < src->ndim; ++i) {
        dl_shape[i]   = (int64_t)src->shape[i];
        dl_strides[i] = (int64_t)src->strides[i];
    }

    out_dl->data        = src->data;
    out_dl->device_type = (int32_t)RCRE_DL_DEVICE_CPU;
    out_dl->device_id   = 0;
    out_dl->ndim        = (int32_t)src->ndim;
    out_dl->dtype_code  = code;
    out_dl->dtype_bits  = bits;
    out_dl->dtype_lanes = 1;
    out_dl->shape       = dl_shape;
    out_dl->strides     = dl_strides;
    out_dl->byte_offset = 0;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_from_dlpack(
    const RCRE_DLTensor* dl,
    RCRE_Tensor* out_view
) {
    if (!dl || !dl->data || !dl->shape || !out_view || dl->ndim <= 0 || dl->ndim > RCRE_MAX_DIMS) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (dl->device_type != (int32_t)RCRE_DL_DEVICE_CPU || dl->dtype_lanes != 1) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    RCRE_DType dtype = RCRE_DTYPE_F32;
    if (dl->dtype_code == (uint8_t)RCRE_DL_UINT && dl->dtype_bits == 8) {
        dtype = RCRE_DTYPE_U8;
    } else if (dl->dtype_code == (uint8_t)RCRE_DL_INT && dl->dtype_bits == 8) {
        dtype = RCRE_DTYPE_I8;
    } else if (dl->dtype_code == (uint8_t)RCRE_DL_INT && dl->dtype_bits == 32) {
        dtype = RCRE_DTYPE_I32;
    } else if (dl->dtype_code == (uint8_t)RCRE_DL_INT && dl->dtype_bits == 64) {
        dtype = RCRE_DTYPE_I64;
    } else if (dl->dtype_code == (uint8_t)RCRE_DL_FLOAT && dl->dtype_bits == 32) {
        dtype = RCRE_DTYPE_F32;
    } else if (dl->dtype_code == (uint8_t)RCRE_DL_FLOAT && dl->dtype_bits == 64) {
        dtype = RCRE_DTYPE_F64;
    } else if (dl->dtype_code == (uint8_t)RCRE_DL_FLOAT && dl->dtype_bits == 16) {
        dtype = RCRE_DTYPE_F16;
    } else if (dl->dtype_code == 4 && dl->dtype_bits == 16) {
        dtype = RCRE_DTYPE_BF16;
    } else {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t shape[RCRE_MAX_DIMS] = {0};
    for (int32_t i = 0; i < dl->ndim; ++i) {
        if (dl->shape[i] <= 0) {
            return RCRE_ERR_INVALID_ARG;
        }
        shape[i] = (size_t)dl->shape[i];
    }

    uint8_t* base_ptr = (uint8_t*)dl->data + dl->byte_offset;
    RCRE_Layout layout = (dl->ndim == 3) ? RCRE_LAYOUT_HWC : RCRE_LAYOUT_TABULAR;
    RCRE_Tensor view = rcre_tensor_from_buffer(base_ptr, shape, (uint8_t)dl->ndim, dtype, layout);

    if (dl->strides) {
        for (int32_t i = 0; i < dl->ndim; ++i) {
            view.strides[i] = (ptrdiff_t)dl->strides[i];
        }
    }

    *out_view = view;
    return RCRE_OK;
}

static void rcre_io_arrow_noop_release_schema(RCRE_ArrowSchema* schema) {
    if (schema) {
        schema->release = nullptr;
    }
}

static void rcre_io_arrow_noop_release_array(RCRE_ArrowArray* array) {
    if (array) {
        array->release = nullptr;
    }
}

RCRE_NODISCARD int rcre_io_to_arrow(
    const RCRE_Tensor* src,
    RCRE_Arena* meta_arena,
    RCRE_ArrowSchema* out_schema,
    RCRE_ArrowArray* out_array
) {
    if (!src || !src->data || !meta_arena || !out_schema || !out_array ||
        src->ndim != 1 || !rcre_tensor_is_contiguous(src)) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_schema, 0, sizeof(RCRE_ArrowSchema));
    memset(out_array, 0, sizeof(RCRE_ArrowArray));

    const char* fmt = nullptr;
    switch (src->dtype) {
        case RCRE_DTYPE_U8:   fmt = "C"; break;
        case RCRE_DTYPE_I8:   fmt = "c"; break;
        case RCRE_DTYPE_F16:  fmt = "e"; break;
        case RCRE_DTYPE_BF16: fmt = "b"; break;
        case RCRE_DTYPE_I32:  fmt = "i"; break;
        case RCRE_DTYPE_I64:  fmt = "l"; break;
        case RCRE_DTYPE_F32:  fmt = "f"; break;
        case RCRE_DTYPE_F64:  fmt = "g"; break;
        default: return RCRE_ERR_UNSUPPORTED_OP;
    }

    const void** bufs = (const void**)rcre_arena_alloc(meta_arena, 2 * sizeof(void*), alignof(void*));
    if (!bufs) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    bufs[0] = nullptr;   /* No null validity bitmap (null_count == 0) */
    bufs[1] = src->data; /* Zero-copy data buffer pointer             */

    out_schema->format       = fmt;
    out_schema->name         = src->label ? src->label : "rcre_col";
    out_schema->flags        = 0;
    out_schema->n_children   = 0;
    out_schema->release      = rcre_io_arrow_noop_release_schema;

    out_array->length        = (int64_t)src->shape[0];
    out_array->null_count    = 0;
    out_array->offset        = 0;
    out_array->n_buffers     = 2;
    out_array->n_children    = 0;
    out_array->buffers       = bufs;
    out_array->release       = rcre_io_arrow_noop_release_array;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_io_from_arrow(
    const RCRE_ArrowSchema* schema,
    const RCRE_ArrowArray* array,
    RCRE_Tensor* out_view
) {
    if (!schema || !schema->format || !array || !array->buffers || !out_view ||
        array->n_buffers < 2 || !array->buffers[1] || array->length <= 0 || array->null_count != 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_DType dtype = RCRE_DTYPE_F32;
    if (strcmp(schema->format, "f") == 0) {
        dtype = RCRE_DTYPE_F32;
    } else if (strcmp(schema->format, "g") == 0) {
        dtype = RCRE_DTYPE_F64;
    } else if (strcmp(schema->format, "e") == 0) {
        dtype = RCRE_DTYPE_F16;
    } else if (strcmp(schema->format, "b") == 0) {
        dtype = RCRE_DTYPE_BF16;
    } else if (strcmp(schema->format, "i") == 0) {
        dtype = RCRE_DTYPE_I32;
    } else if (strcmp(schema->format, "l") == 0) {
        dtype = RCRE_DTYPE_I64;
    } else if (strcmp(schema->format, "c") == 0) {
        dtype = RCRE_DTYPE_I8;
    } else if (strcmp(schema->format, "C") == 0) {
        dtype = RCRE_DTYPE_U8;
    } else {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t elem_sz = rcre_dtype_size(dtype);
    uint8_t* ptr = (uint8_t*)(uintptr_t)array->buffers[1] + ((size_t)array->offset * elem_sz);
    size_t shape[1] = { (size_t)array->length };
    *out_view = rcre_tensor_from_buffer(ptr, shape, 1, dtype, RCRE_LAYOUT_TABULAR);
    return RCRE_OK;
}
