/**
 * @file test_framework.h
 * @brief Pragmatic C23 Zero-Dependency Unit Testing Framework for RCRE
 * 
 * Features:
 *  - Rich assertion macros with formatted failure diagnostics
 *  - Zero dynamic heap allocations in runner infrastructure
 *  - No setjmp/longjmp hazards (100% AddressSanitizer & threadpool safe)
 *  - Sub-millisecond monotonic execution timer per test case
 *  - ANSI color-coded pass/fail reporting (supports NO_COLOR / isatty)
 *  - Array and buffer comparison helpers (floats, bytes, strings)
 */

#ifndef RCRE_TEST_FRAMEWORK_H
#define RCRE_TEST_FRAMEWORK_H

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ========================================================================= */
/* Terminal ANSI Styling (Auto-disabled when piped or NO_COLOR defined)      */
/* ========================================================================= */

static inline bool rcre_test_use_color(void) {
    if (getenv("NO_COLOR") != NULL) return false;
    return isatty(fileno(stdout));
}

#define RCRE_CLR_RESET   (rcre_test_use_color() ? "\033[0m" : "")
#define RCRE_CLR_RED     (rcre_test_use_color() ? "\033[1;31m" : "")
#define RCRE_CLR_GREEN   (rcre_test_use_color() ? "\033[1;32m" : "")
#define RCRE_CLR_YELLOW  (rcre_test_use_color() ? "\033[1;33m" : "")
#define RCRE_CLR_BLUE    (rcre_test_use_color() ? "\033[1;34m" : "")
#define RCRE_CLR_CYAN    (rcre_test_use_color() ? "\033[1;36m" : "")
#define RCRE_CLR_BOLD    (rcre_test_use_color() ? "\033[1m" : "")
#define RCRE_CLR_DIM     (rcre_test_use_color() ? "\033[2m" : "")

/* ========================================================================= */
/* Global Suite State & Monotonic Timer                                      */
/* ========================================================================= */

typedef struct {
    const char* suite_name;
    const char* current_test;
    size_t tests_run;
    size_t tests_passed;
    size_t tests_failed;
    size_t assertions_checked;
    bool current_test_failed;
} RCRE_TestSuiteState;

static RCRE_TestSuiteState s_test_suite_state = {0};

static inline double rcre_test_get_time_ms(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

static inline void test_suite_begin(const char* suite_name) {
    s_test_suite_state.suite_name = suite_name;
    s_test_suite_state.tests_run = 0;
    s_test_suite_state.tests_passed = 0;
    s_test_suite_state.tests_failed = 0;
    s_test_suite_state.assertions_checked = 0;
    printf("\n%s======================================================================%s\n", RCRE_CLR_CYAN, RCRE_CLR_RESET);
    printf("%s🧪 %s%s\n", RCRE_CLR_BOLD, suite_name, RCRE_CLR_RESET);
    printf("%s======================================================================%s\n", RCRE_CLR_CYAN, RCRE_CLR_RESET);
}

static inline int test_suite_report(void) {
    printf("%s----------------------------------------------------------------------%s\n", RCRE_CLR_CYAN, RCRE_CLR_RESET);
    if (s_test_suite_state.tests_failed == 0) {
        printf("  %s🎉 All %zu Tests Passed (%zu assertions)%s\n", 
               RCRE_CLR_GREEN, s_test_suite_state.tests_run, s_test_suite_state.assertions_checked, RCRE_CLR_RESET);
    } else {
        printf("  %s❌ FAILED: %zu/%zu Tests Failed (%zu passed, %zu assertions)%s\n", 
               RCRE_CLR_RED, s_test_suite_state.tests_failed, s_test_suite_state.tests_run, 
               s_test_suite_state.tests_passed, s_test_suite_state.assertions_checked, RCRE_CLR_RESET);
    }
    printf("%s======================================================================%s\n\n", RCRE_CLR_CYAN, RCRE_CLR_RESET);
    return (s_test_suite_state.tests_failed > 0) ? 1 : 0;
}

/* ========================================================================= */
/* Test Runner Macro                                                         */
/* ========================================================================= */

#define TEST_RUN(test_func) do { \
    s_test_suite_state.current_test = #test_func; \
    s_test_suite_state.current_test_failed = false; \
    s_test_suite_state.tests_run++; \
    size_t _assertions_before = s_test_suite_state.assertions_checked; \
    double _t0 = rcre_test_get_time_ms(); \
    test_func(); \
    double _elapsed_ms = rcre_test_get_time_ms() - _t0; \
    size_t _test_assertions = s_test_suite_state.assertions_checked - _assertions_before; \
    if (!s_test_suite_state.current_test_failed) { \
        s_test_suite_state.tests_passed++; \
        printf("  %s✓ [PASS]%s %s%-40s%s %s(%zu checks, %.2f ms)%s\n", \
               RCRE_CLR_GREEN, RCRE_CLR_RESET, \
               RCRE_CLR_BOLD, #test_func, RCRE_CLR_RESET, \
               RCRE_CLR_DIM, _test_assertions, _elapsed_ms, RCRE_CLR_RESET); \
    } else { \
        s_test_suite_state.tests_failed++; \
        printf("  %s✗ [FAIL]%s %s%-40s%s %s(%zu checks, %.2f ms)%s\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, \
               RCRE_CLR_BOLD, #test_func, RCRE_CLR_RESET, \
               RCRE_CLR_DIM, _test_assertions, _elapsed_ms, RCRE_CLR_RESET); \
    } \
} while (0)

/* ========================================================================= */
/* Diagnostic Assertion Macros                                              */
/* ========================================================================= */

#define ASSERT_TRUE(cond) do { \
    s_test_suite_state.assertions_checked++; \
    if (!(cond)) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d:\n" \
               "      Expression '%s' is FALSE (expected TRUE)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #cond); \
        return; \
    } \
} while (0)

#define ASSERT_FALSE(cond) do { \
    s_test_suite_state.assertions_checked++; \
    if (cond) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d:\n" \
               "      Expression '%s' is TRUE (expected FALSE)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #cond); \
        return; \
    } \
} while (0)

#define ASSERT_NOT_NULL(ptr) do { \
    s_test_suite_state.assertions_checked++; \
    if ((ptr) == NULL) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d:\n" \
               "      Pointer '%s' is NULL (expected non-NULL)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #ptr); \
        return; \
    } \
} while (0)

#define ASSERT_NULL(ptr) do { \
    s_test_suite_state.assertions_checked++; \
    if ((ptr) != NULL) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d:\n" \
               "      Pointer '%s' is %p (expected NULL)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #ptr, (void*)(ptr)); \
        return; \
    } \
} while (0)

#define ASSERT_INT_EQ(expected, actual) do { \
    s_test_suite_state.assertions_checked++; \
    long long _exp = (long long)(expected); \
    long long _act = (long long)(actual); \
    if (_exp != _act) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_INT_EQ(%s, %s)\n" \
               "      Expected: %lld (0x%llX)\n" \
               "      Actual:   %lld (0x%llX)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, \
               _exp, (unsigned long long)_exp, _act, (unsigned long long)_act); \
        return; \
    } \
} while (0)

#define ASSERT_UINT_EQ(expected, actual) do { \
    s_test_suite_state.assertions_checked++; \
    unsigned long long _exp = (unsigned long long)(expected); \
    unsigned long long _act = (unsigned long long)(actual); \
    if (_exp != _act) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_UINT_EQ(%s, %s)\n" \
               "      Expected: %llu (0x%llX)\n" \
               "      Actual:   %llu (0x%llX)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, \
               _exp, _exp, _act, _act); \
        return; \
    } \
} while (0)

#define ASSERT_FLOAT_NEAR(expected, actual, eps) do { \
    s_test_suite_state.assertions_checked++; \
    float _exp = (float)(expected); \
    float _act = (float)(actual); \
    float _eps = (float)(eps); \
    float _diff = fabsf(_exp - _act); \
    if (isnan(_act) || isnan(_exp) || _diff > _eps) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_FLOAT_NEAR(%s, %s, %s)\n" \
               "      Expected: %+.7f\n" \
               "      Actual:   %+.7f\n" \
               "      Diff:     %.6e (Tolerance: %.6e)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, #eps, \
               _exp, _act, _diff, _eps); \
        return; \
    } \
} while (0)

#define ASSERT_DOUBLE_NEAR(expected, actual, eps) do { \
    s_test_suite_state.assertions_checked++; \
    double _exp = (double)(expected); \
    double _act = (double)(actual); \
    double _eps = (double)(eps); \
    double _diff = fabs(_exp - _act); \
    if (isnan(_act) || isnan(_exp) || _diff > _eps) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_DOUBLE_NEAR(%s, %s, %s)\n" \
               "      Expected: %+.12f\n" \
               "      Actual:   %+.12f\n" \
               "      Diff:     %.8e (Tolerance: %.8e)\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, #eps, \
               _exp, _act, _diff, _eps); \
        return; \
    } \
} while (0)

#define ASSERT_FLOAT_ARRAY_NEAR(expected, actual, count, eps) do { \
    const float* _exp_arr = (const float*)(expected); \
    const float* _act_arr = (const float*)(actual); \
    size_t _cnt = (size_t)(count); \
    float _eps = (float)(eps); \
    for (size_t _i = 0; _i < _cnt; ++_i) { \
        s_test_suite_state.assertions_checked++; \
        float _diff = fabsf(_exp_arr[_i] - _act_arr[_i]); \
        if (isnan(_act_arr[_i]) || isnan(_exp_arr[_i]) || _diff > _eps) { \
            s_test_suite_state.current_test_failed = true; \
            printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_FLOAT_ARRAY_NEAR(%s, %s, %s) at index [%zu]:\n" \
                   "      Expected: %+.7f\n" \
                   "      Actual:   %+.7f\n" \
                   "      Diff:     %.6e (Tolerance: %.6e)\n", \
                   RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, #eps, \
                   _i, _exp_arr[_i], _act_arr[_i], _diff, _eps); \
            return; \
        } \
    } \
} while (0)

#define ASSERT_MEM_EQ(expected, actual, size) do { \
    const uint8_t* _exp_mem = (const uint8_t*)(expected); \
    const uint8_t* _act_mem = (const uint8_t*)(actual); \
    size_t _sz = (size_t)(size); \
    for (size_t _i = 0; _i < _sz; ++_i) { \
        s_test_suite_state.assertions_checked++; \
        if (_exp_mem[_i] != _act_mem[_i]) { \
            s_test_suite_state.current_test_failed = true; \
            printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_MEM_EQ(%s, %s, %s) mismatch at byte [%zu]:\n" \
                   "      Expected: 0x%02X (%u)\n" \
                   "      Actual:   0x%02X (%u)\n", \
                   RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, #size, \
                   _i, _exp_mem[_i], _exp_mem[_i], _act_mem[_i], _act_mem[_i]); \
            return; \
        } \
    } \
} while (0)

#define ASSERT_STR_EQ(expected, actual) do { \
    s_test_suite_state.assertions_checked++; \
    const char* _exp_str = (const char*)(expected); \
    const char* _act_str = (const char*)(actual); \
    if (_exp_str == NULL || _act_str == NULL || strcmp(_exp_str, _act_str) != 0) { \
        s_test_suite_state.current_test_failed = true; \
        printf("    %s[ASSERT FAILED]%s %s:%d: ASSERT_STR_EQ(%s, %s)\n" \
               "      Expected: \"%s\"\n" \
               "      Actual:   \"%s\"\n", \
               RCRE_CLR_RED, RCRE_CLR_RESET, __FILE__, __LINE__, #expected, #actual, \
               _exp_str ? _exp_str : "(null)", _act_str ? _act_str : "(null)"); \
        return; \
    } \
} while (0)

#endif /* RCRE_TEST_FRAMEWORK_H */
