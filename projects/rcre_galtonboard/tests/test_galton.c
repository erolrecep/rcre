/**
 * @file test_galton.c
 * @brief Comprehensive automated unit and statistical parity tests for RCRE Galton Board.
 */

#define _POSIX_C_SOURCE 200809L

#include "rcre_galton.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "\x1b[1;31m[FAILED]\x1b[0m %s:%d: %s\n", __FILE__, __LINE__, msg); \
            exit(1); \
        } \
    } while(0)

#define TEST_PASS(name) \
    printf("\x1b[1;32m[PASSED]\x1b[0m %s\n", name)

static void test_prng_determinism(void) {
    galton_config_t cfg = {.rows = 12, .bias = 0.5, .seed = 12345ULL, .record_paths = false};
    galton_board_t* b1 = galton_board_create(&cfg);
    galton_board_t* b2 = galton_board_create(&cfg);

    TEST_ASSERT(b1 != nullptr, "b1 creation failed");
    TEST_ASSERT(b2 != nullptr, "b2 creation failed");

    for (int i = 0; i < 500; ++i) {
        uint32_t bin1 = galton_drop_single(b1, nullptr);
        uint32_t bin2 = galton_drop_single(b2, nullptr);
        TEST_ASSERT(bin1 == bin2, "PRNG determinism violation across identical seeds");
    }

    galton_board_destroy(b1);
    galton_board_destroy(b2);
    TEST_PASS("test_prng_determinism");
}

static void test_binomial_pmf_exact(void) {
    /* For n=4, k=2, p=0.5: (4 choose 2) * 0.5^4 = 6/16 = 0.375 */
    double p4_2 = galton_binomial_pmf(4, 2, 0.5);
    TEST_ASSERT(fabs(p4_2 - 0.375) < 1e-9, "Binomial PMF (4,2,0.5) mismatch");

    /* For n=10, sum of PMF from k=0 to 10 must equal 1.0 */
    double sum = 0.0;
    for (uint32_t k = 0; k <= 10; ++k) {
        sum += galton_binomial_pmf(10, k, 0.4);
    }
    TEST_ASSERT(fabs(sum - 1.0) < 1e-9, "Binomial PMF total probability != 1.0");

    TEST_PASS("test_binomial_pmf_exact");
}

static void test_clt_convergence_symmetric(void) {
    /* N=16, p=0.5: mu = 8.0, sigma^2 = 4.0, sigma = 2.0 */
    galton_config_t cfg = {.rows = 16, .bias = 0.5, .seed = 9999ULL, .record_paths = false};
    galton_board_t* b = galton_board_create(&cfg);

    uint64_t n_balls = 200000ULL;
    galton_drop_batch(b, n_balls);

    galton_stats_t stats;
    galton_calculate_stats(b, &stats);

    /* Standard Error of Mean: sigma / sqrt(M) = 2.0 / sqrt(200000) ~ 0.00447 */
    /* Check 3.5-sigma confidence interval */
    double mean_error = fabs(stats.mean - stats.theoretical_mean);
    TEST_ASSERT(mean_error < 0.02, "Empirical mean diverged significantly from Central Limit Theorem expectation");

    double var_error = fabs(stats.variance - stats.theoretical_variance);
    TEST_ASSERT(var_error < 0.05, "Empirical variance diverged from CLT expectation");

    /* Symmetric board must have skewness near zero (|skewness| < 0.03) */
    TEST_ASSERT(fabs(stats.skewness) < 0.03, "Empirical skewness for symmetric board should be close to 0");

    /* Chi-square p-value should not reject null hypothesis with severe extremity */
    TEST_ASSERT(stats.chi_square_p_value > 0.001, "Chi-Square goodness of fit failed with extreme p-value");

    galton_board_destroy(b);
    TEST_PASS("test_clt_convergence_symmetric");
}

static void test_skewed_board(void) {
    /* N=12, p=0.25: mu = 3.0, theoretical skewness = (1 - 2*0.25) / sqrt(12 * 0.25 * 0.75) = 0.5 / 1.5 = +0.3333 */
    galton_config_t cfg = {.rows = 12, .start_pins = 1, .bias = 0.25, .seed = 777ULL, .record_paths = false};
    galton_board_t* b = galton_board_create(&cfg);

    uint64_t n_balls = 200000ULL;
    galton_drop_batch(b, n_balls);

    galton_stats_t stats;
    galton_calculate_stats(b, &stats);

    TEST_ASSERT(fabs(stats.mean - 3.0) < 0.03, "Skewed mean diverged from 3.0");
    TEST_ASSERT(stats.skewness > 0.20 && stats.skewness < 0.45, "Empirical skewness failed to exhibit expected positive asymmetry");

    galton_board_destroy(b);
    TEST_PASS("test_skewed_board");
}

static void test_start_pins_mixture(void) {
    /* Test small N=3, start_pins=1: 4 bins, exact binomial probabilities */
    galton_config_t cfg_small = {.rows = 3, .start_pins = 1, .bias = 0.5, .seed = 101ULL, .record_paths = false};
    galton_board_t* b_small = galton_board_create(&cfg_small);
    TEST_ASSERT(b_small->num_bins == 4, "Small N=3 board must have 4 bins");
    galton_drop_batch(b_small, 100000ULL);
    galton_stats_t stats_small;
    galton_calculate_stats(b_small, &stats_small);
    TEST_ASSERT(fabs(stats_small.mean - 1.5) < 0.02, "N=3 mean must be close to 1.5");
    galton_board_destroy(b_small);

    /* Test N=12, start_pins=4: target mean = (4-1)/2 + 12*0.5 = 7.5, var = (16-1)/12 + 12*0.25 = 4.25 */
    galton_config_t cfg_mix = {.rows = 12, .start_pins = 4, .bias = 0.5, .seed = 202ULL, .record_paths = false};
    galton_board_t* b_mix = galton_board_create(&cfg_mix);
    TEST_ASSERT(b_mix->num_bins == 16, "N=12 with 4 start pins must have 16 bins");
    galton_drop_batch(b_mix, 200000ULL);
    galton_stats_t stats_mix;
    galton_calculate_stats(b_mix, &stats_mix);
    TEST_ASSERT(fabs(stats_mix.mean - 7.5) < 0.03, "Multi-pin apex mean diverged from compound expectation 7.5");
    TEST_ASSERT(fabs(stats_mix.variance - 4.25) < 0.08, "Multi-pin apex variance diverged from compound expectation 4.25");
    galton_board_destroy(b_mix);

    TEST_PASS("test_start_pins_mixture");
}

static void test_csv_and_json_export(void) {
    galton_config_t cfg = {.rows = 8, .bias = 0.5, .seed = 42ULL, .record_paths = false};
    galton_board_t* b = galton_board_create(&cfg);
    galton_drop_batch(b, 1000);

    const char* csv_tmp = "/tmp/test_galton_export.csv";
    const char* json_tmp = "/tmp/test_galton_export.json";

    TEST_ASSERT(galton_export_csv(b, csv_tmp) == 0, "CSV export returned non-zero code");
    TEST_ASSERT(galton_export_json(b, json_tmp) == 0, "JSON export returned non-zero code");

    FILE* f_csv = fopen(csv_tmp, "r");
    TEST_ASSERT(f_csv != nullptr, "Could not open generated CSV file");
    fseek(f_csv, 0, SEEK_END);
    long csv_size = ftell(f_csv);
    fclose(f_csv);
    TEST_ASSERT(csv_size > 50, "CSV output file is suspiciously empty");

    FILE* f_json = fopen(json_tmp, "r");
    TEST_ASSERT(f_json != nullptr, "Could not open generated JSON file");
    fseek(f_json, 0, SEEK_END);
    long json_size = ftell(f_json);
    fclose(f_json);
    TEST_ASSERT(json_size > 100, "JSON output file is suspiciously empty");

    remove(csv_tmp);
    remove(json_tmp);
    galton_board_destroy(b);
    TEST_PASS("test_csv_and_json_export");
}

int main(void) {
    printf("\x1b[1;36m============================================================\x1b[0m\n");
    printf("\x1b[1;37m        Running RCRE Galton Board Test Suite (C23)          \x1b[0m\n");
    printf("\x1b[1;36m============================================================\x1b[0m\n");

    test_prng_determinism();
    test_binomial_pmf_exact();
    test_clt_convergence_symmetric();
    test_skewed_board();
    test_start_pins_mixture();
    test_csv_and_json_export();

    printf("\x1b[1;32m\nAll C23 tests passed successfully!\x1b[0m\n");
    return 0;
}
