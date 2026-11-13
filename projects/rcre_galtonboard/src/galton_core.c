/**
 * @file galton_core.c
 * @brief High-performance ISO C23 Galton Board Engine implementation.
 */

#define _POSIX_C_SOURCE 200809L

#include "rcre_galton.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================= */
/* SplitMix64 & xoshiro256** High-Speed PRNG                                 */
/* ========================================================================= */

static inline uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static void prng_seed(galton_prng_t* prng, uint64_t seed) {
    if (seed == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ 0x517cc1b727220a95ULL;
    }
    uint64_t sm_state = seed;
    prng->s[0] = splitmix64(&sm_state);
    prng->s[1] = splitmix64(&sm_state);
    prng->s[2] = splitmix64(&sm_state);
    prng->s[3] = splitmix64(&sm_state);
}

static inline uint64_t prng_next_u64(galton_prng_t* prng) {
    const uint64_t result = rotl64(prng->s[1] * 5, 7) * 9;
    const uint64_t t = prng->s[1] << 17;

    prng->s[2] ^= prng->s[0];
    prng->s[3] ^= prng->s[1];
    prng->s[1] ^= prng->s[2];
    prng->s[0] ^= prng->s[3];

    prng->s[2] ^= t;
    prng->s[3] = rotl64(prng->s[3], 45);

    return result;
}

static inline double prng_next_double(galton_prng_t* prng) {
    return (prng_next_u64(prng) >> 11) * 0x1.0p-53;
}

/* ========================================================================= */
/* Mathematical & Statistical Utilities                                      */
/* ========================================================================= */

[[nodiscard]] double galton_binomial_pmf(uint32_t n, uint32_t k, double p) {
    if (k > n) return 0.0;
    if (p <= 0.0) return (k == 0) ? 1.0 : 0.0;
    if (p >= 1.0) return (k == n) ? 1.0 : 0.0;

    /* log(n choose k) using standard lgamma */
    double log_comb = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
    double log_prob = log_comb + (double)k * log(p) + (double)(n - k) * log(1.0 - p);
    return exp(log_prob);
}

[[nodiscard]] double galton_mixture_pmf(uint32_t n, uint32_t start_pins, uint32_t k, double p) {
    if (start_pins <= 1) return galton_binomial_pmf(n, k, p);
    double sum = 0.0;
    for (uint32_t j = 0; j < start_pins; ++j) {
        if (k >= j && (k - j) <= n) {
            sum += galton_binomial_pmf(n, k - j, p);
        }
    }
    return sum / (double)start_pins;
}

[[nodiscard]] double galton_gaussian_pdf(double x, double mu, double sigma) {
    if (sigma <= 0.0) return 0.0;
    double diff = (x - mu) / sigma;
    double inv_norm = 1.0 / (sigma * sqrt(2.0 * M_PI));
    return inv_norm * exp(-0.5 * diff * diff);
}

/**
 * @brief Regularized lower incomplete gamma function P(a, x) via series expansion.
 */
static double gammp_series(double a, double x) {
    constexpr int MAX_IT = 300;
    constexpr double EPS = 1e-12;

    double sum = 1.0 / a;
    double term = sum;
    for (int n = 1; n <= MAX_IT; ++n) {
        term *= x / (a + (double)n);
        sum += term;
        if (fabs(term) < fabs(sum) * EPS) {
            break;
        }
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

/**
 * @brief Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x) via continued fraction.
 */
static double gammq_cf(double a, double x) {
    constexpr int MAX_IT = 300;
    constexpr double EPS = 1e-12;
    constexpr double FPMIN = 1e-30;

    double b = x + 1.0 - a;
    double c = 1.0 / FPMIN;
    double d = 1.0 / b;
    double h = d;

    for (int i = 1; i <= MAX_IT; ++i) {
        double an = -(double)i * ((double)i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) <= EPS) {
            break;
        }
    }
    return exp(-x + a * log(x) - lgamma(a)) * h;
}

[[nodiscard]] double galton_chi_square_p_value(double chi2, uint32_t dof) {
    if (chi2 <= 0.0 || dof == 0) return 1.0;
    double a = 0.5 * (double)dof;
    double x = 0.5 * chi2;

    if (x < a + 1.0) {
        return 1.0 - gammp_series(a, x);
    } else {
        return gammq_cf(a, x);
    }
}

/* ========================================================================= */
/* Core Lifecycle Implementation                                             */
/* ========================================================================= */

[[nodiscard]] galton_board_t* galton_board_create(const galton_config_t* config) {
    galton_board_t* board = (galton_board_t*)calloc(1, sizeof(galton_board_t));
    if (!board) return nullptr;

    if (config) {
        board->config = *config;
    } else {
        board->config.rows = RCRE_GALTON_DEFAULT_ROWS;
        board->config.bias = RCRE_GALTON_DEFAULT_BIAS;
        board->config.seed = 0;
        board->config.record_paths = false;
    }

    if (board->config.rows < 1) board->config.rows = 1;
    if (board->config.rows > RCRE_GALTON_MAX_ROWS) board->config.rows = RCRE_GALTON_MAX_ROWS;
    if (board->config.start_pins < 1) board->config.start_pins = 1;
    if (board->config.start_pins > RCRE_GALTON_MAX_START_PINS) board->config.start_pins = RCRE_GALTON_MAX_START_PINS;
    if (board->config.bias < 0.0) board->config.bias = 0.0;
    if (board->config.bias > 1.0) board->config.bias = 1.0;

    board->num_bins = board->config.rows + board->config.start_pins;
    board->total_dropped = 0;

    /* Allocate aligned bin counts buffer (64-byte aligned) */
    size_t bin_bytes = (board->num_bins * sizeof(uint64_t) + 63) & ~63ULL;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bin_bytes) != 0 || !ptr) {
        free(board);
        return nullptr;
    }
    board->bin_counts = (uint64_t*)ptr;
    memset(board->bin_counts, 0, bin_bytes);

    /* Allocate probability lookup tables */
    board->theoretical_pmf = (double*)malloc(board->num_bins * sizeof(double));
    board->gaussian_pdf = (double*)malloc(board->num_bins * sizeof(double));
    if (!board->theoretical_pmf || !board->gaussian_pdf) {
        free(board->theoretical_pmf);
        free(board->gaussian_pdf);
        free(board->bin_counts);
        free(board);
        return nullptr;
    }

    /* Precompute theoretical distributions */
    uint32_t n = board->config.rows;
    uint32_t sp = board->config.start_pins;
    double p = board->config.bias;
    double mu = ((double)sp - 1.0) / 2.0 + (double)n * p;
    double var_u = ((double)(sp * sp) - 1.0) / 12.0;
    double sigma = sqrt(var_u + (double)n * p * (1.0 - p));
    if (sigma < 1e-6) sigma = 1e-6;

    for (uint32_t k = 0; k < board->num_bins; ++k) {
        board->theoretical_pmf[k] = galton_mixture_pmf(n, sp, k, p);
        board->gaussian_pdf[k] = galton_gaussian_pdf((double)k, mu, sigma);
    }

    prng_seed(&board->prng, board->config.seed);
    return board;
}

void galton_board_destroy(galton_board_t* board) {
    if (!board) return;
    if (board->bin_counts) free(board->bin_counts);
    if (board->theoretical_pmf) free(board->theoretical_pmf);
    if (board->gaussian_pdf) free(board->gaussian_pdf);
    free(board);
}

void galton_board_reset(galton_board_t* board) {
    if (!board) return;
    board->total_dropped = 0;
    size_t bin_bytes = board->num_bins * sizeof(uint64_t);
    memset(board->bin_counts, 0, bin_bytes);
    memset(board->last_path, 0, sizeof(board->last_path));
    board->last_bin = 0;
}

/* ========================================================================= */
/* High-Throughput Simulation Kernels                                        */
/* ========================================================================= */

uint32_t galton_drop_single(galton_board_t* board, uint32_t* path_out) {
    const uint32_t rows = board->config.rows;
    const uint32_t start_pins = board->config.start_pins;
    const double bias = board->config.bias;
    uint32_t right_bounces = 0;

    uint32_t init_offset = 0;
    if (start_pins > 1) {
        init_offset = (uint32_t)(prng_next_double(&board->prng) * (double)start_pins);
        if (init_offset >= start_pins) init_offset = start_pins - 1;
    }

    /* Symmetric fast-path (p == 0.5): single bit-level popcount */
    if (fabs(bias - 0.5) < 1e-9 && rows <= 64) {
        uint64_t word = prng_next_u64(&board->prng);
        uint64_t mask = (rows == 64) ? ~0ULL : ((1ULL << rows) - 1ULL);
        uint64_t active_bits = word & mask;
        right_bounces = (uint32_t)__builtin_popcountll(active_bits);

        if (path_out || board->config.record_paths) {
            for (uint32_t r = 0; r < rows; ++r) {
                uint32_t step = (uint32_t)((active_bits >> r) & 1ULL);
                if (path_out) path_out[r] = step;
                if (board->config.record_paths) board->last_path[r] = step;
            }
        }
    } else {
        /* Asymmetric general Bernoulli walk */
        for (uint32_t r = 0; r < rows; ++r) {
            double rnd = prng_next_double(&board->prng);
            uint32_t step = (rnd < bias) ? 1 : 0;
            right_bounces += step;
            if (path_out) path_out[r] = step;
            if (board->config.record_paths) board->last_path[r] = step;
        }
    }

    uint32_t final_bin = init_offset + right_bounces;
    board->bin_counts[final_bin]++;
    board->total_dropped++;
    board->last_bin = final_bin;
    return final_bin;
}

void galton_drop_batch(galton_board_t* board, uint64_t n_balls) {
    const uint32_t rows = board->config.rows;
    const uint32_t start_pins = board->config.start_pins;
    const double bias = board->config.bias;

    /* Fast path for symmetric board with 1 starting pin: unroll popcount by 4 */
    if (start_pins == 1 && fabs(bias - 0.5) < 1e-9 && rows <= 64) {
        const uint64_t mask = (rows == 64) ? ~0ULL : ((1ULL << rows) - 1ULL);
        uint64_t i = 0;
        const uint64_t unroll_limit = (n_balls >= 4) ? (n_balls - 3) : 0;

        for (; i < unroll_limit; i += 4) {
            uint64_t w0 = prng_next_u64(&board->prng) & mask;
            uint64_t w1 = prng_next_u64(&board->prng) & mask;
            uint64_t w2 = prng_next_u64(&board->prng) & mask;
            uint64_t w3 = prng_next_u64(&board->prng) & mask;

            board->bin_counts[__builtin_popcountll(w0)]++;
            board->bin_counts[__builtin_popcountll(w1)]++;
            board->bin_counts[__builtin_popcountll(w2)]++;
            board->bin_counts[__builtin_popcountll(w3)]++;
        }
        for (; i < n_balls; ++i) {
            uint64_t w = prng_next_u64(&board->prng) & mask;
            board->bin_counts[__builtin_popcountll(w)]++;
        }
    } else {
        /* General Bernoulli trials with arbitrary starting apex pins and bias */
        for (uint64_t i = 0; i < n_balls; ++i) {
            uint32_t init_pin = 0;
            if (start_pins > 1) {
                init_pin = (uint32_t)(prng_next_double(&board->prng) * (double)start_pins);
                if (init_pin >= start_pins) init_pin = start_pins - 1;
            }

            uint32_t k = 0;
            if (fabs(bias - 0.5) < 1e-9 && rows <= 64) {
                uint64_t mask = (rows == 64) ? ~0ULL : ((1ULL << rows) - 1ULL);
                k = (uint32_t)__builtin_popcountll(prng_next_u64(&board->prng) & mask);
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    if (prng_next_double(&board->prng) < bias) {
                        k++;
                    }
                }
            }
            board->bin_counts[init_pin + k]++;
        }
    }

    board->total_dropped += n_balls;
}

/* ========================================================================= */
/* Statistical Parity Analytics                                              */
/* ========================================================================= */

void galton_calculate_stats(galton_board_t* board, galton_stats_t* stats_out) {
    if (!board || !stats_out) return;

    const uint32_t n = board->config.rows;
    const uint32_t bins = board->num_bins;
    const double p = board->config.bias;
    const uint64_t total = board->total_dropped;

    stats_out->total_balls = total;
    stats_out->rows = n;
    stats_out->start_pins = board->config.start_pins;
    stats_out->bins = bins;
    stats_out->bias = p;

    /* Theoretical moments (compound mixture if start_pins > 1) */
    double sp = (double)board->config.start_pins;
    double var_u = (sp * sp - 1.0) / 12.0;
    stats_out->theoretical_mean = (sp - 1.0) / 2.0 + (double)n * p;
    stats_out->theoretical_variance = var_u + (double)n * p * (1.0 - p);
    stats_out->theoretical_std_dev = sqrt(stats_out->theoretical_variance);

    if (stats_out->theoretical_variance > 1e-12) {
        stats_out->theoretical_skewness = (1.0 - 2.0 * p) / stats_out->theoretical_std_dev;
        stats_out->theoretical_kurtosis = (1.0 - 6.0 * p * (1.0 - p)) / stats_out->theoretical_variance;
    } else {
        stats_out->theoretical_skewness = 0.0;
        stats_out->theoretical_kurtosis = 0.0;
    }

    if (total == 0) {
        stats_out->mean = 0.0;
        stats_out->variance = 0.0;
        stats_out->std_dev = 0.0;
        stats_out->skewness = 0.0;
        stats_out->kurtosis = 0.0;
        stats_out->chi_square_stat = 0.0;
        stats_out->chi_square_p_value = 1.0;
        stats_out->degrees_of_freedom = (bins > 1) ? (bins - 1) : 1;
        return;
    }

    /* Empirical first moment (Mean) */
    double sum = 0.0;
    for (uint32_t k = 0; k < bins; ++k) {
        sum += (double)k * (double)board->bin_counts[k];
    }
    double mean = sum / (double)total;
    stats_out->mean = mean;

    /* Empirical central moments */
    double sum_sq = 0.0;
    double sum_cube = 0.0;
    double sum_quad = 0.0;

    for (uint32_t k = 0; k < bins; ++k) {
        uint64_t count = board->bin_counts[k];
        if (count == 0) continue;
        double diff = (double)k - mean;
        double diff2 = diff * diff;
        sum_sq += diff2 * (double)count;
        sum_cube += diff2 * diff * (double)count;
        sum_quad += diff2 * diff2 * (double)count;
    }

    double sample_var = (total > 1) ? (sum_sq / (double)(total - 1)) : 0.0;
    double sample_std = sqrt(sample_var);
    stats_out->variance = sample_var;
    stats_out->std_dev = sample_std;

    if (sample_std > 1e-12) {
        stats_out->skewness = (sum_cube / (double)total) / (sample_std * sample_std * sample_std);
        stats_out->kurtosis = ((sum_quad / (double)total) / (sample_var * sample_var)) - 3.0;
    } else {
        stats_out->skewness = 0.0;
        stats_out->kurtosis = 0.0;
    }

    /* Pearson Chi-Square Goodness-of-Fit */
    double chi2 = 0.0;
    uint32_t active_bins = 0;

    for (uint32_t k = 0; k < bins; ++k) {
        double expected = (double)total * board->theoretical_pmf[k];
        if (expected >= 1.0) {  /* Bins with meaningful expected frequency */
            double observed = (double)board->bin_counts[k];
            double diff = observed - expected;
            chi2 += (diff * diff) / expected;
            active_bins++;
        }
    }

    uint32_t dof = (active_bins > 1) ? (active_bins - 1) : 1;
    stats_out->chi_square_stat = chi2;
    stats_out->degrees_of_freedom = dof;
    stats_out->chi_square_p_value = galton_chi_square_p_value(chi2, dof);
}

/* ========================================================================= */
/* Export Routines                                                           */
/* ========================================================================= */

int galton_export_csv(const galton_board_t* board, const char* filepath) {
    if (!board || !filepath) return -1;
    FILE* fp = fopen(filepath, "w");
    if (!fp) return -2;

    fprintf(fp, "bin_index,empirical_count,empirical_probability,theoretical_binomial_pmf,theoretical_gaussian_pdf\n");
    double total = (board->total_dropped > 0) ? (double)board->total_dropped : 1.0;

    for (uint32_t k = 0; k < board->num_bins; ++k) {
        double emp_prob = (double)board->bin_counts[k] / total;
        fprintf(fp, "%u,%llu,%.8f,%.8f,%.8f\n",
                k,
                (unsigned long long)board->bin_counts[k],
                emp_prob,
                board->theoretical_pmf[k],
                board->gaussian_pdf[k]);
    }

    fclose(fp);
    return 0;
}

int galton_export_json(const galton_board_t* board, const char* filepath) {
    if (!board || !filepath) return -1;
    FILE* fp = fopen(filepath, "w");
    if (!fp) return -2;

    galton_stats_t stats;
    galton_calculate_stats((galton_board_t*)board, &stats);

    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": \"1.0.0\",\n");
    fprintf(fp, "  \"configuration\": {\n");
    fprintf(fp, "    \"rows\": %u,\n", board->config.rows);
    fprintf(fp, "    \"bins\": %u,\n", board->num_bins);
    fprintf(fp, "    \"bias\": %.4f,\n", board->config.bias);
    fprintf(fp, "    \"seed\": %llu\n", (unsigned long long)board->config.seed);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"telemetry\": {\n");
    fprintf(fp, "    \"total_balls\": %llu,\n", (unsigned long long)stats.total_balls);
    fprintf(fp, "    \"mean\": %.6f,\n", stats.mean);
    fprintf(fp, "    \"variance\": %.6f,\n", stats.variance);
    fprintf(fp, "    \"std_dev\": %.6f,\n", stats.std_dev);
    fprintf(fp, "    \"skewness\": %.6f,\n", stats.skewness);
    fprintf(fp, "    \"kurtosis\": %.6f,\n", stats.kurtosis);
    fprintf(fp, "    \"theoretical_mean\": %.6f,\n", stats.theoretical_mean);
    fprintf(fp, "    \"theoretical_variance\": %.6f,\n", stats.theoretical_variance);
    fprintf(fp, "    \"theoretical_std_dev\": %.6f,\n", stats.theoretical_std_dev);
    fprintf(fp, "    \"chi_square_stat\": %.6f,\n", stats.chi_square_stat);
    fprintf(fp, "    \"chi_square_p_value\": %.6f,\n", stats.chi_square_p_value);
    fprintf(fp, "    \"degrees_of_freedom\": %u\n", stats.degrees_of_freedom);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"distribution\": [\n");

    double total = (board->total_dropped > 0) ? (double)board->total_dropped : 1.0;
    for (uint32_t k = 0; k < board->num_bins; ++k) {
        double emp_prob = (double)board->bin_counts[k] / total;
        fprintf(fp, "    {\"bin\": %u, \"count\": %llu, \"empirical_prob\": %.8f, \"binomial_pmf\": %.8f, \"gaussian_pdf\": %.8f}%s\n",
                k,
                (unsigned long long)board->bin_counts[k],
                emp_prob,
                board->theoretical_pmf[k],
                board->gaussian_pdf[k],
                (k + 1 < board->num_bins) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
    return 0;
}
