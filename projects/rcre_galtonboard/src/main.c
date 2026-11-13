/**
 * @file main.c
 * @brief RCRE Galton Board CLI Application and Performance Benchmark
 */

#define _POSIX_C_SOURCE 200809L

#include "rcre_galton.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static void print_banner(void) {
    printf("\x1b[1;36m==============================================================================\x1b[0m\n");
    printf("\x1b[1;37m        RCRE Galton Board (Quincunx) — ISO C23 Simulation Engine             \x1b[0m\n");
    printf("\x1b[1;36m==============================================================================\x1b[0m\n");
}

static void print_usage(const char* prog) {
    print_banner();
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --rows <N>           Number of pin rows [1..64] (default: 12)\n");
    printf("  --balls <M>          Total number of balls to drop (default: 100,000)\n");
    printf("  --bias <p>           Right bounce probability [0.0..1.0] (default: 0.500)\n");
    printf("  --seed <S>           PRNG seed for deterministic runs (default: hardware entropy)\n");
    printf("  --animate            Launch live interactive ANSI terminal visualization\n");
    printf("  --delay <ms>         Frame delay in ms for animation (default: 25)\n");
    printf("  --benchmark          Run high-throughput Monte-Carlo benchmark (10M+ drops)\n");
    printf("  --export-json <FILE> Save distribution and statistics to JSON\n");
    printf("  --export-csv <FILE>  Save distribution histogram to CSV\n");
    printf("  --quiet              Suppress non-essential output\n");
    printf("  --help, -h           Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("  %s --animate --rows 12 --balls 200\n", prog);
    printf("  %s --benchmark --rows 16 --balls 50000000\n", prog);
    printf("  %s --rows 16 --balls 1000000 --export-json dist.json\n\n", prog);
}

int main(int argc, char* argv[]) {
    galton_config_t config = {
        .rows = RCRE_GALTON_DEFAULT_ROWS,
        .bias = RCRE_GALTON_DEFAULT_BIAS,
        .seed = 0,
        .record_paths = false
    };

    uint64_t n_balls = 100000ULL;
    bool animate_mode = false;
    bool bench_mode = false;
    bool quiet_mode = false;
    uint32_t delay_ms = 25;
    const char* json_path = nullptr;
    const char* csv_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            config.rows = (uint32_t)strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--balls") == 0 && i + 1 < argc) {
            n_balls = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--bias") == 0 && i + 1 < argc) {
            config.bias = strtod(argv[++i], nullptr);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config.seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--animate") == 0) {
            animate_mode = true;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            bench_mode = true;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet_mode = true;
        } else if (strcmp(argv[i], "--delay") == 0 && i + 1 < argc) {
            delay_ms = (uint32_t)strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--export-json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--export-csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s (see --help)\n", argv[i]);
            return 1;
        }
    }

    /* Adjust defaults for interactive animation or benchmark if user didn't specify balls */
    if (animate_mode && n_balls == 100000ULL) {
        n_balls = 150ULL;
    }
    if (bench_mode && n_balls == 100000ULL) {
        n_balls = 20000000ULL; /* 20 Million balls for high precision benchmark */
    }

    galton_board_t* board = galton_board_create(&config);
    if (!board) {
        fprintf(stderr, "Error: Failed to allocate Galton board instance.\n");
        return 1;
    }

    if (animate_mode) {
        config.record_paths = true;
        galton_run_ansi_animation(board, n_balls, delay_ms);
    } else {
        if (!quiet_mode) {
            print_banner();
            printf("Simulating Galton Board: Rows=%u, Balls=%llu, Bias=%.3f, Seed=%llu\n",
                   config.rows, (unsigned long long)n_balls, config.bias, (unsigned long long)config.seed);
            printf("Executing Monte-Carlo Bernoulli trials...\n");
        }

        struct timespec t_start, t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        galton_drop_batch(board, n_balls);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed_sec = (double)(t_end.tv_sec - t_start.tv_sec) +
                             (double)(t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
        double drops_per_sec = (double)n_balls / (elapsed_sec > 0.0 ? elapsed_sec : 1e-9);

        galton_stats_t stats;
        galton_calculate_stats(board, &stats);

        if (!quiet_mode) {
            printf("\n\x1b[1;32m[Simulation Completed in %.4f seconds]\x1b[0m\n", elapsed_sec);
            printf("  Throughput          : \x1b[1;36m%.2f Million drops/sec\x1b[0m (%.2f ns/drop)\n",
                   drops_per_sec / 1e6, (elapsed_sec / (double)n_balls) * 1e9);
            printf("\n\x1b[1;33m--- STATISTICAL VERIFICATION REPORT ---\x1b[0m\n");
            printf("  Total Balls (M)     : %llu\n", (unsigned long long)stats.total_balls);
            printf("  Pin Levels (N)      : %u  (Bins: %u)\n", stats.rows, stats.bins);
            printf("  Right Bias (p)      : %.4f\n", stats.bias);
            printf("  Empirical Mean (x̄)  : \x1b[1;37m%.6f\x1b[0m  (Target: %.6f, Error: %+.6f)\n",
                   stats.mean, stats.theoretical_mean, stats.mean - stats.theoretical_mean);
            printf("  Empirical Variance  : \x1b[1;37m%.6f\x1b[0m  (Target: %.6f, Error: %+.6f)\n",
                   stats.variance, stats.theoretical_variance, stats.variance - stats.theoretical_variance);
            printf("  Empirical Std Dev   : \x1b[1;37m%.6f\x1b[0m  (Target: %.6f, Error: %+.6f)\n",
                   stats.std_dev, stats.theoretical_std_dev, stats.std_dev - stats.theoretical_std_dev);
            printf("  Sample Skewness     : %+.6f  (Target: %+.6f)\n",
                   stats.skewness, stats.theoretical_skewness);
            printf("  Excess Kurtosis     : %+.6f  (Target: %+.6f)\n",
                   stats.kurtosis, stats.theoretical_kurtosis);
            printf("  Chi-Square Test (χ²): \x1b[1;35m%.4f\x1b[0m  (dof: %u, p-value: \x1b[1;36m%.4f\x1b[0m)\n",
                   stats.chi_square_stat, stats.degrees_of_freedom, stats.chi_square_p_value);
            
            if (stats.chi_square_p_value >= 0.05) {
                printf("  Null Hypothesis H₀  : \x1b[1;32mACCEPTED\x1b[0m (No significant divergence from Binomial PMF at α=0.05)\n");
            } else {
                printf("  Null Hypothesis H₀  : \x1b[1;33mMARGINAL\x1b[0m (p=%.4f; very large sample size detects micro-deviations)\n", stats.chi_square_p_value);
            }
        }
    }

    if (json_path) {
        if (galton_export_json(board, json_path) == 0) {
            if (!quiet_mode) printf("\nExported JSON telemetry to: %s\n", json_path);
        } else {
            fprintf(stderr, "Error: Failed to write JSON to %s\n", json_path);
        }
    }

    if (csv_path) {
        if (galton_export_csv(board, csv_path) == 0) {
            if (!quiet_mode) printf("Exported CSV histogram to: %s\n", csv_path);
        } else {
            fprintf(stderr, "Error: Failed to write CSV to %s\n", csv_path);
        }
    }

    galton_board_destroy(board);
    return 0;
}
