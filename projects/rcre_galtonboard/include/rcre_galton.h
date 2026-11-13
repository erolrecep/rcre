/**
 * @file rcre_galton.h
 * @brief RCRE Galton Board (Quincunx) Simulation Engine & Statistical Parity
 * @version 1.0.0
 * 
 * Strict ISO C23 compliant, zero external dependencies.
 * Implements high-throughput Monte-Carlo Bernoulli trials, deterministic 2D kinematics,
 * analytical Binomial & Gaussian distributions, and Chi-square goodness-of-fit testing.
 */

#ifndef RCRE_GALTON_H
#define RCRE_GALTON_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum supported rows in the pin lattice */
#define RCRE_GALTON_MAX_ROWS 64
#define RCRE_GALTON_DEFAULT_ROWS 12
#define RCRE_GALTON_DEFAULT_BIAS 0.5
#define RCRE_GALTON_DEFAULT_START_PINS 1
#define RCRE_GALTON_MAX_START_PINS 16

/**
 * @brief Configuration parameters for the Galton Board.
 */
typedef struct {
    uint32_t rows;           /**< Number of pin levels / rows (1 to RCRE_GALTON_MAX_ROWS) */
    uint32_t start_pins;     /**< Number of pins at top row (apex) (1 to RCRE_GALTON_MAX_START_PINS, default 1) */
    double bias;             /**< Probability p of deflecting right at each pin [0.0, 1.0] */
    uint64_t seed;           /**< PRNG seed for deterministic runs (0 = use time/hardware entropy) */
    bool record_paths;       /**< Whether to retain recent drop trajectories */
} galton_config_t;

/**
 * @brief Statistical metrics evaluated over accumulated ball distributions.
 */
typedef struct {
    uint64_t total_balls;             /**< Total number of balls dropped */
    uint32_t rows;                    /**< Number of pin levels N */
    uint32_t start_pins;              /**< Number of pins at apex level */
    uint32_t bins;                    /**< Number of collector bins (N + start_pins) */
    double bias;                      /**< Right-deflection probability p */

    /* Empirical sample moments */
    double mean;                      /**< Empirical mean: E[X] */
    double variance;                  /**< Empirical variance: Var(X) */
    double std_dev;                   /**< Empirical standard deviation: sqrt(Var(X)) */
    double skewness;                  /**< Sample skewness (standardized 3rd moment) */
    double kurtosis;                  /**< Sample excess kurtosis (standardized 4th moment - 3) */

    /* Theoretical moments (Binomial & CLT Gaussian target) */
    double theoretical_mean;          /**< Theoretical mean: N * p */
    double theoretical_variance;      /**< Theoretical variance: N * p * (1 - p) */
    double theoretical_std_dev;       /**< Theoretical std dev: sqrt(N * p * (1 - p)) */
    double theoretical_skewness;      /**< Theoretical skewness: (1 - 2p) / sqrt(N * p * (1 - p)) */
    double theoretical_kurtosis;      /**< Theoretical excess kurtosis: (1 - 6p(1-p)) / (N * p * (1 - p)) */

    /* Goodness-of-Fit statistical test */
    double chi_square_stat;           /**< Pearson chi-square statistic against Binomial PMF */
    double chi_square_p_value;        /**< Asymptotic p-value under null hypothesis */
    uint32_t degrees_of_freedom;      /**< Chi-square degrees of freedom */
} galton_stats_t;

/**
 * @brief xoshiro256** 64-bit PRNG state (Period: 2^256 - 1)
 */
typedef struct {
    uint64_t s[4];
} galton_prng_t;

/**
 * @brief Main Galton Board simulation instance.
 * Cache-line aligned (alignas(64)) to eliminate false sharing and maximize memory throughput.
 */
typedef struct {
    galton_config_t config;
    galton_prng_t prng;
    uint32_t num_bins;
    uint64_t total_dropped;

    /* Cache-line aligned histogram counters (heap allocated via arena or aligned_alloc) */
    alignas(64) uint64_t* bin_counts;

    /* Temporary buffer for theoretical probabilities */
    double* theoretical_pmf;
    double* gaussian_pdf;

    /* Optional path history buffer for visualization */
    uint32_t last_path[RCRE_GALTON_MAX_ROWS];
    uint32_t last_bin;
} galton_board_t;

/* ========================================================================= */
/* Core Lifecycle Functions                                                  */
/* ========================================================================= */

/**
 * @brief Allocate and initialize a Galton Board simulation instance.
 * @param config Pointer to configuration struct (NULL for defaults).
 * @return Pointer to initialized galton_board_t or NULL on allocation error.
 */
[[nodiscard]] galton_board_t* galton_board_create(const galton_config_t* config);

/**
 * @brief Destroy and free all memory associated with the Galton Board.
 * @param board Pointer to galton_board_t.
 */
void galton_board_destroy(galton_board_t* board);

/**
 * @brief Reset all accumulated bin counts and statistics while preserving configuration and seed.
 * @param board Pointer to galton_board_t.
 */
void galton_board_reset(galton_board_t* board);

/* ========================================================================= */
/* Simulation Execution                                                      */
/* ========================================================================= */

/**
 * @brief Simulate a single ball dropping through the pin grid.
 * @param board Pointer to board instance.
 * @param path_out Optional array of size at least config.rows to receive deflection sequence (0=left, 1=right).
 * @return The index of the bin the ball landed into [0, rows].
 */
uint32_t galton_drop_single(galton_board_t* board, uint32_t* path_out);

/**
 * @brief Drop a high-throughput batch of balls using optimized unrolled Bernoulli trials.
 * @param board Pointer to board instance.
 * @param n_balls Number of balls to simulate.
 */
void galton_drop_batch(galton_board_t* board, uint64_t n_balls);

/* ========================================================================= */
/* Statistical Analytics & Parity Routines                                  */
/* ========================================================================= */

/**
 * @brief Calculate complete statistical metrics and goodness-of-fit.
 * @param board Pointer to board instance.
 * @param stats_out Pointer to output stats struct.
 */
void galton_calculate_stats(galton_board_t* board, galton_stats_t* stats_out);

/**
 * @brief Calculate exact Binomial Probability Mass Function (PMF).
 * Uses high-precision log-gamma to avoid floating point overflow.
 * P(K = k) = (n choose k) * p^k * (1-p)^(n-k)
 */
[[nodiscard]] double galton_binomial_pmf(uint32_t n, uint32_t k, double p);

/**
 * @brief Calculate compound Mixture PMF for multi-pin apex lattices.
 * P(K = k) = (1 / start_pins) * sum_{j=0}^{start_pins-1} BinomialPMF(n, k - j, p)
 */
[[nodiscard]] double galton_mixture_pmf(uint32_t n, uint32_t start_pins, uint32_t k, double p);

/**
 * @brief Calculate Gaussian Normal Probability Density Function (PDF).
 * f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
 */
[[nodiscard]] double galton_gaussian_pdf(double x, double mu, double sigma);

/**
 * @brief Calculate Pearson Chi-square p-value using the regularized upper incomplete gamma function.
 */
[[nodiscard]] double galton_chi_square_p_value(double chi2, uint32_t dof);

/* ========================================================================= */
/* Data Export & Visual Terminal Rendering                                   */
/* ========================================================================= */

/**
 * @brief Export current bin histogram and comparison data to CSV.
 * @param board Pointer to board instance.
 * @param filepath Destination file path.
 * @return 0 on success, non-zero on I/O error.
 */
int galton_export_csv(const galton_board_t* board, const char* filepath);

/**
 * @brief Export full board configuration, bins, and metrics to JSON.
 * @param board Pointer to board instance.
 * @param filepath Destination file path.
 * @return 0 on success, non-zero on I/O error.
 */
int galton_export_json(const galton_board_t* board, const char* filepath);

/**
 * @brief Run an interactive real-time ANSI terminal animation of falling balls and live histogram.
 * @param board Pointer to board instance.
 * @param n_balls Total balls to animate.
 * @param delay_ms Milliseconds delay per ball or frame step.
 */
void galton_run_ansi_animation(galton_board_t* board, uint64_t n_balls, uint32_t delay_ms);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_GALTON_H */
