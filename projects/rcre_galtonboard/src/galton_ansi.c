/**
 * @file galton_ansi.c
 * @brief Live ANSI terminal visualizer for RCRE Galton Board.
 */

#define _POSIX_C_SOURCE 200809L

#include "rcre_galton.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

static volatile sig_atomic_t g_stop_animation = 0;

static void handle_sigint(int sig) {
    (void)sig;
    g_stop_animation = 1;
}

static void sleep_ms(uint32_t ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/**
 * @brief Render a single visual frame to the terminal.
 */
static void render_frame(const galton_board_t* board,
                         const galton_stats_t* stats,
                         uint32_t active_row,
                         uint32_t active_col,
                         bool ball_active) {
    const uint32_t rows = board->config.rows;
    const uint32_t bins = board->num_bins;

    /* Move cursor to home and clear screen */
    printf("\x1b[H\x1b[2J");

    /* Header */
    printf("\x1b[1;36m========================================================================================\x1b[0m\n");
    printf("\x1b[1;37m        RCRE Galton Board (Quincunx) — Live C23 Flow & Central Limit Theorem           \x1b[0m\n");
    printf("\x1b[1;36m========================================================================================\x1b[0m\n\n");

    /* 1. Pin Pyramid and Falling Ball */
    int term_width = 80;
    int center_x = term_width / 2;

    /* Draw funnel top */
    int funnel_indent = center_x - 3;
    printf("%*s\x1b[38;5;244m\\   /\x1b[0m\n", funnel_indent, "");
    printf("%*s\x1b[38;5;244m \\v/ \x1b[0m\n", funnel_indent, "");

    /* Draw Pin Pyramid */
    for (uint32_t r = 0; r < rows; ++r) {
        int row_pegs = r + 1;
        int row_width = (row_pegs - 1) * 4;
        int start_x = center_x - (row_width / 2);
        if (start_x < 2) start_x = 2;

        printf("%*s", start_x, "");
        for (int c = 0; c < row_pegs; ++c) {
            if (ball_active && r == active_row && (uint32_t)c == active_col) {
                /* Active falling marble: bright neon amber */
                printf("\x1b[1;38;5;220m(●)\x1b[0m ");
            } else {
                /* Static peg: cyan/steel blue */
                printf("\x1b[38;5;75m • \x1b[0m ");
            }
        }
        printf("\n");
    }

    /* 2. Bin Dividers */
    int bins_width = (bins - 1) * 4;
    int bins_start_x = center_x - (bins_width / 2);
    if (bins_start_x < 2) bins_start_x = 2;

    printf("\n%*s\x1b[38;5;240m", bins_start_x, "");
    for (uint32_t b = 0; b < bins; ++b) {
        printf("|   ");
    }
    printf("|\x1b[0m\n");

    /* 3. Real-time Histogram Bars (Unicode vertical blocks) */
    uint64_t max_count = 1;
    for (uint32_t b = 0; b < bins; ++b) {
        if (board->bin_counts[b] > max_count) {
            max_count = board->bin_counts[b];
        }
    }

    constexpr int HIST_HEIGHT = 8;
    static const char* const blocks[] = {" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

    for (int h = HIST_HEIGHT; h >= 1; --h) {
        printf("%*s", bins_start_x, "");
        for (uint32_t b = 0; b < bins; ++b) {
            double fraction = (double)board->bin_counts[b] / (double)max_count;
            int level = (int)(fraction * HIST_HEIGHT);
            if (level >= h) {
                printf("\x1b[1;38;5;48m █  \x1b[0m");
            } else if (level == h - 1) {
                int sub = (int)((fraction * HIST_HEIGHT - (h - 1)) * 8.0);
                if (sub < 0) sub = 0;
                if (sub > 8) sub = 8;
                printf("\x1b[38;5;48m %s  \x1b[0m", blocks[sub]);
            } else {
                printf("    ");
            }
        }
        printf("\n");
    }

    /* Bin indices */
    printf("%*s\x1b[38;5;245m", bins_start_x, "");
    for (uint32_t b = 0; b < bins; ++b) {
        printf("%3u ", b);
    }
    printf("\x1b[0m\n\n");

    /* 4. Telemetry and Statistical Parity HUD */
    printf("\x1b[1;33m--- STATISTICAL TELEMETRY & CLT CONVERGENCE ---\x1b[0m\n");
    printf("  Total Balls Dropped : \x1b[1;32m%llu\x1b[0m\n", (unsigned long long)stats->total_balls);
    printf("  Lattice Rows (N)    : %u    |  Right Bias (p) : %.3f\n", stats->rows, stats->bias);
    printf("  Empirical Mean (x̄)  : \x1b[1;37m%.4f\x1b[0m (Target: %.4f, Δ: %+.4f)\n",
           stats->mean, stats->theoretical_mean, stats->mean - stats->theoretical_mean);
    printf("  Empirical StdDev (s): \x1b[1;37m%.4f\x1b[0m (Target: %.4f, Δ: %+.4f)\n",
           stats->std_dev, stats->theoretical_std_dev, stats->std_dev - stats->theoretical_std_dev);
    printf("  Sample Skewness     : %+.4f (Target: %+.4f)\n", stats->skewness, stats->theoretical_skewness);
    printf("  Excess Kurtosis     : %+.4f (Target: %+.4f)\n", stats->kurtosis, stats->theoretical_kurtosis);
    printf("  Chi-Square (χ²)     : \x1b[1;35m%.3f\x1b[0m (dof: %u, p-value: \x1b[1;36m%.4f\x1b[0m)\n",
           stats->chi_square_stat, stats->degrees_of_freedom, stats->chi_square_p_value);
    printf("\x1b[38;5;244mPress Ctrl+C to halt animation and return to prompt.\x1b[0m\n");
    fflush(stdout);
}

void galton_run_ansi_animation(galton_board_t* board, uint64_t n_balls, uint32_t delay_ms) {
    if (!board) return;

    /* Install SIGINT handler */
    g_stop_animation = 0;
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    /* Hide terminal cursor */
    printf("\x1b[?25l");
    fflush(stdout);

    const uint32_t rows = board->config.rows;
    galton_stats_t stats;

    for (uint64_t i = 0; i < n_balls && !g_stop_animation; ++i) {
        uint32_t path[RCRE_GALTON_MAX_ROWS] = {0};
        uint32_t landed_bin = galton_drop_single(board, path);
        (void)landed_bin;

        /* Trace the path down row by row if delay is perceptible */
        if (delay_ms >= 10 && rows <= 20) {
            uint32_t col = 0;
            for (uint32_t r = 0; r < rows && !g_stop_animation; ++r) {
                if (path[r] == 1) {
                    col++;
                }
                galton_calculate_stats(board, &stats);
                render_frame(board, &stats, r, col, true);
                sleep_ms(delay_ms / rows > 2 ? delay_ms / rows : 2);
            }
        }

        galton_calculate_stats(board, &stats);
        render_frame(board, &stats, rows, 0, false);

        if (delay_ms > 0) {
            sleep_ms(delay_ms);
        }
    }

    /* Restore terminal cursor */
    printf("\x1b[?25h\n");
    fflush(stdout);
}
