/**
 * @file vis_ds.c
 * @brief RCRE Data Science Terminal Visualizer Implementation (ISO C23)
 */

#include "rcre_vis_ds.h"
#include "rcre_stat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void rcre_vis_terminal_heatmap(
    const RCRE_Tensor* corr_matrix,
    const char* const* col_names,
    size_t p
) {
    if (!corr_matrix || !corr_matrix->data || corr_matrix->ndim != 2 || p == 0) {
        return;
    }

    const float* mat = (const float*)corr_matrix->data;

    printf("\n📊 Correlation Matrix Heatmap [%zux%zu]:\n", p, p);

    /* Header */
    printf("        ");
    for (size_t j = 0; j < p; ++j) {
        char name[8];
        if (col_names && col_names[j]) {
            snprintf(name, sizeof(name), "%-7s", col_names[j]);
        } else {
            snprintf(name, sizeof(name), "x%-6zu", j);
        }
        printf("%s ", name);
    }
    printf("\n");

    for (size_t i = 0; i < p; ++i) {
        char row_name[8];
        if (col_names && col_names[i]) {
            snprintf(row_name, sizeof(row_name), "%-7s", col_names[i]);
        } else {
            snprintf(row_name, sizeof(row_name), "x%-6zu", i);
        }
        printf("%s ", row_name);

        for (size_t j = 0; j < p; ++j) {
            float r = mat[i * p + j];

            /* Choose ANSI 256 background color based on r in [-1, 1] */
            int color_code = 240; /* Gray for zero */
            if (r > 0.05f) {
                int intensity = (int)(r * 5.0f);
                if (intensity > 5) intensity = 5;
                /* Blues/Cyans: 24 to 39 */
                color_code = 24 + intensity * 3;
            } else if (r < -0.05f) {
                int intensity = (int)(-r * 5.0f);
                if (intensity > 5) intensity = 5;
                /* Reds/Oranges: 160 to 196 */
                color_code = 160 + intensity * 6;
            }

            printf("\033[48;5;%dm%6.2f \033[0m", color_code, (double)r);
        }
        printf("\n");
    }
    printf("\n");
}

void rcre_vis_corr_heatmap(
    const RCRE_Tensor* corr_matrix,
    const char* const* col_names
) {
    if (!corr_matrix || corr_matrix->ndim != 2) return;
    rcre_vis_terminal_heatmap(corr_matrix, col_names, corr_matrix->shape[0]);
}

void rcre_vis_terminal_hist(
    const float* x,
    size_t n,
    size_t num_bins,
    const char* title
) {
    if (!x || n == 0 || num_bins == 0) return;
    if (num_bins > 32) num_bins = 32;

    float min_v = x[0], max_v = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] < min_v) min_v = x[i];
        if (x[i] > max_v) max_v = x[i];
    }

    float range = max_v - min_v;
    if (range <= 1e-12f) range = 1.0f;

    size_t counts[32] = {0};
    for (size_t i = 0; i < n; ++i) {
        size_t b = (size_t)(((x[i] - min_v) / range) * (float)num_bins);
        if (b >= num_bins) b = num_bins - 1;
        counts[b]++;
    }

    size_t max_count = 1;
    for (size_t b = 0; b < num_bins; ++b) {
        if (counts[b] > max_count) max_count = counts[b];
    }

    printf("\n📈 Distribution Histogram: %s [N=%zu, Range: %.2f .. %.2f]\n",
           title ? title : "Data", n, (double)min_v, (double)max_v);

    const int max_bar_len = 36;
    for (size_t b = 0; b < num_bins; ++b) {
        float bin_lo = min_v + (float)b * (range / (float)num_bins);
        float bin_hi = bin_lo + (range / (float)num_bins);

        int bar_len = (int)(((double)counts[b] / (double)max_count) * (double)max_bar_len);

        printf("[%6.2f..%6.2f] | ", (double)bin_lo, (double)bin_hi);
        for (int k = 0; k < bar_len; ++k) {
            printf("█");
        }
        printf(" (%zu)\n", counts[b]);
    }
    printf("\n");
}

void rcre_vis_histogram(
    const float* x,
    size_t n,
    size_t num_bins,
    const char* title
) {
    rcre_vis_terminal_hist(x, n, num_bins, title);
}

void rcre_vis_terminal_boxplot(
    const RCRE_StatSummary* stats,
    const char* label
) {
    if (!stats) return;

    printf("\n📦 Boxplot: %s [Min: %.2f | Q25: %.2f | Median: %.2f | Q75: %.2f | Max: %.2f]\n",
           label ? label : "Feature",
           (double)stats->min, (double)stats->q25,
           (double)stats->median, (double)stats->q75,
           (double)stats->max);

    float total_range = stats->max - stats->min;
    if (total_range <= 1e-12f) total_range = 1.0f;

    const int width = 50;
    char line[52];
    memset(line, ' ', width);
    line[width] = '\0';

    int pos_min = 0;
    int pos_q25 = (int)(((stats->q25 - stats->min) / total_range) * (float)(width - 1));
    int pos_med = (int)(((stats->median - stats->min) / total_range) * (float)(width - 1));
    int pos_q75 = (int)(((stats->q75 - stats->min) / total_range) * (float)(width - 1));
    int pos_max = width - 1;

    /* Draw whisker 1 */
    for (int i = pos_min; i < pos_q25; ++i) line[i] = '-';
    line[pos_min] = '|';

    /* Draw box */
    line[pos_q25] = '[';
    for (int i = pos_q25 + 1; i < pos_q75; ++i) line[i] = '=';
    line[pos_med] = '|';
    line[pos_q75] = ']';

    /* Draw whisker 2 */
    for (int i = pos_q75 + 1; i <= pos_max; ++i) line[i] = '-';
    line[pos_max] = '|';

    printf("  %s\n\n", line);
}

void rcre_vis_boxplot(
    const float* x,
    size_t n,
    const char* label,
    RCRE_Arena* scratch
) {
    if (!x || n == 0 || !scratch) return;
    RCRE_StatSummary stats = rcre_stat_describe(x, n, scratch);
    rcre_vis_terminal_boxplot(&stats, label);
}
