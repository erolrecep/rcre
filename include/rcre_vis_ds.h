/**
 * @file rcre_vis_ds.h
 * @brief RCRE Data Science Terminal Visualizer (ISO C23)
 *
 * Implements ANSI 256-color correlation heatmaps, Unicode distribution histograms,
 * and horizontal boxplots directly in the shell.
 */

#ifndef RCRE_VIS_DS_H
#define RCRE_VIS_DS_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declare RCRE_StatSummary */
typedef struct RCRE_StatSummary RCRE_StatSummary;

/**
 * @brief Renders an ANSI 256-color correlation matrix heatmap to stdout.
 * @param corr_matrix P x P Pearson/Spearman correlation tensor.
 * @param col_names Optional array of column label strings [p].
 * @param p Feature count.
 */
void rcre_vis_terminal_heatmap(
    const RCRE_Tensor* corr_matrix,
    const char* const* col_names,
    size_t p
);

void rcre_vis_corr_heatmap(
    const RCRE_Tensor* corr_matrix,
    const char* const* col_names
);

/**
 * @brief Renders a Unicode horizontal histogram to stdout.
 * @param x Float data array [n].
 * @param n Element count.
 * @param num_bins Number of histogram bins (e.g. 10 to 20).
 * @param title Optional title string.
 */
void rcre_vis_terminal_hist(
    const float* x,
    size_t n,
    size_t num_bins,
    const char* title
);

void rcre_vis_histogram(
    const float* x,
    size_t n,
    size_t num_bins,
    const char* title
);

/**
 * @brief Renders a Unicode horizontal boxplot to stdout based on summary statistics.
 * @param stats Precomputed summary statistics (min, q25, median, q75, max).
 * @param label Variable name or description.
 */
void rcre_vis_terminal_boxplot(
    const RCRE_StatSummary* stats,
    const char* label
);

void rcre_vis_boxplot(
    const float* x,
    size_t n,
    const char* label,
    RCRE_Arena* scratch
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_VIS_DS_H */
