/**
 * @file vis.c
 * @brief Terminal ANSI plots, ASCII confusion matrices, SVG chart exporters,
 *        and 3D point cloud HTML exports.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void rcre_vis_loss_history(const float* loss_history, size_t count, const char* title) {
    if (loss_history == nullptr || count == 0) return;

    printf("\n\033[1;36m=== 📈 %s ===\033[0m\n", title ? title : "Training Loss Progression");

    float min_val = loss_history[0];
    float max_val = loss_history[0];
    for (size_t i = 1; i < count; ++i) {
        if (loss_history[i] < min_val) min_val = loss_history[i];
        if (loss_history[i] > max_val) max_val = loss_history[i];
    }

    const size_t height = 10;
    const size_t width = (count < 60) ? count : 60;

    for (int row = (int)height; row >= 0; --row) {
        float threshold = min_val + ((float)row / (float)height) * (max_val - min_val);
        printf("%8.4f | ", threshold);

        for (size_t col = 0; col < width; ++col) {
            size_t idx = (col * count) / width;
            float val = loss_history[idx];
            if (fabsf(val - threshold) <= (max_val - min_val) / (float)(height * 2)) {
                printf("\033[1;32m*\033[0m");
            } else if (val >= threshold) {
                printf("\033[34m.\033[0m");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    printf("         +");
    for (size_t c = 0; c < width; ++c) printf("-");
    printf("\n          Start: %.4f | End: %.4f | Min: %.4f | Max: %.4f\n\n",
           loss_history[0], loss_history[count - 1], min_val, max_val);
}

void rcre_vis_confusion_matrix(const size_t* confusion_matrix, size_t num_classes, const char* const* class_names) {
    if (confusion_matrix == nullptr || num_classes == 0) return;

    printf("\n\033[1;33m=== 📊 Confusion Matrix & Classification Metrics ===\033[0m\n\n");
    printf("%15s", "Predicted ->");
    for (size_t c = 0; c < num_classes; ++c) {
        if (class_names && class_names[c]) {
            printf("%10s", class_names[c]);
        } else {
            printf("%7s %zu", "Class", c);
        }
    }
    printf("%12s%12s\n", "Recall", "Total");

    size_t total_samples = 0;
    size_t correct_samples = 0;

    for (size_t r = 0; r < num_classes; ++r) {
        if (class_names && class_names[r]) {
            printf("%15s", class_names[r]);
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "Class %zu", r);
            printf("%15s", buf);
        }

        size_t row_total = 0;
        for (size_t c = 0; c < num_classes; ++c) {
            size_t val = confusion_matrix[r * num_classes + c];
            row_total += val;
            total_samples += val;
            if (r == c) correct_samples += val;
            if (r == c) {
                printf("\033[1;32m%10zu\033[0m", val);
            } else {
                printf("%10zu", val);
            }
        }
        size_t tp = confusion_matrix[r * num_classes + r];
        float recall = row_total > 0 ? (float)tp / (float)row_total : 0.0f;
        printf("%11.2f%%%12zu\n", recall * 100.0f, row_total);
    }

    printf("\n\033[1;32mOverall Accuracy: %.2f%% (%zu / %zu)\033[0m\n\n",
           total_samples > 0 ? (float)correct_samples / (float)total_samples * 100.0f : 0.0f,
           correct_samples, total_samples);
}

bool rcre_vis_export_svg_curve(const float* y_vals, size_t count, const char* filepath, const char* title) {
    if (y_vals == nullptr || count == 0 || filepath == nullptr) return false;

    FILE* f = fopen(filepath, "w");
    if (!f) return false;

    float min_val = y_vals[0];
    float max_val = y_vals[0];
    for (size_t i = 1; i < count; ++i) {
        if (y_vals[i] < min_val) min_val = y_vals[i];
        if (y_vals[i] > max_val) max_val = y_vals[i];
    }
    if (max_val - min_val < 1e-6f) max_val += 1.0f;

    const int width = 800;
    const int height = 400;
    const int pad_x = 60;
    const int pad_y = 50;

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\" width=\"%d\" height=\"%d\" style=\"background:#121820; font-family:sans-serif;\">\n", width, height, width, height);
    fprintf(f, "  <text x=\"%d\" y=\"30\" fill=\"#E2E8F0\" font-size=\"18\" font-weight=\"bold\" text-anchor=\"middle\">%s</text>\n", width / 2, title ? title : "Loss Curve");

    /* Axes */
    fprintf(f, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#4A5568\" stroke-width=\"1.5\" />\n", pad_x, height - pad_y, width - pad_x, height - pad_y);
    fprintf(f, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#4A5568\" stroke-width=\"1.5\" />\n", pad_x, pad_y, pad_x, height - pad_y);

    /* Polyline coordinates */
    fprintf(f, "  <polyline fill=\"none\" stroke=\"#38BDF8\" stroke-width=\"2.5\" points=\"");
    for (size_t i = 0; i < count; ++i) {
        float norm_x = (float)i / (float)(count - 1);
        float norm_y = (y_vals[i] - min_val) / (max_val - min_val);
        int px = pad_x + (int)(norm_x * (float)(width - 2 * pad_x));
        int py = (height - pad_y) - (int)(norm_y * (float)(height - 2 * pad_y));
        fprintf(f, "%d,%d ", px, py);
    }
    fprintf(f, "\" />\n");

    /* Min and max labels */
    fprintf(f, "  <text x=\"%d\" y=\"%d\" fill=\"#94A3B8\" font-size=\"12\" text-anchor=\"end\">%.4f</text>\n", pad_x - 10, pad_y + 10, max_val);
    fprintf(f, "  <text x=\"%d\" y=\"%d\" fill=\"#94A3B8\" font-size=\"12\" text-anchor=\"end\">%.4f</text>\n", pad_x - 10, height - pad_y, min_val);

    fprintf(f, "</svg>\n");
    fclose(f);
    return true;
}

bool rcre_vis_export_3d_clusters_html(const RCRE_AGTensor* points, const size_t* assignments, const RCRE_AGTensor* centroids, size_t k, const char* filepath) {
    if (points == nullptr || centroids == nullptr || filepath == nullptr) return false;
    if (points->shape[1] < 3 || centroids->shape[1] < 3) return false;

    FILE* f = fopen(filepath, "w");
    if (!f) return false;

    fprintf(f, "<!DOCTYPE html>\n<html>\n<head>\n");
    fprintf(f, "<title>RCRE - 3D K-Means Clustering Geometry</title>\n");
    fprintf(f, "<script src=\"https://cdn.plot.ly/plotly-2.29.1.min.js\"></script>\n");
    fprintf(f, "<style>body { margin: 0; background: #0F172A; color: #F8FAFC; font-family: -apple-system, sans-serif; }</style>\n");
    fprintf(f, "</head>\n<body>\n");
    fprintf(f, "<div id=\"plot\" style=\"width:100vw; height:100vh;\"></div>\n");
    fprintf(f, "<script>\n");

    const char* colors[8] = { "#38BDF8", "#F43F5E", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#6366F1" };

    fprintf(f, "const data = [\n");

    size_t n = points->shape[0];
    for (size_t cluster_id = 0; cluster_id < k; ++cluster_id) {
        fprintf(f, "  {\n");
        fprintf(f, "    x: [");
        for (size_t i = 0; i < n; ++i) {
            if (assignments[i] == cluster_id) fprintf(f, "%.4f, ", points->data[i * points->shape[1] + 0]);
        }
        fprintf(f, "],\n    y: [");
        for (size_t i = 0; i < n; ++i) {
            if (assignments[i] == cluster_id) fprintf(f, "%.4f, ", points->data[i * points->shape[1] + 1]);
        }
        fprintf(f, "],\n    z: [");
        for (size_t i = 0; i < n; ++i) {
            if (assignments[i] == cluster_id) fprintf(f, "%.4f, ", points->data[i * points->shape[1] + 2]);
        }
        fprintf(f, "],\n    mode: 'markers',\n");
        fprintf(f, "    name: 'Cluster %zu',\n", cluster_id);
        fprintf(f, "    marker: { size: 3, color: '%s', opacity: 0.8 },\n", colors[cluster_id % 8]);
        fprintf(f, "    type: 'scatter3d'\n  },\n");
    }

    fprintf(f, "  {\n");
    fprintf(f, "    x: [");
    for (size_t c = 0; c < k; ++c) fprintf(f, "%.4f, ", centroids->data[c * centroids->shape[1] + 0]);
    fprintf(f, "],\n    y: [");
    for (size_t c = 0; c < k; ++c) fprintf(f, "%.4f, ", centroids->data[c * centroids->shape[1] + 1]);
    fprintf(f, "],\n    z: [");
    for (size_t c = 0; c < k; ++c) fprintf(f, "%.4f, ", centroids->data[c * centroids->shape[1] + 2]);
    fprintf(f, "],\n    mode: 'markers',\n");
    fprintf(f, "    name: 'Centroids',\n");
    fprintf(f, "    marker: { size: 9, symbol: 'diamond', color: '#FFFFFF', line: { color: '#000000', width: 2 } },\n");
    fprintf(f, "    type: 'scatter3d'\n  }\n");

    fprintf(f, "];\n");
    fprintf(f, "const layout = {\n");
    fprintf(f, "  title: 'RCRE: 3D K-Means Cluster Geometry',\n");
    fprintf(f, "  paper_bgcolor: '#0F172A',\n");
    fprintf(f, "  font: { color: '#F8FAFC' },\n");
    fprintf(f, "  scene: {\n");
    fprintf(f, "    xaxis: { title: 'X1', backgroundcolor: '#1E293B', gridcolor: '#334155' },\n");
    fprintf(f, "    yaxis: { title: 'X2', backgroundcolor: '#1E293B', gridcolor: '#334155' },\n");
    fprintf(f, "    zaxis: { title: 'X3', backgroundcolor: '#1E293B', gridcolor: '#334155' }\n");
    fprintf(f, "  }\n};\n");
    fprintf(f, "Plotly.newPlot('plot', data, layout);\n");
    fprintf(f, "</script>\n</body>\n</html>\n");

    fclose(f);
    return true;
}
