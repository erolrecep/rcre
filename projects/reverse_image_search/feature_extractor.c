/**
 * @file feature_extractor.c
 * @brief RCRE 64-Dimensional Spatial Pyramid Feature Extractor Implementation
 */

#include "feature_extractor.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_NOT_REALLY
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

/**
 * @brief High-quality bilinear resampling of RGB image to canonical target resolution
 */
static void resample_bilinear_rgb(const uint8_t* src, size_t src_w, size_t src_h,
                                  uint8_t* dst, size_t dst_w, size_t dst_h) {
    if (src_w == dst_w && src_h == dst_h) {
        memcpy(dst, src, dst_w * dst_h * 3);
        return;
    }

    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    for (size_t dy = 0; dy < dst_h; ++dy) {
        float sy = ((float)dy + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(sy);
        int y1 = y0 + 1;
        float wy1 = sy - (float)y0;
        float wy0 = 1.0f - wy1;

        if (y0 < 0) { y0 = 0; y1 = 0; wy0 = 1.0f; wy1 = 0.0f; }
        if (y1 >= (int)src_h) { y1 = (int)src_h - 1; if (y0 >= (int)src_h) y0 = (int)src_h - 1; }

        const uint8_t* row0 = src + (size_t)y0 * src_w * 3;
        const uint8_t* row1 = src + (size_t)y1 * src_w * 3;
        uint8_t* dst_row = dst + dy * dst_w * 3;

        for (size_t dx = 0; dx < dst_w; ++dx) {
            float sx = ((float)dx + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(sx);
            int x1 = x0 + 1;
            float wx1 = sx - (float)x0;
            float wx0 = 1.0f - wx1;

            if (x0 < 0) { x0 = 0; x1 = 0; wx0 = 1.0f; wx1 = 0.0f; }
            if (x1 >= (int)src_w) { x1 = (int)src_w - 1; if (x0 >= (int)src_w) x0 = (int)src_w - 1; }

            for (size_t c = 0; c < 3; ++c) {
                float p00 = (float)row0[x0 * 3 + c];
                float p01 = (float)row0[x1 * 3 + c];
                float p10 = (float)row1[x0 * 3 + c];
                float p11 = (float)row1[x1 * 3 + c];

                float val = wy0 * (wx0 * p00 + wx1 * p01) + wy1 * (wx0 * p10 + wx1 * p11);
                if (val < 0.0f) val = 0.0f;
                if (val > 255.0f) val = 255.0f;
                dst_row[dx * 3 + c] = (uint8_t)(val + 0.5f);
            }
        }
    }
}

int rcre_extract_features_rgb(const uint8_t* rgb_pixels, 
                             size_t width, 
                             size_t height, 
                             float* out_embedding, 
                             RCRE_Arena* scratch_arena) {
    if (!rgb_pixels || width == 0 || height == 0 || !out_embedding || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_arena_offset = scratch_arena->offset;

    const size_t canon_w = RCRE_CANONICAL_SIZE;
    const size_t canon_h = RCRE_CANONICAL_SIZE;

    /* Step 1: Allocate canonical RGB buffer in arena */
    uint8_t* canonical_rgb = (uint8_t*)rcre_arena_alloc(scratch_arena, canon_w * canon_h * 3, 64);
    if (!canonical_rgb) {
        scratch_arena->offset = saved_arena_offset;
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Bilinear resample input to 256x256 */
    resample_bilinear_rgb(rgb_pixels, width, height, canonical_rgb, canon_w, canon_h);

    /* Step 2: Create RCRE tensors for CV pipeline */
    size_t shape_rgb[3]  = {canon_h, canon_w, 3};
    size_t shape_gray[2] = {canon_h, canon_w};

    RCRE_Tensor frame_tensor = rcre_tensor_create(shape_rgb, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    memcpy(frame_tensor.data, canonical_rgb, canon_w * canon_h * 3);

    RCRE_Tensor gray_tensor = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor blur_tensor = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor sobel_dx    = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor sobel_dy    = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);

    if (!frame_tensor.data || !gray_tensor.data || !blur_tensor.data || !sobel_dx.data || !sobel_dy.data) {
        scratch_arena->offset = saved_arena_offset;
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Step 3: Run RCRE Computer Vision Pipeline */
    /* Rec.601 integer fixed-point luma */
    int status = rcre_cv_rgb_to_gray(&frame_tensor, &gray_tensor);
    if (status != RCRE_OK) {
        scratch_arena->offset = saved_arena_offset;
        return status;
    }

    /* Separable Gaussian blur (sigma = 1.0) for noise attenuation */
    status = rcre_cv_gaussian_blur(&gray_tensor, &blur_tensor, 1.0f, scratch_arena);
    if (status != RCRE_OK) {
        scratch_arena->offset = saved_arena_offset;
        return status;
    }

    /* Spatial 3x3 Sobel gradient operators (dx, dy) */
    status = rcre_cv_sobel(&blur_tensor, &sobel_dx, &sobel_dy, scratch_arena);
    if (status != RCRE_OK) {
        scratch_arena->offset = saved_arena_offset;
        return status;
    }

    /* Step 4: Extract 4x4 Spatial Pyramid Features (16 cells * 4 descriptors = 64-D) */
    const size_t cell_w = canon_w / RCRE_GRID_CELLS_X; /* 64 pixels */
    const size_t cell_h = canon_h / RCRE_GRID_CELLS_Y; /* 64 pixels */
    const float inv_cell_pixels = 1.0f / (float)(cell_w * cell_h);
    const float norm_factor = inv_cell_pixels / 255.0f;

    const uint8_t* rgb_data = (const uint8_t*)frame_tensor.data;
    const uint8_t* dx_data  = (const uint8_t*)sobel_dx.data;
    const uint8_t* dy_data  = (const uint8_t*)sobel_dy.data;

    for (size_t gy = 0; gy < RCRE_GRID_CELLS_Y; ++gy) {
        size_t y_start = gy * cell_h;
        size_t y_end = y_start + cell_h;

        for (size_t gx = 0; gx < RCRE_GRID_CELLS_X; ++gx) {
            size_t x_start = gx * cell_w;
            size_t x_end = x_start + cell_w;

            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;
            uint32_t sum_edge = 0;

            for (size_t y = y_start; y < y_end; ++y) {
                const uint8_t* row_rgb = rgb_data + y * canon_w * 3;
                const uint8_t* row_dx  = dx_data + y * canon_w;
                const uint8_t* row_dy  = dy_data + y * canon_w;

                for (size_t x = x_start; x < x_end; ++x) {
                    sum_r += row_rgb[x * 3 + 0];
                    sum_g += row_rgb[x * 3 + 1];
                    sum_b += row_rgb[x * 3 + 2];

                    /* Combined spatial gradient magnitude: |dx| + |dy| */
                    uint32_t edge_val = (uint32_t)row_dx[x] + (uint32_t)row_dy[x];
                    sum_edge += (edge_val > 255 ? 255 : edge_val);
                }
            }

            size_t cell_idx = gy * RCRE_GRID_CELLS_X + gx;
            size_t feat_idx = cell_idx * 4;

            out_embedding[feat_idx + 0] = (float)sum_r * norm_factor;
            out_embedding[feat_idx + 1] = (float)sum_g * norm_factor;
            out_embedding[feat_idx + 2] = (float)sum_b * norm_factor;
            /* Boost edge weight slightly for structural saliency */
            out_embedding[feat_idx + 3] = (float)sum_edge * norm_factor * 1.5f;
        }
    }

    /* Step 5: L2-normalize the 64-dimensional feature vector */
    float norm_sq = 0.0f;
    for (size_t i = 0; i < RCRE_FEATURE_DIM; ++i) {
        norm_sq += out_embedding[i] * out_embedding[i];
    }

    float inv_norm = 1.0f / sqrtf(norm_sq > 1e-8f ? norm_sq : 1e-8f);
    for (size_t i = 0; i < RCRE_FEATURE_DIM; ++i) {
        out_embedding[i] *= inv_norm;
    }

    /* Bulk scratchpad rewind */
    scratch_arena->offset = saved_arena_offset;
    return RCRE_OK;
}

int rcre_extract_features_file(const char* filepath, 
                             float* out_embedding, 
                             RCRE_Arena* scratch_arena) {
    if (!filepath || !out_embedding || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    /* Load image via single-header STB (forces 3 channels RGB) */
    uint8_t* pixels = stbi_load(filepath, &width, &height, &channels, 3);
    if (!pixels) {
        return RCRE_ERR_INVALID_ARG;
    }

    int status = rcre_extract_features_rgb(pixels, (size_t)width, (size_t)height, out_embedding, scratch_arena);
    stbi_image_free(pixels);

    return status;
}

float rcre_compute_cosine_similarity(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

float rcre_compute_l2_distance(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq);
}
