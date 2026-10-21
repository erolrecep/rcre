/**
 * @file landmark_detector.c
 * @brief Anthropometric Facial Landmark Localization & Geometry Engine
 */

#include "landmark_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CANON_DIM 128

/* Find pupil center using iris-sclera horizontal contrast (dark center, bright sclera) */
static void find_pupil_center(const uint8_t* gray, int w, 
                              int rx0, int ry0, int rx1, int ry1, 
                              float* out_x, float* out_y) {
    float best_x = (float)(rx0 + rx1) * 0.5f;
    float best_y = (float)(ry0 + ry1) * 0.5f;
    float max_contrast = -1e9f;

    for (int y = ry0; y <= ry1; ++y) {
        for (int x = rx0; x <= rx1; ++x) {
            int center_val = (int)gray[y * w + x];

            /* Sclera samples: 5 pixels left and right of candidate pupil */
            int left_sclera = (x >= 5) ? (int)gray[y * w + (x - 5)] : center_val;
            int right_sclera = (x + 5 < w) ? (int)gray[y * w + (x + 5)] : center_val;

            /* Pupil has maximum contrast against lateral sclera and minimum absolute intensity */
            float contrast = (float)(left_sclera - center_val) + 
                             (float)(right_sclera - center_val) - 
                             (float)center_val * 0.5f;

            if (contrast > max_contrast) {
                max_contrast = contrast;
                best_x = (float)x;
                best_y = (float)y;
            }
        }
    }

    *out_x = best_x;
    *out_y = best_y;
}

/* Find eyebrow arch directly above localized eye center */
static void find_eyebrow_arch(const uint8_t* gray, int w, 
                              int px, int py, 
                              float* out_x, float* out_y) {
    int min_val = 256;
    float best_x = (float)px;
    float best_y = (float)(py - 14);

    int y0 = py - 28;
    if (y0 < 10) y0 = 10;
    int y1 = py - 9;
    if (y1 < y0) y1 = y0;

    for (int y = y0; y <= y1; ++y) {
        for (int x = px - 8; x <= px + 8; ++x) {
            if (x < 0 || x >= w) continue;
            int val = (int)gray[y * w + x];
            if (val < min_val) {
                min_val = val;
                best_x = (float)x;
                best_y = (float)y;
            }
        }
    }

    *out_x = best_x;
    *out_y = best_y;
}

/* Localize vertical center of mouth fissure and trace upper/lower lip margins */
static void find_mouth_fissure(const uint8_t* rgb, const uint8_t* gray, int w,
                               int cx, int ry0, int ry1,
                               float* out_center_y, float* out_upper_y, float* out_lower_y) {
    float best_y = (float)(ry0 + ry1) * 0.5f;
    float max_score = -1e9f;

    /* Scan vertical strip [cx - 3, cx + 3] */
    for (int y = ry0; y <= ry1; ++y) {
        float row_score = 0.0f;
        int count = 0;
        for (int x = cx - 3; x <= cx + 3; ++x) {
            if (x < 0 || x >= w) continue;
            size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 3;
            int r = rgb[idx + 0];
            int g = rgb[idx + 1];
            if (r <= g) continue;

            int intensity = gray[y * w + x];
            row_score += (float)(r - g) * 2.0f + (float)(255 - intensity);
            count++;
        }
        if (count > 0) {
            row_score /= (float)count;
            if (row_score > max_score) {
                max_score = row_score;
                best_y = (float)y;
            }
        }
    }

    /* Trace upper lip boundary */
    int up_y = (int)best_y;
    while (up_y > ry0) {
        size_t idx = ((size_t)up_y * (size_t)w + (size_t)cx) * 3;
        int r = rgb[idx + 0];
        int g = rgb[idx + 1];
        int intensity = gray[up_y * w + cx];
        bool is_oral = ((r - g) >= 80) || (intensity < 65 && r >= g);
        if (!is_oral) break;
        up_y--;
    }

    /* Trace lower lip boundary */
    int low_y = (int)best_y;
    while (low_y < ry1) {
        size_t idx = ((size_t)low_y * (size_t)w + (size_t)cx) * 3;
        int r = rgb[idx + 0];
        int g = rgb[idx + 1];
        int intensity = gray[low_y * w + cx];
        bool is_oral = ((r - g) >= 80) || (intensity < 65 && r >= g);
        if (!is_oral) break;
        low_y++;
    }

    float upper_y = (float)up_y;
    float lower_y = (float)low_y;
    if (lower_y - upper_y < 8.0f) {
        upper_y = best_y - 4.0f;
        lower_y = best_y + 4.0f;
    }

    *out_center_y = best_y;
    *out_upper_y = upper_y;
    *out_lower_y = lower_y;
}

/* Find mouth corner constrained to anthropometric mouth span */
static void find_mouth_corner_constrained(const uint8_t* rgb, const uint8_t* gray, int w, 
                                          int rx0, int ry0, int rx1, int ry1, bool is_left, 
                                          float* out_x, float* out_y) {
    float best_x = (float)(rx0 + rx1) * 0.5f;
    float best_y = (float)(ry0 + ry1) * 0.5f;
    float max_score = -1e9f;

    for (int y = ry0; y <= ry1; ++y) {
        for (int x = rx0; x <= rx1; ++x) {
            if (x < 0 || x >= w) continue;
            size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 3;
            int r = rgb[idx + 0];
            int g = rgb[idx + 1];
            if (r <= g) continue;

            int intensity = gray[y * w + x];
            float score = (float)(r - g) * 2.0f + (float)(255 - intensity) * 0.8f;

            if (is_left) {
                score += (float)(rx1 - x) * 0.5f;
            } else {
                score += (float)(x - rx0) * 0.5f;
            }

            if (score > max_score) {
                max_score = score;
                best_x = (float)x;
                best_y = (float)y;
            }
        }
    }

    *out_x = best_x;
    *out_y = best_y;
}

RCRE_NODISCARD int rcre_detect_landmarks(const uint8_t* canonical_rgb, 
                                         RCRE_FacialLandmarks* out_landmarks, 
                                         RCRE_Arena* scratch_arena) {
    if (!canonical_rgb || !out_landmarks || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_offset = scratch_arena->offset;
    const int W = CANON_DIM;
    const int H = CANON_DIM;

    /* 1. Allocate Grayscale and Gaussian tensors */
    size_t shape_rgb[3]  = {(size_t)H, (size_t)W, 3};
    size_t shape_gray[2] = {(size_t)H, (size_t)W};

    RCRE_Tensor tensor_rgb  = rcre_tensor_create(shape_rgb, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor tensor_gray = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor tensor_blur = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);

    if (!tensor_rgb.data || !tensor_gray.data || !tensor_blur.data) {
        scratch_arena->offset = saved_offset;
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    memcpy(tensor_rgb.data, canonical_rgb, (size_t)(W * H * 3));
    (void)rcre_cv_rgb_to_gray(&tensor_rgb, &tensor_gray);
    (void)rcre_cv_gaussian_blur(&tensor_gray, &tensor_blur, 1.0f, scratch_arena);

    const uint8_t* blurred = (const uint8_t*)tensor_blur.data;

    /* 2. Pupil Centers via Iris-Sclera Horizontal Contrast */
    float left_eye_x = 0.0f, left_eye_y = 0.0f;
    float right_eye_x = 0.0f, right_eye_y = 0.0f;

    find_pupil_center(blurred, W, 30, 46, 52, 60, &left_eye_x, &left_eye_y);
    find_pupil_center(blurred, W, 76, 46, 98, 60, &right_eye_x, &right_eye_y);

    /* Enforce vertical level between eyes (anatomical balance) */
    if (fabsf(left_eye_y - right_eye_y) > 6.0f) {
        float avg_y = (left_eye_y + right_eye_y) * 0.5f;
        left_eye_y = avg_y;
        right_eye_y = avg_y;
    }

    out_landmarks->points[LANDMARK_LEFT_EYE_CENTER] = (RCRE_LandmarkPoint){left_eye_x, left_eye_y, 0.90f};
    out_landmarks->points[LANDMARK_RIGHT_EYE_CENTER] = (RCRE_LandmarkPoint){right_eye_x, right_eye_y, 0.90f};

    /* Inter-Ocular Distance (IOD) */
    float iod_dx = right_eye_x - left_eye_x;
    float iod_dy = right_eye_y - left_eye_y;
    float iod = sqrtf(iod_dx * iod_dx + iod_dy * iod_dy);
    if (iod < 28.0f) iod = 48.0f; /* Anthropometric fallback */
    out_landmarks->inter_ocular_dist = iod;

    /* Eye corners relative to centers */
    float eye_half_w = iod * 0.16f;
    out_landmarks->points[LANDMARK_LEFT_EYE_OUTER] = (RCRE_LandmarkPoint){left_eye_x - eye_half_w, left_eye_y, 0.85f};
    out_landmarks->points[LANDMARK_LEFT_EYE_INNER] = (RCRE_LandmarkPoint){left_eye_x + eye_half_w, left_eye_y, 0.85f};
    out_landmarks->points[LANDMARK_RIGHT_EYE_INNER] = (RCRE_LandmarkPoint){right_eye_x - eye_half_w, right_eye_y, 0.85f};
    out_landmarks->points[LANDMARK_RIGHT_EYE_OUTER] = (RCRE_LandmarkPoint){right_eye_x + eye_half_w, right_eye_y, 0.85f};

    /* 3. Eyebrows (Anchored strictly above the localized eye centers) */
    float left_brow_x = 0.0f, left_brow_y = 0.0f;
    float right_brow_x = 0.0f, right_brow_y = 0.0f;

    find_eyebrow_arch(blurred, W, (int)left_eye_x, (int)left_eye_y, &left_brow_x, &left_brow_y);
    find_eyebrow_arch(blurred, W, (int)right_eye_x, (int)right_eye_y, &right_brow_x, &right_brow_y);

    out_landmarks->points[LANDMARK_LEFT_BROW_ARCH] = (RCRE_LandmarkPoint){left_brow_x, left_brow_y, 0.85f};
    out_landmarks->points[LANDMARK_RIGHT_BROW_ARCH] = (RCRE_LandmarkPoint){right_brow_x, right_brow_y, 0.85f};

    /* 4. Nose Apex (Central midline between eyes and mouth) */
    float mid_eye_x = (left_eye_x + right_eye_x) * 0.5f;
    float nose_y = (left_eye_y + right_eye_y) * 0.5f + iod * 0.38f;
    if (nose_y > 82.0f) nose_y = 82.0f;
    out_landmarks->points[LANDMARK_NOSE_APEX] = (RCRE_LandmarkPoint){mid_eye_x, nose_y, 0.85f};

    /* 5. Mouth Complex (Constrained to anatomical oral cavity span) */
    float mouth_cy = nose_y + iod * 0.36f;
    if (mouth_cy > 108.0f) mouth_cy = 108.0f;

    float mouth_center_y = mouth_cy, upper_lip_y = mouth_cy - 5.0f, lower_lip_y = mouth_cy + 5.0f;
    find_mouth_fissure(canonical_rgb, blurred, W, (int)mid_eye_x, (int)(nose_y + 10.0f), (int)(H - 12),
                       &mouth_center_y, &upper_lip_y, &lower_lip_y);

    float mouth_half_w = iod * 0.42f; /* Half width of mouth ~ 20px */
    if (mouth_half_w < 16.0f) mouth_half_w = 16.0f;
    if (mouth_half_w > 26.0f) mouth_half_w = 26.0f;

    int left_c_x0 = (int)(mid_eye_x - mouth_half_w - 6.0f);
    int left_c_x1 = (int)(mid_eye_x - mouth_half_w + 6.0f);
    int right_c_x0 = (int)(mid_eye_x + mouth_half_w - 6.0f);
    int right_c_x1 = (int)(mid_eye_x + mouth_half_w + 6.0f);
    int my_min = (int)(mouth_center_y - iod * 0.28f);
    int my_max = (int)(mouth_center_y + iod * 0.28f);

    float mouth_lx = mid_eye_x - mouth_half_w, mouth_ly = mouth_center_y;
    float mouth_rx = mid_eye_x + mouth_half_w, mouth_ry = mouth_center_y;

    find_mouth_corner_constrained(canonical_rgb, blurred, W, left_c_x0, my_min, left_c_x1, my_max, true, &mouth_lx, &mouth_ly);
    find_mouth_corner_constrained(canonical_rgb, blurred, W, right_c_x0, my_min, right_c_x1, my_max, false, &mouth_rx, &mouth_ry);

    out_landmarks->points[LANDMARK_MOUTH_LEFT_CORNER] = (RCRE_LandmarkPoint){mouth_lx, mouth_ly, 0.85f};
    out_landmarks->points[LANDMARK_MOUTH_RIGHT_CORNER] = (RCRE_LandmarkPoint){mouth_rx, mouth_ry, 0.85f};
    out_landmarks->points[LANDMARK_UPPER_LIP_PEAK] = (RCRE_LandmarkPoint){mid_eye_x, upper_lip_y, 0.85f};
    out_landmarks->points[LANDMARK_LOWER_LIP_CENTER] = (RCRE_LandmarkPoint){mid_eye_x, lower_lip_y, 0.85f};

    scratch_arena->offset = saved_offset;
    return RCRE_OK;
}
