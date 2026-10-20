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

/* Find minimum intensity coordinate (dark valley) in ROI */
static void find_dark_valley(const uint8_t* gray, int w, 
                             int rx0, int ry0, int rx1, int ry1, 
                             float* out_x, float* out_y) {
    int min_val = 256;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float count = 0.0f;

    for (int y = ry0; y <= ry1; ++y) {
        for (int x = rx0; x <= rx1; ++x) {
            int val = (int)gray[y * w + x];
            if (val < min_val) {
                min_val = val;
            }
        }
    }

    /* Centroid of the darkest 15% threshold in the region */
    int threshold = min_val + 20;
    for (int y = ry0; y <= ry1; ++y) {
        for (int x = rx0; x <= rx1; ++x) {
            int val = (int)gray[y * w + x];
            if (val <= threshold) {
                float weight = (float)(threshold - val + 1);
                sum_x += (float)x * weight;
                sum_y += (float)y * weight;
                count += weight;
            }
        }
    }

    if (count > 0.0f) {
        *out_x = sum_x / count;
        *out_y = sum_y / count;
    } else {
        *out_x = (float)(rx0 + rx1) * 0.5f;
        *out_y = (float)(ry0 + ry1) * 0.5f;
    }
}

/* Find mouth corner using lip chrominance & darkness */
static void find_mouth_corner(const uint8_t* rgb, const uint8_t* gray, int w, 
                              int rx0, int ry0, int rx1, int ry1, bool is_left, 
                              float* out_x, float* out_y) {
    float best_x = (float)(rx0 + rx1) * 0.5f;
    float best_y = (float)(ry0 + ry1) * 0.5f;
    float max_score = -1e9f;

    for (int y = ry0; y <= ry1; ++y) {
        for (int x = rx0; x <= rx1; ++x) {
            size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 3;
            int r = rgb[idx + 0];
            int g = rgb[idx + 1];
            (void)rgb[idx + 2];
            if (r <= g) continue; /* Reject non-skin/non-lip background */

            int intensity = gray[y * w + x];

            /* Mouth fissure is dark and red-biased: (R - G) + (255 - intensity) */
            float score = (float)(r - g) * 2.0f + (float)(255 - intensity) * 0.8f;

            /* Prefer horizontal extrema depending on whether finding left or right corner */
            if (is_left) {
                score += (float)(rx1 - x) * 1.5f;
            } else {
                score += (float)(x - rx0) * 1.5f;
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

/* Localize vertical center of mouth and upper/lower lip extremities at central column */
static void find_mouth_center_and_lips(const uint8_t* rgb, const uint8_t* gray, int w,
                                       int cx, int ry0, int ry1,
                                       float* out_center_y, float* out_upper_y, float* out_lower_y) {
    float best_y = (float)(ry0 + ry1) * 0.5f;
    float max_score = -1e9f;

    /* Scan vertical strip [cx - 4, cx + 4] */
    for (int y = ry0; y <= ry1; ++y) {
        float row_score = 0.0f;
        int count = 0;
        for (int x = cx - 4; x <= cx + 4; ++x) {
            if (x < 0 || x >= w) continue;
            size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 3;
            int r = rgb[idx + 0];
            int g = rgb[idx + 1];
            if (r <= g) continue;

            int intensity = gray[y * w + x];
            /* Peak lip prominence: red excess + oral cavity / fissure darkness */
            row_score += (float)(r - g) * 2.5f + (float)(255 - intensity);
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
        bool is_lip = ((r - g) >= 75) || (intensity < 70 && r > g);
        if (!is_lip) break; /* Transition from lip to skin */
        up_y--;
    }

    /* Trace lower lip boundary */
    int low_y = (int)best_y;
    while (low_y < ry1) {
        size_t idx = ((size_t)low_y * (size_t)w + (size_t)cx) * 3;
        int r = rgb[idx + 0];
        int g = rgb[idx + 1];
        int intensity = gray[low_y * w + cx];
        bool is_lip = ((r - g) >= 75) || (intensity < 70 && r > g);
        if (!is_lip) break; /* Transition from lip to skin */
        low_y++;
    }

    float upper_y = (float)up_y;
    float lower_y = (float)low_y;
    if (lower_y - upper_y < 6.0f) {
        upper_y = best_y - 4.0f;
        lower_y = best_y + 4.0f;
    }

    *out_center_y = best_y;
    *out_upper_y = upper_y;
    *out_lower_y = lower_y;
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

    /* 2. Eye Centers (Dark Valley) */
    float left_eye_x = 0.0f, left_eye_y = 0.0f;
    float right_eye_x = 0.0f, right_eye_y = 0.0f;

    find_dark_valley(blurred, W, 22, 38, 54, 60, &left_eye_x, &left_eye_y);
    find_dark_valley(blurred, W, 74, 38, 106, 60, &right_eye_x, &right_eye_y);

    out_landmarks->points[LANDMARK_LEFT_EYE_CENTER] = (RCRE_LandmarkPoint){left_eye_x, left_eye_y, 0.90f};
    out_landmarks->points[LANDMARK_RIGHT_EYE_CENTER] = (RCRE_LandmarkPoint){right_eye_x, right_eye_y, 0.90f};

    /* Inter-Ocular Distance (IOD) */
    float iod_dx = right_eye_x - left_eye_x;
    float iod_dy = right_eye_y - left_eye_y;
    float iod = sqrtf(iod_dx * iod_dx + iod_dy * iod_dy);
    if (iod < 24.0f) iod = 48.0f; /* Anthropometric fallback */
    out_landmarks->inter_ocular_dist = iod;

    /* Eye corners relative to centers */
    float eye_half_w = iod * 0.18f;
    out_landmarks->points[LANDMARK_LEFT_EYE_OUTER] = (RCRE_LandmarkPoint){left_eye_x - eye_half_w, left_eye_y, 0.85f};
    out_landmarks->points[LANDMARK_LEFT_EYE_INNER] = (RCRE_LandmarkPoint){left_eye_x + eye_half_w, left_eye_y, 0.85f};
    out_landmarks->points[LANDMARK_RIGHT_EYE_INNER] = (RCRE_LandmarkPoint){right_eye_x - eye_half_w, right_eye_y, 0.85f};
    out_landmarks->points[LANDMARK_RIGHT_EYE_OUTER] = (RCRE_LandmarkPoint){right_eye_x + eye_half_w, right_eye_y, 0.85f};

    /* 3. Eyebrows (Scan above eye centers for darkest horizontal ridge) */
    float left_brow_x = left_eye_x, left_brow_y = 0.0f;
    float right_brow_x = right_eye_x, right_brow_y = 0.0f;

    find_dark_valley(blurred, W, (int)(left_eye_x - 14), 16, (int)(left_eye_x + 14), (int)(left_eye_y - 4),
                     &left_brow_x, &left_brow_y);
    find_dark_valley(blurred, W, (int)(right_eye_x - 14), 16, (int)(right_eye_x + 14), (int)(right_eye_y - 4),
                     &right_brow_x, &right_brow_y);

    out_landmarks->points[LANDMARK_LEFT_BROW_ARCH] = (RCRE_LandmarkPoint){left_brow_x, left_brow_y, 0.85f};
    out_landmarks->points[LANDMARK_RIGHT_BROW_ARCH] = (RCRE_LandmarkPoint){right_brow_x, right_brow_y, 0.85f};

    /* 4. Nose Apex (Central region between eyes and mouth) */
    float nose_x = 0.0f, nose_y = 0.0f;
    int nose_y_min = (int)((left_eye_y + right_eye_y) * 0.5f + iod * 0.25f);
    int nose_y_max = (int)(nose_y_min + iod * 0.35f);
    if (nose_y_max >= H - 20) nose_y_max = H - 21;
    find_dark_valley(blurred, W, 52, nose_y_min, 76, nose_y_max, &nose_x, &nose_y);
    out_landmarks->points[LANDMARK_NOSE_APEX] = (RCRE_LandmarkPoint){nose_x, nose_y, 0.80f};

    /* 5. Mouth Complex (Left corner, right corner, upper lip, lower lip) */
    float mouth_lx = 0.0f, mouth_ly = 0.0f;
    float mouth_rx = 0.0f, mouth_ry = 0.0f;

    int mouth_y0 = (int)(nose_y + iod * 0.12f);
    int mouth_y1 = (int)(mouth_y0 + iod * 0.70f);
    if (mouth_y1 >= H - 4) mouth_y1 = H - 4;

    find_mouth_corner(canonical_rgb, blurred, W, 22, mouth_y0, 56, mouth_y1, true, &mouth_lx, &mouth_ly);
    find_mouth_corner(canonical_rgb, blurred, W, 72, mouth_y0, 106, mouth_y1, false, &mouth_rx, &mouth_ry);

    out_landmarks->points[LANDMARK_MOUTH_LEFT_CORNER] = (RCRE_LandmarkPoint){mouth_lx, mouth_ly, 0.85f};
    out_landmarks->points[LANDMARK_MOUTH_RIGHT_CORNER] = (RCRE_LandmarkPoint){mouth_rx, mouth_ry, 0.85f};

    /* Mouth center column */
    int mouth_cx_int = (int)((mouth_lx + mouth_rx) * 0.5f);
    float mouth_center_y = 0.0f, upper_lip_y = 0.0f, lower_lip_y = 0.0f;
    find_mouth_center_and_lips(canonical_rgb, blurred, W, mouth_cx_int, mouth_y0, mouth_y1,
                               &mouth_center_y, &upper_lip_y, &lower_lip_y);

    out_landmarks->points[LANDMARK_UPPER_LIP_PEAK] = (RCRE_LandmarkPoint){(float)mouth_cx_int, upper_lip_y, 0.80f};
    out_landmarks->points[LANDMARK_LOWER_LIP_CENTER] = (RCRE_LandmarkPoint){(float)mouth_cx_int, lower_lip_y, 0.80f};

    scratch_arena->offset = saved_offset;
    return RCRE_OK;
}
