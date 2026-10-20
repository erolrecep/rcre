/**
 * @file action_units.c
 * @brief Facial Action Coding System (FACS) Biomechanical Action Unit Extractor
 */

#include "action_units.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CANON_DIM 128

static float distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

RCRE_NODISCARD int rcre_extract_action_units(const uint8_t* canonical_rgb, 
                                            const RCRE_FacialLandmarks* landmarks, 
                                            RCRE_ActionUnits* out_au, 
                                            RCRE_Arena* scratch_arena) {
    if (!canonical_rgb || !landmarks || !out_au || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_offset = scratch_arena->offset;
    const int W = CANON_DIM;
    const int H = CANON_DIM;

    float iod = landmarks->inter_ocular_dist;
    if (iod < 10.0f) iod = 48.0f;

    const RCRE_LandmarkPoint* pts = landmarks->points;

    /* 1. Mouth Geometry */
    float mouth_w = distance_2d(pts[LANDMARK_MOUTH_LEFT_CORNER].x, pts[LANDMARK_MOUTH_LEFT_CORNER].y,
                                pts[LANDMARK_MOUTH_RIGHT_CORNER].x, pts[LANDMARK_MOUTH_RIGHT_CORNER].y);
    float lip_h = distance_2d(pts[LANDMARK_UPPER_LIP_PEAK].x, pts[LANDMARK_UPPER_LIP_PEAK].y,
                              pts[LANDMARK_LOWER_LIP_CENTER].x, pts[LANDMARK_LOWER_LIP_CENTER].y);

    out_au->mouth_aspect_ratio = lip_h / (mouth_w > 1e-4f ? mouth_w : 1e-4f);
    out_au->mouth_width_ratio = mouth_w / iod;

    /* Smile Curvature (AU12): Corner elevation relative to mouth horizontal center */
    float mouth_center_y = (pts[LANDMARK_UPPER_LIP_PEAK].y + pts[LANDMARK_LOWER_LIP_CENTER].y) * 0.5f;
    float avg_corner_y = (pts[LANDMARK_MOUTH_LEFT_CORNER].y + pts[LANDMARK_MOUTH_RIGHT_CORNER].y) * 0.5f;
    /* When corners are elevated (smiling), avg_corner_y is smaller than mouth_center_y in screen coords */
    out_au->smile_curvature = (mouth_center_y - avg_corner_y) / iod;

    /* 2. Eye Aspect Ratios (EAR) */
    float eye_left_w = distance_2d(pts[LANDMARK_LEFT_EYE_OUTER].x, pts[LANDMARK_LEFT_EYE_OUTER].y,
                                   pts[LANDMARK_LEFT_EYE_INNER].x, pts[LANDMARK_LEFT_EYE_INNER].y);
    float eye_right_w = distance_2d(pts[LANDMARK_RIGHT_EYE_INNER].x, pts[LANDMARK_RIGHT_EYE_INNER].y,
                                    pts[LANDMARK_RIGHT_EYE_OUTER].x, pts[LANDMARK_RIGHT_EYE_OUTER].y);

    out_au->eye_aspect_ratio_left = (iod * 0.16f) / (eye_left_w > 1e-4f ? eye_left_w : 1e-4f);
    out_au->eye_aspect_ratio_right = (iod * 0.16f) / (eye_right_w > 1e-4f ? eye_right_w : 1e-4f);

    /* 3. Eyebrow Activations (AU1, AU2, AU4) */
    out_au->brow_elevation_left = (pts[LANDMARK_LEFT_EYE_CENTER].y - pts[LANDMARK_LEFT_BROW_ARCH].y) / iod;
    out_au->brow_elevation_right = (pts[LANDMARK_RIGHT_EYE_CENTER].y - pts[LANDMARK_RIGHT_BROW_ARCH].y) / iod;

    out_au->brow_furrow_distance = distance_2d(pts[LANDMARK_LEFT_BROW_ARCH].x, pts[LANDMARK_LEFT_BROW_ARCH].y,
                                               pts[LANDMARK_RIGHT_BROW_ARCH].x, pts[LANDMARK_RIGHT_BROW_ARCH].y) / iod;

    /* Symmetry Index */
    out_au->facial_symmetry_index = 1.0f - fabsf(out_au->brow_elevation_left - out_au->brow_elevation_right);
    if (out_au->facial_symmetry_index < 0.0f) out_au->facial_symmetry_index = 0.0f;

    /* 4. Texture & Wrinkle Energy (Glabella & Nasolabial folds) */
    size_t shape_rgb[3]  = {(size_t)H, (size_t)W, 3};
    size_t shape_gray[2] = {(size_t)H, (size_t)W};

    RCRE_Tensor tensor_rgb  = rcre_tensor_create(shape_rgb, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor tensor_gray = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor tensor_dx   = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, scratch_arena);
    RCRE_Tensor tensor_dy   = rcre_tensor_create(shape_gray, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, scratch_arena);

    if (tensor_rgb.data && tensor_gray.data && tensor_dx.data && tensor_dy.data) {
        memcpy(tensor_rgb.data, canonical_rgb, (size_t)(W * H * 3));
        (void)rcre_cv_rgb_to_gray(&tensor_rgb, &tensor_gray);
        (void)rcre_cv_sobel(&tensor_gray, &tensor_dx, &tensor_dy, scratch_arena);

        const float* dx = (const float*)tensor_dx.data;
        const float* dy = (const float*)tensor_dy.data;

        /* Glabella energy (between eyebrows: x in [56, 72], y in [24, 42]) */
        float glabella_sum = 0.0f;
        int glabella_pixels = 0;
        for (int y = 24; y <= 42; ++y) {
            for (int x = 56; x <= 72; ++x) {
                float gx = dx[y * W + x];
                float gy = dy[y * W + x];
                glabella_sum += sqrtf(gx * gx + gy * gy);
                glabella_pixels++;
            }
        }
        out_au->glabella_wrinkle_energy = glabella_sum / (float)(glabella_pixels > 0 ? glabella_pixels * 128 : 1);

        /* Nasolabial energy (cheeks / smile crease region) */
        float nasolabial_sum = 0.0f;
        int nasolabial_pixels = 0;
        for (int y = 68; y <= 92; ++y) {
            for (int x = 32; x <= 46; ++x) {
                float gx = dx[y * W + x];
                float gy = dy[y * W + x];
                nasolabial_sum += sqrtf(gx * gx + gy * gy);
                nasolabial_pixels++;
            }
            for (int x = 82; x <= 96; ++x) {
                float gx = dx[y * W + x];
                float gy = dy[y * W + x];
                nasolabial_sum += sqrtf(gx * gx + gy * gy);
                nasolabial_pixels++;
            }
        }
        out_au->nasolabial_wrinkle_energy = nasolabial_sum / (float)(nasolabial_pixels > 0 ? nasolabial_pixels * 128 : 1);
    } else {
        out_au->glabella_wrinkle_energy = 0.0f;
        out_au->nasolabial_wrinkle_energy = 0.0f;
    }

    /* 5. Pack Normalized 16-D Feature Vector */
    out_au->features[0]  = out_au->mouth_aspect_ratio;
    out_au->features[1]  = out_au->mouth_width_ratio;
    out_au->features[2]  = out_au->smile_curvature;
    out_au->features[3]  = out_au->eye_aspect_ratio_left;
    out_au->features[4]  = out_au->eye_aspect_ratio_right;
    out_au->features[5]  = out_au->brow_elevation_left;
    out_au->features[6]  = out_au->brow_elevation_right;
    out_au->features[7]  = (out_au->brow_elevation_left + out_au->brow_elevation_right) * 0.5f;
    out_au->features[8]  = out_au->brow_furrow_distance;
    out_au->features[9]  = out_au->glabella_wrinkle_energy;
    out_au->features[10] = out_au->nasolabial_wrinkle_energy;
    out_au->features[11] = out_au->facial_symmetry_index;
    out_au->features[12] = lip_h / iod;
    out_au->features[13] = (pts[LANDMARK_UPPER_LIP_PEAK].y - pts[LANDMARK_NOSE_APEX].y) / iod;
    out_au->features[14] = (pts[LANDMARK_LOWER_LIP_CENTER].y - pts[LANDMARK_NOSE_APEX].y) / iod;
    out_au->features[15] = (pts[LANDMARK_LEFT_EYE_CENTER].y + pts[LANDMARK_RIGHT_EYE_CENTER].y) / (2.0f * (float)H);

    scratch_arena->offset = saved_offset;
    return RCRE_OK;
}
