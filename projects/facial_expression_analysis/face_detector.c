/**
 * @file face_detector.c
 * @brief High-Speed, Pure C23 Face Detection & Canonical Normalization Engine
 */

#include "face_detector.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_LINEAR
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_RAW_PROPOSALS 512

/* Fast skin locus classification in YCbCr / RGB space */
static inline bool is_skin_pixel(uint8_t r, uint8_t g, uint8_t b) {
    /* Basic RGB bounding rule */
    if (r <= g || g <= b || (r - g) < 15 || r < 60) {
        return false;
    }

    /* Standard YCbCr transform and locus */
    float cb = -0.168736f * (float)r - 0.331264f * (float)g + 0.5f * (float)b + 128.0f;
    float cr =  0.5f * (float)r - 0.418688f * (float)g - 0.081312f * (float)b + 128.0f;

    return (cb >= 77.0f && cb <= 127.0f && cr >= 133.0f && cr <= 173.0f);
}

/* Intersection-over-Union (IoU) calculation */
static float compute_iou(const RCRE_FaceBox* a, const RCRE_FaceBox* b) {
    int x1 = a->x > b->x ? a->x : b->x;
    int y1 = a->y > b->y ? a->y : b->y;
    int x2 = (a->x + a->width) < (b->x + b->width) ? (a->x + a->width) : (b->x + b->width);
    int y2 = (a->y + a->height) < (b->y + b->height) ? (a->y + a->height) : (b->y + b->height);

    int inter_w = x2 - x1;
    int inter_h = y2 - y1;
    if (inter_w <= 0 || inter_h <= 0) {
        return 0.0f;
    }

    float inter_area = (float)(inter_w * inter_h);
    float area_a = (float)(a->width * a->height);
    float area_b = (float)(b->width * b->height);
    float union_area = area_a + area_b - inter_area;

    return union_area > 0.0f ? (inter_area / union_area) : 0.0f;
}

/* Non-Maximum Suppression (NMS) */
static size_t apply_nms(RCRE_FaceBox* proposals, size_t num_proposals, float iou_threshold, 
                        RCRE_FaceBox* out_faces, size_t max_faces) {
    /* Sort proposals descending by confidence using selection sort */
    for (size_t i = 0; i < num_proposals; ++i) {
        for (size_t j = i + 1; j < num_proposals; ++j) {
            if (proposals[j].confidence > proposals[i].confidence) {
                RCRE_FaceBox tmp = proposals[i];
                proposals[i] = proposals[j];
                proposals[j] = tmp;
            }
        }
    }

    bool suppressed[MAX_RAW_PROPOSALS];
    memset(suppressed, 0, sizeof(suppressed));

    size_t count = 0;
    for (size_t i = 0; i < num_proposals && count < max_faces; ++i) {
        if (suppressed[i]) continue;

        out_faces[count++] = proposals[i];

        for (size_t j = i + 1; j < num_proposals; ++j) {
            if (!suppressed[j]) {
                if (compute_iou(&proposals[i], &proposals[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return count;
}

RCRE_NODISCARD int rcre_detect_faces(const uint8_t* rgb, 
                                     int width, int height, 
                                     RCRE_FaceBox* out_faces, 
                                     size_t max_faces, 
                                     RCRE_Arena* scratch_arena) {
    if (!rgb || width <= 0 || height <= 0 || !out_faces || max_faces == 0 || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_offset = scratch_arena->offset;

    /* 1. Build Integral Image of Skin Mask (int32 SAT) */
    size_t sat_stride = (size_t)(width + 1);
    size_t sat_elements = sat_stride * (size_t)(height + 1);
    int32_t* sat = (int32_t*)rcre_arena_alloc(scratch_arena, sat_elements * sizeof(int32_t), 64);
    if (!sat) {
        scratch_arena->offset = saved_offset;
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(sat, 0, sat_elements * sizeof(int32_t));

    for (int y = 0; y < height; ++y) {
        int32_t row_sum = 0;
        for (int x = 0; x < width; ++x) {
            size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3;
            uint8_t r = rgb[idx + 0];
            uint8_t g = rgb[idx + 1];
            uint8_t b = rgb[idx + 2];

            bool skin = is_skin_pixel(r, g, b);

            row_sum += skin ? 1 : 0;
            sat[(y + 1) * sat_stride + (x + 1)] = sat[y * sat_stride + (x + 1)] + row_sum;
        }
    }

    /* Helper lambda to query sum inside rectangle (x, y, w, h) in O(1) */
    #define QUERY_SAT(rx, ry, rw, rh) \
        (sat[((ry) + (rh)) * sat_stride + ((rx) + (rw))] - \
         sat[(ry) * sat_stride + ((rx) + (rw))] - \
         sat[((ry) + (rh)) * sat_stride + (rx)] + \
         sat[(ry) * sat_stride + (rx)])

    /* 2. Multi-Scale Scanning Window */
    RCRE_FaceBox raw_proposals[MAX_RAW_PROPOSALS];
    size_t num_proposals = 0;

    int min_face_w = width < height ? (width / 8) : (height / 8);
    if (min_face_w < 32) min_face_w = 32;
    int max_face_w = width < height ? width : height;

    /* Search scales with 1.25x scale factor */
    for (int win_w = min_face_w; win_w <= max_face_w && num_proposals < MAX_RAW_PROPOSALS; win_w = (int)(win_w * 1.25f)) {
        int win_h = (int)(win_w * 1.25f); /* Anthropometric face ratio 1 : 1.25 */
        if (win_h > height || win_w > width) break;

        int stride_x = win_w / 4 > 8 ? win_w / 4 : 8;
        int stride_y = win_h / 4 > 8 ? win_h / 4 : 8;

        for (int y = 0; y + win_h <= height && num_proposals < MAX_RAW_PROPOSALS; y += stride_y) {
            for (int x = 0; x + win_w <= width && num_proposals < MAX_RAW_PROPOSALS; x += stride_x) {
                int total_pixels = win_w * win_h;
                int skin_in_win = QUERY_SAT(x, y, win_w, win_h);
                float skin_density = (float)skin_in_win / (float)total_pixels;

                /* Human faces typically exhibit 35% to 85% skin locus coverage */
                if (skin_density >= 0.30f && skin_density <= 0.88f) {
                    /* Anthropometric test: Eye band (top 20%-45%) vs Cheek band (45%-70%) */
                    int eye_y = y + (int)(win_h * 0.20f);
                    int eye_h = (int)(win_h * 0.25f);
                    int cheek_y = y + (int)(win_h * 0.45f);
                    int cheek_h = (int)(win_h * 0.25f);

                    int eye_skin = QUERY_SAT(x, eye_y, win_w, eye_h);
                    int cheek_skin = QUERY_SAT(x, cheek_y, win_w, cheek_h);

                    float eye_density = (float)eye_skin / (float)(win_w * eye_h);
                    float cheek_density = (float)cheek_skin / (float)(win_w * cheek_h);

                    /* Cheeks have higher unbroken skin density than eye sockets/eyebrows */
                    float confidence = skin_density * 0.6f + (cheek_density >= eye_density ? 0.4f : 0.1f);

                    raw_proposals[num_proposals++] = (RCRE_FaceBox){
                        .x = x,
                        .y = y,
                        .width = win_w,
                        .height = win_h,
                        .confidence = confidence
                    };
                }
            }
        }
    }

    /* Fallback: if skin segmentation was too sparse (e.g. grayscale or extreme color tint),
       generate a center-crop canonical candidate proposal */
    if (num_proposals == 0) {
        int w = width < height ? (int)(width * 0.65f) : (int)(height * 0.65f);
        int h = (int)(w * 1.25f);
        if (h > height) h = height;
        raw_proposals[num_proposals++] = (RCRE_FaceBox){
            .x = (width - w) / 2,
            .y = (height - h) / 3, /* Faces are typically in upper center */
            .width = w,
            .height = h,
            .confidence = 0.50f
        };
    }

    /* 3. Non-Maximum Suppression (NMS) with 0.35 IoU */
    size_t final_count = apply_nms(raw_proposals, num_proposals, 0.35f, out_faces, max_faces);

    #undef QUERY_SAT
    scratch_arena->offset = saved_offset;
    return (int)final_count;
}

RCRE_NODISCARD int rcre_detect_faces_file(const char* filepath, 
                                          RCRE_FaceBox* out_faces, 
                                          size_t max_faces, 
                                          int* out_width, 
                                          int* out_height, 
                                          RCRE_Arena* scratch_arena) {
    if (!filepath || !out_faces || max_faces == 0 || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    int width = 0, height = 0, channels = 0;
    uint8_t* rgb = stbi_load(filepath, &width, &height, &channels, 3);
    if (!rgb) {
        return RCRE_ERR_IO;
    }

    if (out_width) *out_width = width;
    if (out_height) *out_height = height;

    int count = rcre_detect_faces(rgb, width, height, out_faces, max_faces, scratch_arena);
    stbi_image_free(rgb);

    return count;
}

RCRE_NODISCARD int rcre_crop_canonical_face(const uint8_t* rgb, 
                                            int width, int height, 
                                            const RCRE_FaceBox* face, 
                                            uint8_t* out_canonical_rgb, 
                                            RCRE_Arena* scratch_arena) {
    (void)scratch_arena;
    if (!rgb || width <= 0 || height <= 0 || !face || !out_canonical_rgb) {
        return RCRE_ERR_INVALID_ARG;
    }

    const int target_w = RCRE_CANONICAL_FACE_SIZE; /* 128 */
    const int target_h = RCRE_CANONICAL_FACE_SIZE; /* 128 */

    float fx = (float)face->width / (float)target_w;
    float fy = (float)face->height / (float)target_h;

    /* Bilinear interpolation with clamping */
    for (int dst_y = 0; dst_y < target_h; ++dst_y) {
        float src_y_f = (float)face->y + ((float)dst_y + 0.5f) * fy - 0.5f;
        int y0 = (int)floorf(src_y_f);
        int y1 = y0 + 1;
        float dy = src_y_f - (float)y0;

        if (y0 < 0) y0 = 0;
        if (y1 >= height) y1 = height - 1;
        if (y0 >= height) y0 = height - 1;

        for (int dst_x = 0; dst_x < target_w; ++dst_x) {
            float src_x_f = (float)face->x + ((float)dst_x + 0.5f) * fx - 0.5f;
            int x0 = (int)floorf(src_x_f);
            int x1 = x0 + 1;
            float dx = src_x_f - (float)x0;

            if (x0 < 0) x0 = 0;
            if (x1 >= width) x1 = width - 1;
            if (x0 >= width) x0 = width - 1;

            float w00 = (1.0f - dx) * (1.0f - dy);
            float w10 = dx * (1.0f - dy);
            float w01 = (1.0f - dx) * dy;
            float w11 = dx * dy;

            size_t idx00 = ((size_t)y0 * (size_t)width + (size_t)x0) * 3;
            size_t idx10 = ((size_t)y0 * (size_t)width + (size_t)x1) * 3;
            size_t idx01 = ((size_t)y1 * (size_t)width + (size_t)x0) * 3;
            size_t idx11 = ((size_t)y1 * (size_t)width + (size_t)x1) * 3;

            size_t dst_idx = ((size_t)dst_y * (size_t)target_w + (size_t)dst_x) * 3;

            for (int c = 0; c < 3; ++c) {
                float val = w00 * (float)rgb[idx00 + c] +
                            w10 * (float)rgb[idx10 + c] +
                            w01 * (float)rgb[idx01 + c] +
                            w11 * (float)rgb[idx11 + c];

                if (val < 0.0f) val = 0.0f;
                if (val > 255.0f) val = 255.0f;
                out_canonical_rgb[dst_idx + c] = (uint8_t)val;
            }
        }
    }

    return RCRE_OK;
}
