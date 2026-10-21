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

#define MAX_RAW_PROPOSALS 4096
#define MAX_CLUSTERS 64

/* Fast, illumination-robust skin locus classification with hemoglobin chromophore separation */
static inline bool is_skin_pixel(uint8_t r, uint8_t g, uint8_t b) {
    int y_lum = (77 * (int)r + 150 * (int)g + 29 * (int)b) >> 8;
    int cb = ((-43 * (int)r - 85 * (int)g + 128 * (int)b) >> 8) + 128;
    int cr = ((128 * (int)r - 107 * (int)g - 21 * (int)b) >> 8) + 128;

    return (cb >= 75 && cb <= 145 && 
            cr >= 120 && cr <= 180 && 
            y_lum >= 25 && y_lum <= 245 && 
            ((int)r - (int)b >= 12) && 
            ((int)r - (int)g >= 6));
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

typedef struct {
    int x, y, width, height;
    float score;
} RawProposal;

typedef struct {
    int x, y, width, height;
    float total_score;
    int count;
} FaceCluster;

RCRE_NODISCARD int rcre_detect_faces(const uint8_t* rgb, 
                                     int width, int height, 
                                     RCRE_FaceBox* out_faces, 
                                     size_t max_faces, 
                                     RCRE_Arena* scratch_arena) {
    if (!rgb || width <= 0 || height <= 0 || !out_faces || max_faces == 0 || !scratch_arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_offset = scratch_arena->offset;

    size_t sat_stride = (size_t)(width + 1);
    size_t sat_elements = sat_stride * (size_t)(height + 1);

    /* 1. Build Integral Images: Luminance, Squared Luminance (for variance), and Skin SAT */
    uint32_t* sat_lum = (uint32_t*)rcre_arena_alloc(scratch_arena, sat_elements * sizeof(uint32_t), 64);
    uint64_t* sat_lum_sq = (uint64_t*)rcre_arena_alloc(scratch_arena, sat_elements * sizeof(uint64_t), 64);
    int32_t*  sat_skin = (int32_t*)rcre_arena_alloc(scratch_arena, sat_elements * sizeof(int32_t), 64);

    if (!sat_lum || !sat_lum_sq || !sat_skin) {
        scratch_arena->offset = saved_offset;
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    memset(sat_lum, 0, sat_elements * sizeof(uint32_t));
    memset(sat_lum_sq, 0, sat_elements * sizeof(uint64_t));
    memset(sat_skin, 0, sat_elements * sizeof(int32_t));

    for (int y = 0; y < height; ++y) {
        uint32_t row_lum = 0;
        uint64_t row_lum_sq = 0;
        int32_t  row_skin = 0;

        for (int x = 0; x < width; ++x) {
            size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3;
            uint8_t r = rgb[idx + 0];
            uint8_t g = rgb[idx + 1];
            uint8_t b = rgb[idx + 2];

            uint32_t y_val = (uint32_t)((77 * (int)r + 150 * (int)g + 29 * (int)b) >> 8);
            bool skin = is_skin_pixel(r, g, b);

            row_lum += y_val;
            row_lum_sq += (uint64_t)y_val * (uint64_t)y_val;
            row_skin += skin ? 1 : 0;

            size_t curr_idx = (size_t)(y + 1) * sat_stride + (size_t)(x + 1);
            size_t prev_idx = (size_t)y * sat_stride + (size_t)(x + 1);

            sat_lum[curr_idx] = sat_lum[prev_idx] + row_lum;
            sat_lum_sq[curr_idx] = sat_lum_sq[prev_idx] + row_lum_sq;
            sat_skin[curr_idx] = sat_skin[prev_idx] + row_skin;
        }
    }

    /* Fast O(1) rectangle query macros */
    #define QUERY_LUM(rx, ry, rw, rh) \
        (float)(sat_lum[((ry) + (rh)) * sat_stride + ((rx) + (rw))] - \
                sat_lum[(ry) * sat_stride + ((rx) + (rw))] - \
                sat_lum[((ry) + (rh)) * sat_stride + (rx)] + \
                sat_lum[(ry) * sat_stride + (rx)])

    #define QUERY_LUM_SQ(rx, ry, rw, rh) \
        (double)(sat_lum_sq[((ry) + (rh)) * sat_stride + ((rx) + (rw))] - \
                 sat_lum_sq[(ry) * sat_stride + ((rx) + (rw))] - \
                 sat_lum_sq[((ry) + (rh)) * sat_stride + (rx)] + \
                 sat_lum_sq[(ry) * sat_stride + (rx)])

    #define QUERY_SKIN(rx, ry, rw, rh) \
        (sat_skin[((ry) + (rh)) * sat_stride + ((rx) + (rw))] - \
         sat_skin[(ry) * sat_stride + ((rx) + (rw))] - \
         sat_skin[((ry) + (rh)) * sat_stride + (rx)] + \
         sat_skin[(ry) * sat_stride + (rx)])

    /* 2. Coarse-to-Fine Multi-Scale Cascaded Scanning Window */
    RawProposal proposals[MAX_RAW_PROPOSALS];
    size_t num_proposals = 0;

    int min_face_w = 44;
    int max_face_w = (int)(height * 0.70f);
    if (max_face_w > width) max_face_w = width;
    if (min_face_w > max_face_w) min_face_w = max_face_w;

    /* Scales from min to max with 1.15x geometric progression */
    int scales[32];
    int num_scales = 0;
    for (int s = min_face_w; s <= max_face_w && num_scales < 32; s = (int)(s * 1.15f)) {
        scales[num_scales++] = s;
    }

    const float aspect_ratios[2] = { 1.15f, 1.0f };

    for (int s_idx = num_scales - 1; s_idx >= 0 && num_proposals < MAX_RAW_PROPOSALS; --s_idx) {
        int win_w = scales[s_idx];
        for (int a_idx = 0; a_idx < 2 && num_proposals < MAX_RAW_PROPOSALS; ++a_idx) {
            int win_h = (int)(win_w * aspect_ratios[a_idx]);
            if (win_h > height || win_w > width) continue;

            int stride_x = win_w / 10 > 5 ? win_w / 10 : 5;
            int stride_y = win_h / 10 > 5 ? win_h / 10 : 5;

            int max_y = height - win_h - 10;
            if (max_y > (int)(height * 0.72f)) max_y = (int)(height * 0.72f);

            for (int y = 10; y <= max_y && num_proposals < MAX_RAW_PROPOSALS; y += stride_y) {
                for (int x = 10; x + win_w <= width - 10 && num_proposals < MAX_RAW_PROPOSALS; x += stride_x) {
                    float tot_pix = (float)(win_w * win_h);

                    /* Stage 1: Illumination-Invariant Texture Filter */
                    float sum_lum = QUERY_LUM(x, y, win_w, win_h);
                    float mean_lum = sum_lum / tot_pix;
                    if (mean_lum < 25.0f) continue;

                    double sum_sq = QUERY_LUM_SQ(x, y, win_w, win_h);
                    float variance = (float)((sum_sq / (double)tot_pix) - ((double)mean_lum * (double)mean_lum));
                    if ((variance / (mean_lum * mean_lum)) < 0.035f) {
                        continue;
                    }

                    /* Stage 2: Skin Probability Filter */
                    int skin_count = QUERY_SKIN(x, y, win_w, win_h);
                    float skin_ratio = (float)skin_count / tot_pix;
                    if (skin_ratio < 0.38f) {
                        continue;
                    }

                    /* Upper & Lower Skin Symmetry */
                    int half_h = win_h / 2;
                    float upper_skin = (float)QUERY_SKIN(x, y, win_w, half_h) / (float)(win_w * half_h);
                    float lower_skin = (float)QUERY_SKIN(x, y + half_h, win_w, win_h - half_h) / (float)(win_w * (win_h - half_h));
                    if (upper_skin < 0.28f || lower_skin < 0.28f) {
                        continue;
                    }

                    /* Stage 3: Anthropometric Facial Bio-Geometry */
                    int fh_h = (int)(win_h * 0.20f);
                    int eye_y = y + fh_h;
                    int eye_h = (int)(win_h * 0.25f);
                    if (eye_h < 4) eye_h = 4;

                    int col_w = win_w / 5;
                    if (col_w < 4) col_w = 4;
                    float area_col = (float)(col_w * eye_h);

                    int cheek_y = eye_y + eye_h;
                    int cheek_h = (int)(win_h * 0.25f);
                    if (cheek_h < 4) cheek_h = 4;

                    /* 3 columns: Left Eye (c1), Nasal Bridge (c2), Right Eye (c3) */
                    float c1 = QUERY_LUM(x + col_w, eye_y, col_w, eye_h) / area_col;
                    float c2 = QUERY_LUM(x + 2 * col_w, eye_y, col_w, eye_h) / area_col;
                    float c3 = QUERY_LUM(x + 3 * col_w, eye_y, col_w, eye_h) / area_col;

                    float d1 = (c2 - c1) / mean_lum;
                    float d2 = (c2 - c3) / mean_lum;
                    float rel_bridge = d1 + d2;
                    if (rel_bridge < 0.05f) continue;
                    if (fmaxf(d1, d2) < 0.03f) continue;
                    if (fabsf(c1 - c3) / mean_lum > 0.45f) continue;

                    /* Forehead & Cheek Vertical Valley around Eye Band */
                    float eye_lum = (c1 + c2 + c3) / 3.0f;
                    float fh_lum = QUERY_LUM(x + col_w, y, 3 * col_w, fh_h) / (float)(3 * col_w * fh_h);
                    float cheek_lum = QUERY_LUM(x + col_w, cheek_y, 3 * col_w, cheek_h) / (float)(3 * col_w * cheek_h);

                    float v_fh = (fh_lum - eye_lum) / mean_lum;
                    float v_cheek = (cheek_lum - eye_lum) / mean_lum;
                    if (v_cheek < -0.15f) continue;

                    float score = (rel_bridge * 2.0f + v_fh + v_cheek + skin_ratio) * 100.0f;

                    proposals[num_proposals++] = (RawProposal){
                        .x = x,
                        .y = y,
                        .width = win_w,
                        .height = win_h,
                        .score = score
                    };
                }
            }
        }
    }

    #undef QUERY_LUM
    #undef QUERY_LUM_SQ
    #undef QUERY_SKIN

    if (num_proposals == 0) {
        scratch_arena->offset = saved_offset;
        return 0;
    }

    /* 3. Pre-Clustering Sort: prioritize high-confidence, large face windows */
    for (size_t i = 0; i < num_proposals; ++i) {
        for (size_t j = i + 1; j < num_proposals; ++j) {
            float rank_i = proposals[i].score * sqrtf((float)(proposals[i].width * proposals[i].height));
            float rank_j = proposals[j].score * sqrtf((float)(proposals[j].width * proposals[j].height));
            if (rank_j > rank_i) {
                RawProposal tmp = proposals[i];
                proposals[i] = proposals[j];
                proposals[j] = tmp;
            }
        }
    }

    /* 4. Density Clustering with Weighted Centroid Fusion */
    typedef struct {
        int x, y, width, height;
        float total_score;
        float mass;
        int count;
    } ResolvedCluster;

    ResolvedCluster clusters[MAX_CLUSTERS];
    size_t num_clusters = 0;

    bool* assigned = (bool*)rcre_arena_alloc(scratch_arena, num_proposals * sizeof(bool), 16);
    if (!assigned) {
        scratch_arena->offset = saved_offset;
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(assigned, 0, num_proposals * sizeof(bool));

    for (size_t i = 0; i < num_proposals && num_clusters < MAX_CLUSTERS; ++i) {
        if (assigned[i]) continue;
        assigned[i] = true;

        RCRE_FaceBox anchor_box = {
            .x = proposals[i].x,
            .y = proposals[i].y,
            .width = proposals[i].width,
            .height = proposals[i].height,
            .confidence = proposals[i].score
        };

        /* Anchor proposal gets 2x weight to anchor box stability */
        float sum_x = (float)proposals[i].x * proposals[i].score * 2.0f;
        float sum_y = (float)proposals[i].y * proposals[i].score * 2.0f;
        float sum_w = (float)proposals[i].width * proposals[i].score * 2.0f;
        float sum_h = (float)proposals[i].height * proposals[i].score * 2.0f;
        float sum_s = proposals[i].score * 2.0f;
        int votes = 1;

        for (size_t j = i + 1; j < num_proposals; ++j) {
            if (assigned[j]) continue;
            RCRE_FaceBox cand_box = {
                .x = proposals[j].x,
                .y = proposals[j].y,
                .width = proposals[j].width,
                .height = proposals[j].height,
                .confidence = proposals[j].score
            };

            if (compute_iou(&anchor_box, &cand_box) >= 0.30f) {
                assigned[j] = true;
                sum_x += (float)proposals[j].x * proposals[j].score;
                sum_y += (float)proposals[j].y * proposals[j].score;
                sum_w += (float)proposals[j].width * proposals[j].score;
                sum_h += (float)proposals[j].height * proposals[j].score;
                sum_s += proposals[j].score;
                votes++;
            }
        }

        int avg_x = (int)(sum_x / sum_s);
        int avg_y = (int)(sum_y / sum_s);
        int avg_w = (int)(sum_w / sum_s);
        int avg_h = (int)(sum_h / sum_s);
        float mass = (float)votes * sqrtf((float)(avg_w * avg_h));

        clusters[num_clusters++] = (ResolvedCluster){
            .x = avg_x,
            .y = avg_y,
            .width = avg_w,
            .height = avg_h,
            .total_score = sum_s,
            .mass = mass,
            .count = votes
        };
    }

    /* 5. Subsumption: Suppress smaller sub-boxes inside larger faces */
    bool suppressed[MAX_CLUSTERS];
    memset(suppressed, 0, sizeof(suppressed));

    for (size_t i = 0; i < num_clusters; ++i) {
        for (size_t j = 0; j < num_clusters; ++j) {
            if (i == j || suppressed[j]) continue;
            int area_i = clusters[i].width * clusters[i].height;
            int area_j = clusters[j].width * clusters[j].height;

            if (area_j > (int)((float)area_i * 1.3f)) {
                int x1 = clusters[i].x > clusters[j].x ? clusters[i].x : clusters[j].x;
                int y1 = clusters[i].y > clusters[j].y ? clusters[i].y : clusters[j].y;
                int x2 = (clusters[i].x + clusters[i].width) < (clusters[j].x + clusters[j].width) ?
                         (clusters[i].x + clusters[i].width) : (clusters[j].x + clusters[j].width);
                int y2 = (clusters[i].y + clusters[i].height) < (clusters[j].y + clusters[j].height) ?
                         (clusters[i].y + clusters[i].height) : (clusters[j].y + clusters[j].height);

                int inter_w = x2 - x1;
                int inter_h = y2 - y1;
                if (inter_w > 0 && inter_h > 0) {
                    float inter_area = (float)(inter_w * inter_h);
                    if (inter_area >= 0.50f * (float)area_i) {
                        suppressed[i] = true;
                        break;
                    }
                }
            }
        }
    }

    /* 6. Sort valid clusters descending by mass */
    for (size_t i = 0; i < num_clusters; ++i) {
        for (size_t j = i + 1; j < num_clusters; ++j) {
            if (clusters[j].mass > clusters[i].mass) {
                ResolvedCluster tmp = clusters[i];
                clusters[i] = clusters[j];
                clusters[j] = tmp;

                bool sup_tmp = suppressed[i];
                suppressed[i] = suppressed[j];
                suppressed[j] = sup_tmp;
            }
        }
    }

    size_t final_count = 0;
    for (size_t i = 0; i < num_clusters && final_count < max_faces; ++i) {
        if (suppressed[i] || clusters[i].count < 3) continue;

        float conf = 0.70f + (float)clusters[i].count * 0.01f + (clusters[i].total_score / 25000.0f);
        if (conf > 0.99f) conf = 0.99f;

        out_faces[final_count++] = (RCRE_FaceBox){
            .x = clusters[i].x,
            .y = clusters[i].y,
            .width = clusters[i].width,
            .height = clusters[i].height,
            .confidence = conf
        };
    }

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
