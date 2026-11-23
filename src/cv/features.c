/**
 * @file features.c
 * @brief Harris Corner Detection, Patch Descriptors, and Feature Matching
 *
 * Implements structure tensor computation, cornerness response, 2D NMS,
 * local gradient patch descriptors, and ratio-tested feature matching.
 */

#include "rcre.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline int feat_clamp_int(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline float feat_get_pixel_f32(const RCRE_Tensor* t, int x, int y, int c) {
    x = feat_clamp_int(x, 0, (int)t->shape[1] - 1);
    y = feat_clamp_int(y, 0, (int)t->shape[0] - 1);
    int num_c = (t->ndim == 3) ? (int)t->shape[2] : 1;
    c = feat_clamp_int(c, 0, num_c - 1);

    const float* data = (const float*)t->data;
    if (t->ndim == 2) {
        return data[y * (int)t->strides[0] + x * (int)t->strides[1]];
    }
    return data[y * (int)t->strides[0] + x * (int)t->strides[1] + c];
}

RCRE_NODISCARD int rcre_cv_harris_response(const RCRE_Tensor* src, RCRE_Tensor* dst_r,
                                          float sigma, float k, RCRE_Arena* arena) {
    if (!src || !src->data || !dst_r || !dst_r->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst_r->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];

    if (dst_r->shape[0] != height || dst_r->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);

    /* Step 1: Ensure grayscale single-channel source for derivatives */
    size_t shape_2d[2] = {height, width};
    RCRE_Tensor gray = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!gray.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    int status = RCRE_OK;
    if (src->ndim == 3 && src->shape[2] >= 3) {
        status = rcre_cv_rgb_to_gray(src, &gray);
        if (status != RCRE_OK) {
            rcre_arena_temp_end(scratch);
            return status;
        }
    } else {
        /* Copy single channel */
        const float* src_data = (const float*)src->data;
        float* gray_data = (float*)gray.data;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                gray_data[y * width + x] = src_data[y * src->strides[0] + x * src->strides[1]];
            }
        }
    }

    /* Step 2: Sobel derivatives dx and dy */
    RCRE_Tensor dx = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor dy = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!dx.data || !dy.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    status = rcre_cv_sobel(&gray, &dx, &dy, arena);
    if (status != RCRE_OK) {
        rcre_arena_temp_end(scratch);
        return status;
    }

    /* Step 3: Compute structure tensor components Sxx, Syy, Sxy */
    size_t shape_3d[3] = {height, width, 3};
    RCRE_Tensor s_raw = rcre_tensor_create(shape_3d, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!s_raw.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    const float* dx_data = (const float*)dx.data;
    const float* dy_data = (const float*)dy.data;
    float* s_raw_data = (float*)s_raw.data;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float ix = dx_data[y * width + x];
            float iy = dy_data[y * width + x];
            size_t idx = (y * width + x) * 3;
            s_raw_data[idx + 0] = ix * ix;     /* Ixx */
            s_raw_data[idx + 1] = iy * iy;     /* Iyy */
            s_raw_data[idx + 2] = ix * iy;     /* Ixy */
        }
    }

    /* Step 4: Gaussian smooth the structure tensor */
    RCRE_Tensor s_smooth = rcre_tensor_create(shape_3d, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!s_smooth.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    float eff_sigma = (sigma <= 0.0f) ? 1.5f : sigma;
    status = rcre_cv_gaussian_blur(&s_raw, &s_smooth, eff_sigma, arena);
    if (status != RCRE_OK) {
        rcre_arena_temp_end(scratch);
        return status;
    }

    /* Step 5: Cornerness response R = det(S) - k * trace(S)^2 */
    const float* s_data = (const float*)s_smooth.data;
    float* r_data = (float*)dst_r->data;
    float eff_k = (k <= 0.0f) ? 0.06f : k;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t s_idx = (y * width + x) * 3;
            float ixx = s_data[s_idx + 0];
            float iyy = s_data[s_idx + 1];
            float ixy = s_data[s_idx + 2];

            float det = ixx * iyy - ixy * ixy;
            float trace = ixx + iyy;
            float r_val = det - eff_k * trace * trace;

            if (dst_r->ndim == 2) {
                r_data[y * dst_r->strides[0] + x * dst_r->strides[1]] = r_val;
            } else {
                r_data[y * dst_r->strides[0] + x * dst_r->strides[1] + 0] = r_val;
            }
        }
    }

    rcre_arena_temp_end(scratch);
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_nms_2d(const RCRE_Tensor* src, RCRE_Tensor* dst,
                                 int window_radius, float threshold) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }
    if (src->shape[0] != dst->shape[0] || src->shape[1] != dst->shape[1]) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];
    int r = (window_radius < 1) ? 1 : window_radius;

    const float* in = (const float*)src->data;
    float* out = (float*)dst->data;

    int src_sy = (int)src->strides[0];
    int src_sx = (int)src->strides[1];
    int dst_sy = (int)dst->strides[0];
    int dst_sx = (int)dst->strides[1];

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float val = in[y * src_sy + x * src_sx];
            int is_max = 1;

            if (val <= threshold) {
                is_max = 0;
            } else {
                for (int dy = -r; dy <= r && is_max; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int ny = (int)y + dy;
                        int nx = (int)x + dx;
                        if (ny >= 0 && ny < (int)height && nx >= 0 && nx < (int)width) {
                            if (in[ny * src_sy + nx * src_sx] > val) {
                                is_max = 0;
                                break;
                            }
                        }
                    }
                }
            }

            if (is_max) {
                out[y * dst_sy + x * dst_sx] = val;
            } else {
                out[y * dst_sy + x * dst_sx] = 0.0f;
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_harris_detect(const RCRE_Tensor* src, float sigma, float threshold,
                                        int nms_radius, RCRE_CV_Descriptor** out_desc,
                                        int* out_count, RCRE_Arena* arena) {
    if (!src || !src->data || !out_desc || !out_count || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];
    size_t channels = (src->ndim == 3) ? src->shape[2] : 1;

    size_t shape_2d[2] = {height, width};
    RCRE_Tensor resp = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor nms = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!resp.data || !nms.data) return RCRE_ERR_OUT_OF_MEMORY;

    int status = rcre_cv_harris_response(src, &resp, sigma, 0.06f, arena);
    if (status != RCRE_OK) return status;

    int eff_nms = (nms_radius < 1) ? 2 : nms_radius;
    status = rcre_cv_nms_2d(&resp, &nms, eff_nms, threshold);
    if (status != RCRE_OK) return status;

    /* Count detected keypoints */
    const float* nms_data = (const float*)nms.data;
    int count = 0;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            if (nms_data[y * nms.strides[0] + x * nms.strides[1]] > threshold) {
                count++;
            }
        }
    }

    *out_count = count;
    if (count == 0) {
        *out_desc = NULL;
        return RCRE_OK;
    }

    RCRE_CV_Descriptor* desc_array = (RCRE_CV_Descriptor*)rcre_arena_alloc(arena, (size_t)count * sizeof(RCRE_CV_Descriptor), 64);
    if (!desc_array) return RCRE_ERR_OUT_OF_MEMORY;

    /* Extract 5x5 local intensity gradient patch descriptors */
    const int pw = 5;
    int r = pw / 2;
    int desc_len = pw * pw * (int)channels;

    int idx = 0;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            if (nms_data[y * nms.strides[0] + x * nms.strides[1]] > threshold) {
                desc_array[idx].pt.x = (float)x;
                desc_array[idx].pt.y = (float)y;
                desc_array[idx].len = desc_len;
                desc_array[idx].data = (float*)rcre_arena_alloc(arena, (size_t)desc_len * sizeof(float), 64);
                if (!desc_array[idx].data) return RCRE_ERR_OUT_OF_MEMORY;

                int p_idx = 0;
                for (size_t c = 0; c < channels; ++c) {
                    float center_val = feat_get_pixel_f32(src, (int)x, (int)y, (int)c);
                    for (int dy = -r; dy <= r; ++dy) {
                        for (int dx = -r; dx <= r; ++dx) {
                            float neighbor_val = feat_get_pixel_f32(src, (int)x + dx, (int)y + dy, (int)c);
                            desc_array[idx].data[p_idx++] = center_val - neighbor_val;
                        }
                    }
                }
                idx++;
            }
        }
    }

    *out_desc = desc_array;
    return RCRE_OK;
}

static inline float feat_descriptor_dist_l1(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum;
}

static int feat_compare_matches(const void* a, const void* b) {
    const RCRE_CV_Match* ma = (const RCRE_CV_Match*)a;
    const RCRE_CV_Match* mb = (const RCRE_CV_Match*)b;
    if (ma->distance < mb->distance) return -1;
    if (ma->distance > mb->distance) return 1;
    return 0;
}

RCRE_NODISCARD int rcre_cv_match_descriptors(const RCRE_CV_Descriptor* a, int na,
                                            const RCRE_CV_Descriptor* b, int nb,
                                            float ratio_thresh, RCRE_CV_Match** out_matches,
                                            int* out_count, RCRE_Arena* arena) {
    if (!a || !b || !out_matches || !out_count || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (na <= 0 || nb <= 0) {
        *out_count = 0;
        *out_matches = NULL;
        return RCRE_OK;
    }

    RCRE_CV_Match* raw_matches = (RCRE_CV_Match*)rcre_arena_alloc(arena, (size_t)na * sizeof(RCRE_CV_Match), 64);
    if (!raw_matches) return RCRE_ERR_OUT_OF_MEMORY;

    int candidate_count = 0;
    float eff_ratio = (ratio_thresh <= 0.0f) ? 0.90f : ratio_thresh;

    for (int j = 0; j < na; ++j) {
        int best_idx = -1;
        float best_dist = 1e9f;
        float second_dist = 1e9f;

        for (int k = 0; k < nb; ++k) {
            int len = (a[j].len < b[k].len) ? a[j].len : b[k].len;
            float dist = feat_descriptor_dist_l1(a[j].data, b[k].data, len);

            if (dist < best_dist) {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = k;
            } else if (dist < second_dist) {
                second_dist = dist;
            }
        }

        /* Lowe's ratio test */
        if (nb > 1 && best_dist > eff_ratio * second_dist) {
            continue;
        }

        if (best_idx >= 0) {
            raw_matches[candidate_count].idx_a = j;
            raw_matches[candidate_count].idx_b = best_idx;
            raw_matches[candidate_count].p = a[j].pt;
            raw_matches[candidate_count].q = b[best_idx].pt;
            raw_matches[candidate_count].distance = best_dist;
            candidate_count++;
        }
    }

    if (candidate_count == 0) {
        *out_count = 0;
        *out_matches = NULL;
        return RCRE_OK;
    }

    qsort(raw_matches, (size_t)candidate_count, sizeof(RCRE_CV_Match), feat_compare_matches);

    /* Enforce 1-to-1 uniqueness on index_b */
    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);
    uint8_t* seen = (uint8_t*)rcre_arena_alloc(arena, (size_t)nb * sizeof(uint8_t), 16);
    if (!seen) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(seen, 0, (size_t)nb * sizeof(uint8_t));

    int unique_count = 0;
    for (int i = 0; i < candidate_count; ++i) {
        int bi = raw_matches[i].idx_b;
        if (!seen[bi]) {
            seen[bi] = 1;
            raw_matches[unique_count++] = raw_matches[i];
        }
    }
    rcre_arena_temp_end(scratch);

    *out_count = unique_count;
    *out_matches = raw_matches;
    return RCRE_OK;
}

void rcre_cv_draw_markers(RCRE_Tensor* image, const RCRE_CV_Point* points, int count,
                          float r, float g, float b) {
    if (!image || !image->data || !points || count <= 0) return;
    if (image->dtype != RCRE_DTYPE_F32 || image->ndim < 2) return;

    size_t height = image->shape[0];
    size_t width = image->shape[1];
    size_t channels = (image->ndim == 3) ? image->shape[2] : 1;

    float* data = (float*)image->data;
    int sy = (int)image->strides[0];
    int sx = (int)image->strides[1];

    for (int k = 0; k < count; ++k) {
        int cx = (int)roundf(points[k].x);
        int cy = (int)roundf(points[k].y);

        for (int d = -4; d <= 4; ++d) {
            /* Horizontal arm */
            int x1 = cx + d;
            int y1 = cy;
            if (x1 >= 0 && x1 < (int)width && y1 >= 0 && y1 < (int)height) {
                if (channels >= 3) {
                    data[y1 * sy + x1 * sx + 0] = r;
                    data[y1 * sy + x1 * sx + 1] = g;
                    data[y1 * sy + x1 * sx + 2] = b;
                } else {
                    data[y1 * sy + x1 * sx] = (r + g + b) / 3.0f;
                }
            }

            /* Vertical arm */
            int x2 = cx;
            int y2 = cy + d;
            if (x2 >= 0 && x2 < (int)width && y2 >= 0 && y2 < (int)height) {
                if (channels >= 3) {
                    data[y2 * sy + x2 * sx + 0] = r;
                    data[y2 * sy + x2 * sx + 1] = g;
                    data[y2 * sy + x2 * sx + 2] = b;
                } else {
                    data[y2 * sy + x2 * sx] = (r + g + b) / 3.0f;
                }
            }
        }
    }
}

/* ========================================================================= */
/* FAST-9 Corner Detector & Rotated BRIEF (ORB) Binary Descriptors           */
/* ========================================================================= */

static const int feat_fast_dx[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
static const int feat_fast_dy[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

static int feat_cmp_orb_desc(const void* a, const void* b) {
    const RCRE_ORBKeypoint* ka = (const RCRE_ORBKeypoint*)a;
    const RCRE_ORBKeypoint* kb = (const RCRE_ORBKeypoint*)b;
    if (kb->score > ka->score) return 1;
    if (kb->score < ka->score) return -1;
    return 0;
}

RCRE_NODISCARD int rcre_cv_fast_detect(const RCRE_Tensor* gray_f32, float threshold,
                                       int max_keypoints, RCRE_ORBKeypoint** out_kps,
                                       int* out_count, RCRE_Arena* arena) {
    if (!gray_f32 || !gray_f32->data || !out_kps || !out_count || !arena ||
        gray_f32->dtype != RCRE_DTYPE_F32 || gray_f32->ndim < 2) {
        return RCRE_ERR_INVALID_ARG;
    }
    int H = (int)gray_f32->shape[0];
    int W = (int)gray_f32->shape[1];
    if (H < 12 || W < 12) {
        *out_kps = nullptr;
        *out_count = 0;
        return RCRE_OK;
    }
    const float* d = (const float*)gray_f32->data;
    int sy = (int)gray_f32->strides[0];
    int sx = (int)gray_f32->strides[1];

    int cap = (max_keypoints > 0 && max_keypoints < 4096) ? 4096 : (max_keypoints > 0 ? max_keypoints : 2048);
    RCRE_ORBKeypoint* cands = (RCRE_ORBKeypoint*)rcre_arena_alloc(arena, (size_t)cap * sizeof(RCRE_ORBKeypoint), 16);
    if (!cands) return RCRE_ERR_OUT_OF_MEMORY;

    int count = 0;
    for (int y = 5; y < H - 5 && count < cap; ++y) {
        for (int x = 5; x < W - 5 && count < cap; ++x) {
            float center = d[y * sy + x * sx];
            /* Quick 4-cardinal rejection check (0, 4, 8, 12) */
            float c0  = d[(y + feat_fast_dy[0]) * sy + (x + feat_fast_dx[0]) * sx];
            float c4  = d[(y + feat_fast_dy[4]) * sy + (x + feat_fast_dx[4]) * sx];
            float c8  = d[(y + feat_fast_dy[8]) * sy + (x + feat_fast_dx[8]) * sx];
            float c12 = d[(y + feat_fast_dy[12]) * sy + (x + feat_fast_dx[12]) * sx];
            int nb = (c0 > center + threshold) + (c4 > center + threshold) +
                     (c8 > center + threshold) + (c12 > center + threshold);
            int nd = (c0 < center - threshold) + (c4 < center - threshold) +
                     (c8 < center - threshold) + (c12 < center - threshold);
            if (nb < 2 && nd < 2) continue;

            int run_b = 0, run_d = 0, max_b = 0, max_d = 0;
            float score_sum = 0.0f;
            for (int k = 0; k < 25; ++k) { /* wrap around ring of 16 + 9 */
                int idx = k & 15;
                float val = d[(y + feat_fast_dy[idx]) * sy + (x + feat_fast_dx[idx]) * sx];
                if (val > center + threshold) {
                    run_b++;
                    run_d = 0;
                    if (run_b > max_b) max_b = run_b;
                    score_sum += val - center;
                } else if (val < center - threshold) {
                    run_d++;
                    run_b = 0;
                    if (run_d > max_d) max_d = run_d;
                    score_sum += center - val;
                } else {
                    run_b = 0;
                    run_d = 0;
                }
            }
            if (max_b >= 9 || max_d >= 9) {
                /* Intensity centroid orientation inside radius 5 */
                float m10 = 0.0f, m01 = 0.0f;
                for (int dy = -5; dy <= 5; ++dy) {
                    for (int dx = -5; dx <= 5; ++dx) {
                        if (dx * dx + dy * dy <= 25) {
                            float pv = d[(y + dy) * sy + (x + dx) * sx];
                            m10 += (float)dx * pv;
                            m01 += (float)dy * pv;
                        }
                    }
                }
                RCRE_ORBKeypoint kp;
                memset(&kp, 0, sizeof(kp));
                kp.x = (float)x;
                kp.y = (float)y;
                kp.angle = atan2f(m01, m10);
                kp.score = (uint32_t)(score_sum * 100.0f);
                cands[count++] = kp;
            }
        }
    }

    if (count > 1) {
        qsort(cands, (size_t)count, sizeof(RCRE_ORBKeypoint), feat_cmp_orb_desc);
    }
    if (max_keypoints > 0 && count > max_keypoints) count = max_keypoints;
    *out_kps = cands;
    *out_count = count;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_orb_extract(const RCRE_Tensor* gray_f32, RCRE_ORBKeypoint* kps, int count) {
    if (!gray_f32 || !gray_f32->data || !kps || count < 0 ||
        gray_f32->dtype != RCRE_DTYPE_F32 || gray_f32->ndim < 2) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (count == 0) return RCRE_OK;
    int H = (int)gray_f32->shape[0];
    int W = (int)gray_f32->shape[1];
    const float* d = (const float*)gray_f32->data;
    int sy = (int)gray_f32->strides[0];
    int sx = (int)gray_f32->strides[1];

    /* Deterministic 256 rotated-BRIEF test pairs inside [-11, 11] patch */
    uint32_t seed = 0x517CC1B7U;
    int8_t pat[256][4];
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 4; ++j) {
            seed = seed * 1664525U + 1013904223U;
            pat[i][j] = (int8_t)(((int)(seed >> 24) % 23) - 11);
        }
    }

    for (int k = 0; k < count; ++k) {
        float ca = cosf(kps[k].angle);
        float sa = sinf(kps[k].angle);
        float cx = kps[k].x, cy = kps[k].y;
        memset(kps[k].bits, 0, sizeof(kps[k].bits));

        for (int b = 0; b < 256; ++b) {
            int x1 = feat_clamp_int((int)lrintf(cx + pat[b][0] * ca - pat[b][1] * sa), 0, W - 1);
            int y1 = feat_clamp_int((int)lrintf(cy + pat[b][0] * sa + pat[b][1] * ca), 0, H - 1);
            int x2 = feat_clamp_int((int)lrintf(cx + pat[b][2] * ca - pat[b][3] * sa), 0, W - 1);
            int y2 = feat_clamp_int((int)lrintf(cy + pat[b][2] * sa + pat[b][3] * ca), 0, H - 1);
            if (d[y1 * sy + x1 * sx] < d[y2 * sy + x2 * sx]) {
                kps[k].bits[b >> 5] |= (1U << (b & 31));
            }
        }
    }
    return RCRE_OK;
}

static inline int feat_popcnt32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int c = 0;
    while (x) { x &= x - 1U; c++; }
    return c;
#endif
}

RCRE_NODISCARD int rcre_cv_orb_match(const RCRE_ORBKeypoint* a, int na,
                                     const RCRE_ORBKeypoint* b, int nb,
                                     int max_hamming, RCRE_CV_Match** out_matches,
                                     int* out_count, RCRE_Arena* arena) {
    if (!a || !b || !out_matches || !out_count || !arena || na <= 0 || nb <= 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    RCRE_CV_Match* m = (RCRE_CV_Match*)rcre_arena_alloc(arena, (size_t)na * sizeof(RCRE_CV_Match), 16);
    if (!m) return RCRE_ERR_OUT_OF_MEMORY;

    int mc = 0;
    for (int i = 0; i < na; ++i) {
        int best_dist = 257;
        int best_j = -1;
        for (int j = 0; j < nb; ++j) {
            int dist = 0;
            for (int w = 0; w < 8; ++w) dist += feat_popcnt32(a[i].bits[w] ^ b[j].bits[w]);
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }
        if (best_j >= 0 && best_dist <= max_hamming) {
            m[mc++] = (RCRE_CV_Match){
                .p        = { a[i].x, a[i].y },
                .q        = { b[best_j].x, b[best_j].y },
                .idx_a    = i,
                .idx_b    = best_j,
                .distance = (float)best_dist
            };
        }
    }
    *out_matches = m;
    *out_count = mc;
    return RCRE_OK;
}
