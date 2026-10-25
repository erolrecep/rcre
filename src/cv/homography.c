/**
 * @file homography.c
 * @brief Direct Linear Transformation (DLT), RANSAC, Perspective Warping & Panorama Stitching
 *
 * Implements linear system solving for 8-DOF projective homographies,
 * robust consensus matching, cylindrical projection, and image stitching.
 */

#include "rcre.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline float homo_point_dist_l2(RCRE_CV_Point p, RCRE_CV_Point q) {
    float dx = p.x - q.x;
    float dy = p.y - q.y;
    return sqrtf(dx * dx + dy * dy);
}

static inline RCRE_CV_Point homo_project_point_h(const float H[3][3], RCRE_CV_Point p) {
    float x = H[0][0] * p.x + H[0][1] * p.y + H[0][2];
    float y = H[1][0] * p.x + H[1][1] * p.y + H[1][2];
    float w = H[2][0] * p.x + H[2][1] * p.y + H[2][2];

    RCRE_CV_Point q = {0.0f, 0.0f};
    if (fabsf(w) > 1e-7f) {
        q.x = x / w;
        q.y = y / w;
    }
    return q;
}

static inline bool homo_invert_3x3(const float m[3][3], float inv[3][3]) {
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (fabsf(det) < 1e-8f) return false;
    float inv_det = 1.0f / det;

    inv[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
    inv[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

    inv[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
    inv[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;

    inv[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
    inv[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    return true;
}

/* Solves 8x8 linear system A * x = b via Gauss-Jordan elimination with partial pivoting */
static bool homo_solve_8x8(const float A[8][8], const float b[8], float x[8]) {
    float M[8][9];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            M[i][j] = A[i][j];
        }
        M[i][8] = b[i];
    }

    for (int k = 0; k < 8; ++k) {
        /* Pivot selection */
        int max_row = k;
        float max_val = fabsf(M[k][k]);
        for (int i = k + 1; i < 8; ++i) {
            float val = fabsf(M[i][k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        if (max_val < 1e-9f) return false;

        /* Swap rows */
        if (max_row != k) {
            for (int j = k; j < 9; ++j) {
                float tmp = M[k][j];
                M[k][j] = M[max_row][j];
                M[max_row][j] = tmp;
            }
        }

        /* Normalize row k */
        float pivot = M[k][k];
        for (int j = k; j < 9; ++j) {
            M[k][j] /= pivot;
        }

        /* Eliminate column k in other rows */
        for (int i = 0; i < 8; ++i) {
            if (i == k) continue;
            float factor = M[i][k];
            for (int j = k; j < 9; ++j) {
                M[i][j] -= factor * M[k][j];
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        x[i] = M[i][8];
    }
    return true;
}

RCRE_NODISCARD int rcre_cv_compute_homography_dlt(const RCRE_CV_Match* matches, int n,
                                                 float H[3][3], RCRE_Arena* arena) {
    (void)arena;
    if (!matches || n < 4 || !H) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* Normal equations: MtM * a = Mtb where M is 2n x 8 and b is 2n x 1 */
    float MtM[8][8] = {0};
    float Mtb[8] = {0};

    for (int i = 0; i < n; ++i) {
        float x = matches[i].p.x;
        float y = matches[i].p.y;
        float xp = matches[i].q.x;
        float yp = matches[i].q.y;

        /* Row 1 of M: [x, y, 1, 0, 0, 0, -x*xp, -y*xp], rhs: xp */
        float r1[8] = {x, y, 1.0f, 0.0f, 0.0f, 0.0f, -x * xp, -y * xp};
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                MtM[r][c] += r1[r] * r1[c];
            }
            Mtb[r] += r1[r] * xp;
        }

        /* Row 2 of M: [0, 0, 0, x, y, 1, -x*yp, -y*yp], rhs: yp */
        float r2[8] = {0.0f, 0.0f, 0.0f, x, y, 1.0f, -x * yp, -y * yp};
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                MtM[r][c] += r2[r] * r2[c];
            }
            Mtb[r] += r2[r] * yp;
        }
    }

    float a[8];
    if (!homo_solve_8x8(MtM, Mtb, a)) {
        /* Fallback to identity */
        memset(H, 0, 9 * sizeof(float));
        H[0][0] = 1.0f; H[1][1] = 1.0f; H[2][2] = 1.0f;
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    H[0][0] = a[0]; H[0][1] = a[1]; H[0][2] = a[2];
    H[1][0] = a[3]; H[1][1] = a[4]; H[1][2] = a[5];
    H[2][0] = a[6]; H[2][1] = a[7]; H[2][2] = 1.0f;
    return RCRE_OK;
}

static int homo_count_inliers(const float H[3][3], const RCRE_CV_Match* matches, int n,
                              float thresh, RCRE_CV_Match* inliers_out) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        RCRE_CV_Point p_proj = homo_project_point_h(H, matches[i].p);
        float dist = homo_point_dist_l2(p_proj, matches[i].q);
        if (dist < thresh) {
            if (inliers_out) {
                inliers_out[count] = matches[i];
            }
            count++;
        }
    }
    return count;
}

RCRE_NODISCARD int rcre_cv_find_homography_ransac(RCRE_CV_Match* matches, int n,
                                                 float inlier_thresh, int iters, int cutoff,
                                                 float out_H[3][3], int* out_inliers,
                                                 RCRE_Arena* arena) {
    if (!matches || !out_H || !out_inliers || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* Default identity */
    memset(out_H, 0, 9 * sizeof(float));
    out_H[0][0] = 1.0f; out_H[1][1] = 1.0f; out_H[2][2] = 1.0f;
    *out_inliers = 0;

    if (n < 4) return RCRE_OK;

    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);
    RCRE_CV_Match* inliers_buf = (RCRE_CV_Match*)rcre_arena_alloc(arena, (size_t)n * sizeof(RCRE_CV_Match), 64);
    if (!inliers_buf) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    int best_inliers = 0;
    float best_H[3][3];
    memcpy(best_H, out_H, 9 * sizeof(float));

    int num_iters = (iters <= 0) ? 1000 : iters;
    float thresh = (inlier_thresh <= 0.0f) ? 3.0f : inlier_thresh;

    for (int iter = 0; iter < num_iters; ++iter) {
        /* Sample 4 distinct random points */
        RCRE_CV_Match sample[4];
        int idxs[4];
        for (int s = 0; s < 4; ++s) {
            bool unique;
            int r;
            do {
                unique = true;
                r = rand() % n;
                for (int p = 0; p < s; ++p) {
                    if (idxs[p] == r) {
                        unique = false;
                        break;
                    }
                }
            } while (!unique);
            idxs[s] = r;
            sample[s] = matches[r];
        }

        float H_cand[3][3];
        if (rcre_cv_compute_homography_dlt(sample, 4, H_cand, arena) != RCRE_OK) {
            continue;
        }

        int inliers = homo_count_inliers(H_cand, matches, n, thresh, inliers_buf);
        if (inliers > best_inliers) {
            best_inliers = inliers;
            /* Refine homography with all current inliers */
            if (inliers >= 4) {
                (void)rcre_cv_compute_homography_dlt(inliers_buf, inliers, best_H, arena);
            } else {
                memcpy(best_H, H_cand, 9 * sizeof(float));
            }

            if (cutoff > 0 && best_inliers >= cutoff) {
                break;
            }
        }
    }

    memcpy(out_H, best_H, 9 * sizeof(float));
    *out_inliers = best_inliers;
    rcre_arena_temp_end(scratch);
    return RCRE_OK;
}

static inline float homo_sample_bilinear_src(const RCRE_Tensor* src, float x, float y, size_t c) {
    size_t w = src->shape[1];
    size_t h = src->shape[0];

    int x1 = (int)floorf(x);
    int x2 = x1 + 1;
    int y1 = (int)floorf(y);
    int y2 = y1 + 1;

    float d1 = x - (float)x1;
    float d2 = 1.0f - d1;
    float d3 = y - (float)y1;
    float d4 = 1.0f - d3;

    int cx1 = (x1 < 0) ? 0 : (x1 >= (int)w ? (int)w - 1 : x1);
    int cx2 = (x2 < 0) ? 0 : (x2 >= (int)w ? (int)w - 1 : x2);
    int cy1 = (y1 < 0) ? 0 : (y1 >= (int)h ? (int)h - 1 : y1);
    int cy2 = (y2 < 0) ? 0 : (y2 >= (int)h ? (int)h - 1 : y2);

    const float* in = (const float*)src->data;
    int sy = (int)src->strides[0];
    int sx = (int)src->strides[1];

    float v00 = in[cy1 * sy + cx1 * sx + c];
    float v10 = in[cy1 * sy + cx2 * sx + c];
    float v01 = in[cy2 * sy + cx1 * sx + c];
    float v11 = in[cy2 * sy + cx2 * sx + c];

    return d2 * d4 * v00 + d1 * d4 * v10 + d2 * d3 * v01 + d1 * d3 * v11;
}

RCRE_NODISCARD int rcre_cv_warp_perspective(const RCRE_Tensor* src, const float H[3][3],
                                           RCRE_Tensor* dst, RCRE_Arena* arena) {
    (void)arena;
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    float Hinv[3][3];
    if (!homo_invert_3x3(H, Hinv)) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t in_h = src->shape[0];
    size_t in_w = src->shape[1];
    size_t in_c = (src->ndim == 3) ? src->shape[2] : 1;

    size_t out_h = dst->shape[0];
    size_t out_w = dst->shape[1];
    size_t out_c = (dst->ndim == 3) ? dst->shape[2] : 1;

    if (in_c != out_c) return RCRE_ERR_SHAPE_MISMATCH;

    float* out = (float*)dst->data;
    int dst_sy = (int)dst->strides[0];
    int dst_sx = (int)dst->strides[1];

    for (size_t y = 0; y < out_h; ++y) {
        for (size_t x = 0; x < out_w; ++x) {
            RCRE_CV_Point p = {(float)x, (float)y};
            RCRE_CV_Point src_p = homo_project_point_h(Hinv, p);

            if (src_p.x >= 0.0f && src_p.x < (float)in_w &&
                src_p.y >= 0.0f && src_p.y < (float)in_h) {
                for (size_t c = 0; c < in_c; ++c) {
                    float val = homo_sample_bilinear_src(src, src_p.x, src_p.y, c);
                    out[y * dst_sy + x * dst_sx + c] = val;
                }
            } else {
                for (size_t c = 0; c < in_c; ++c) {
                    out[y * dst_sy + x * dst_sx + c] = 0.0f;
                }
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_cylindrical_project(const RCRE_Tensor* src, float f,
                                              RCRE_Tensor* dst, RCRE_Arena* arena) {
    (void)arena;
    if (!src || !src->data || !dst || !dst->data || f <= 0.0f) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t h = src->shape[0];
    size_t w = src->shape[1];
    size_t c = (src->ndim == 3) ? src->shape[2] : 1;

    if (dst->shape[0] != h || dst->shape[1] != w) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    float* out = (float*)dst->data;
    int dst_sy = (int)dst->strides[0];
    int dst_sx = (int)dst->strides[1];

    float cx = (float)w / 2.0f;
    float cy = (float)h / 2.0f;

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float theta = ((float)x - cx) / f;
            float h_ratio = ((float)y - cy) / f;

            float X = sinf(theta);
            float Y = h_ratio;
            float Z = cosf(theta);

            float src_x = f * (X / Z) + cx;
            float src_y = f * (Y / Z) + cy;

            if (src_x >= 0.0f && src_x < (float)w && src_y >= 0.0f && src_y < (float)h) {
                for (size_t ch = 0; ch < c; ++ch) {
                    out[y * dst_sy + x * dst_sx + ch] = homo_sample_bilinear_src(src, src_x, src_y, ch);
                }
            } else {
                for (size_t ch = 0; ch < c; ++ch) {
                    out[y * dst_sy + x * dst_sx + ch] = 0.0f;
                }
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_stitch_panorama(const RCRE_Tensor* a, const RCRE_Tensor* b,
                                          float f_cylindrical, float sigma, float thresh,
                                          int nms, float inlier_thresh, int iters,
                                          RCRE_Tensor* dst, RCRE_Arena* arena) {
    if (!a || !b || !dst || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);

    /* Optional cylindrical projection */
    const RCRE_Tensor* img_a = a;
    const RCRE_Tensor* img_b = b;
    RCRE_Tensor proj_a, proj_b;

    if (f_cylindrical > 0.0f) {
        proj_a = rcre_tensor_create(a->shape, a->ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
        proj_b = rcre_tensor_create(b->shape, b->ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
        if (!proj_a.data || !proj_b.data) {
            rcre_arena_temp_end(scratch);
            return RCRE_ERR_OUT_OF_MEMORY;
        }
        (void)rcre_cv_cylindrical_project(a, f_cylindrical, &proj_a, arena);
        (void)rcre_cv_cylindrical_project(b, f_cylindrical, &proj_b, arena);
        img_a = &proj_a;
        img_b = &proj_b;
    }

    /* Feature detection */
    RCRE_CV_Descriptor *da = NULL, *db = NULL;
    int na = 0, nb = 0;

    (void)rcre_cv_harris_detect(img_a, sigma, thresh, nms, &da, &na, arena);
    (void)rcre_cv_harris_detect(img_b, sigma, thresh, nms, &db, &nb, arena);

    if (na < 4 || nb < 4) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_INVALID_ARG;
    }

    /* Matching */
    RCRE_CV_Match* matches = NULL;
    int num_matches = 0;
    (void)rcre_cv_match_descriptors(da, na, db, nb, 0.90f, &matches, &num_matches, arena);

    if (num_matches < 4) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_INVALID_ARG;
    }

    /* RANSAC homography estimation */
    float H[3][3];
    int inliers = 0;
    (void)rcre_cv_find_homography_ransac(matches, num_matches, inlier_thresh, iters, 0, H, &inliers, arena);

    /* Warp and blend image b onto image a */
    float Hinv[3][3];
    if (!homo_invert_3x3(H, Hinv)) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_INVALID_ARG;
    }

    size_t out_h = dst->shape[0];
    size_t out_w = dst->shape[1];
    size_t out_c = (dst->ndim == 3) ? dst->shape[2] : 1;

    float* dst_data = (float*)dst->data;
    int dst_sy = (int)dst->strides[0];
    int dst_sx = (int)dst->strides[1];

    for (size_t y = 0; y < out_h; ++y) {
        for (size_t x = 0; x < out_w; ++x) {
            bool has_a = (x < img_a->shape[1] && y < img_a->shape[0]);
            RCRE_CV_Point p_dst = {(float)x, (float)y};
            RCRE_CV_Point p_b = homo_project_point_h(Hinv, p_dst);
            bool has_b = (p_b.x >= 0.0f && p_b.x < (float)img_b->shape[1] &&
                          p_b.y >= 0.0f && p_b.y < (float)img_b->shape[0]);

            for (size_t c = 0; c < out_c; ++c) {
                float val_a = has_a ? homo_sample_bilinear_src(img_a, (float)x, (float)y, c) : 0.0f;
                float val_b = has_b ? homo_sample_bilinear_src(img_b, p_b.x, p_b.y, c) : 0.0f;

                if (has_a && has_b) {
                    dst_data[y * dst_sy + x * dst_sx + c] = 0.5f * (val_a + val_b);
                } else if (has_a) {
                    dst_data[y * dst_sy + x * dst_sx + c] = val_a;
                } else if (has_b) {
                    dst_data[y * dst_sy + x * dst_sx + c] = val_b;
                } else {
                    dst_data[y * dst_sy + x * dst_sx + c] = 0.0f;
                }
            }
        }
    }

    rcre_arena_temp_end(scratch);
    return RCRE_OK;
}
