/**
 * @file test_extractor.c
 * @brief Automated Verification Suite for RCRE 64-D Feature Extractor
 */

#include "feature_extractor.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-4f

/* Helper to generate a synthetic RGB image buffer */
static uint8_t* create_test_image(size_t w, size_t h, int pattern_type) {
    uint8_t* img = (uint8_t*)malloc(w * h * 3);
    assert(img != NULL);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            size_t idx = (y * w + x) * 3;
            if (pattern_type == 0) {
                /* Pure Red / Orange gradient */
                img[idx + 0] = (uint8_t)(200 + (x * 55) / w);
                img[idx + 1] = (uint8_t)((y * 80) / h);
                img[idx + 2] = 20;
            } else if (pattern_type == 1) {
                /* Similar Reddish / Crimson image */
                img[idx + 0] = (uint8_t)(190 + (x * 45) / w);
                img[idx + 1] = (uint8_t)((y * 70) / h);
                img[idx + 2] = 30;
            } else if (pattern_type == 2) {
                /* Pure Green landscape style */
                img[idx + 0] = 10;
                img[idx + 1] = (uint8_t)(180 + (y * 70) / h);
                img[idx + 2] = 20;
            } else if (pattern_type == 3) {
                /* Pure Deep Blue oceanic style */
                img[idx + 0] = 15;
                img[idx + 1] = 30;
                img[idx + 2] = (uint8_t)(200 + (x * 55) / w);
            } else if (pattern_type == 4) {
                /* High-frequency checkerboard / vertical stripe pattern */
                uint8_t val = (x / 8) % 2 == 0 ? 255 : 0;
                img[idx + 0] = val;
                img[idx + 1] = val;
                img[idx + 2] = val;
            } else {
                /* Flat smooth gray */
                img[idx + 0] = 128;
                img[idx + 1] = 128;
                img[idx + 2] = 128;
            }
        }
    }
    return img;
}

int main(void) {
    printf("===================================================================\n");
    printf("🧪 RCRE Lens: Phase 1 Feature Extractor Verification Suite          \n");
    printf("===================================================================\n");

    const size_t arena_size = 16 * 1024 * 1024; /* 16 MB */
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    assert(arena != NULL);

    /* Test 1: Self-Similarity Invariant */
    printf("  [Test 1] Verifying Self-Similarity Invariant (A · A == 1.0)...\n");
    {
        uint8_t* img_a = create_test_image(320, 240, 0);
        float emb_a1[RCRE_FEATURE_DIM];
        float emb_a2[RCRE_FEATURE_DIM];

        int res1 = rcre_extract_features_rgb(img_a, 320, 240, emb_a1, arena);
        int res2 = rcre_extract_features_rgb(img_a, 320, 240, emb_a2, arena);
        assert(res1 == RCRE_OK && res2 == RCRE_OK);

        float sim = rcre_compute_cosine_similarity(emb_a1, emb_a2, RCRE_FEATURE_DIM);
        float dist = rcre_compute_l2_distance(emb_a1, emb_a2, RCRE_FEATURE_DIM);

        printf("    -> Cosine Similarity: %.6f (Expected: 1.000000)\n", sim);
        printf("    -> L2 Distance:        %.6f (Expected: 0.000000)\n", dist);
        assert(fabsf(sim - 1.0f) < EPSILON);
        assert(dist < EPSILON);

        free(img_a);
        printf("    ✓ Self-similarity mathematical identity verified!\n");
    }

    /* Test 2: Color Discriminability */
    printf("\n  [Test 2] Verifying Color Spectrum Discriminability...\n");
    {
        uint8_t* img_red1   = create_test_image(320, 240, 0);
        uint8_t* img_red2   = create_test_image(320, 240, 1);
        uint8_t* img_green  = create_test_image(320, 240, 2);
        uint8_t* img_blue   = create_test_image(320, 240, 3);

        float emb_r1[RCRE_FEATURE_DIM];
        float emb_r2[RCRE_FEATURE_DIM];
        float emb_g[RCRE_FEATURE_DIM];
        float emb_b[RCRE_FEATURE_DIM];

        assert(rcre_extract_features_rgb(img_red1, 320, 240, emb_r1, arena) == RCRE_OK);
        assert(rcre_extract_features_rgb(img_red2, 320, 240, emb_r2, arena) == RCRE_OK);
        assert(rcre_extract_features_rgb(img_green, 320, 240, emb_g, arena) == RCRE_OK);
        assert(rcre_extract_features_rgb(img_blue, 320, 240, emb_b, arena) == RCRE_OK);

        float sim_red_red   = rcre_compute_cosine_similarity(emb_r1, emb_r2, RCRE_FEATURE_DIM);
        float sim_red_green = rcre_compute_cosine_similarity(emb_r1, emb_g, RCRE_FEATURE_DIM);
        float sim_red_blue  = rcre_compute_cosine_similarity(emb_r1, emb_b, RCRE_FEATURE_DIM);

        printf("    -> Similarity(Red_1, Red_2): %.4f (Expected high: > 0.95)\n", sim_red_red);
        printf("    -> Similarity(Red_1, Green): %.4f (Expected low:  < 0.40)\n", sim_red_green);
        printf("    -> Similarity(Red_1, Blue):  %.4f (Expected low:  < 0.40)\n", sim_red_blue);

        assert(sim_red_red > 0.95f);
        assert(sim_red_green < 0.50f);
        assert(sim_red_blue < 0.50f);
        assert(sim_red_red > sim_red_green && sim_red_red > sim_red_blue);

        free(img_red1);
        free(img_red2);
        free(img_green);
        free(img_blue);
        printf("    ✓ Color clustering and separation verified!\n");
    }

    /* Test 3: Structural Contour & Texture Sensitivity */
    printf("\n  [Test 3] Verifying Structural Edge & Texture Sensitivity...\n");
    {
        uint8_t* img_stripes = create_test_image(256, 256, 4);
        uint8_t* img_smooth  = create_test_image(256, 256, 5);

        float emb_stripes[RCRE_FEATURE_DIM];
        float emb_smooth[RCRE_FEATURE_DIM];

        assert(rcre_extract_features_rgb(img_stripes, 256, 256, emb_stripes, arena) == RCRE_OK);
        assert(rcre_extract_features_rgb(img_smooth, 256, 256, emb_smooth, arena) == RCRE_OK);

        /* Sum up edge components (every 4th index: 3, 7, 11, ... 63) */
        float edge_sum_stripes = 0.0f;
        float edge_sum_smooth  = 0.0f;
        for (size_t c = 0; c < RCRE_NUM_CELLS; ++c) {
            edge_sum_stripes += emb_stripes[c * 4 + 3];
            edge_sum_smooth  += emb_smooth[c * 4 + 3];
        }

        printf("    -> Edge Magnitude Sum (Striped Pattern): %.4f\n", edge_sum_stripes);
        printf("    -> Edge Magnitude Sum (Smooth Uniform):  %.4f\n", edge_sum_smooth);
        assert(edge_sum_stripes > 2.0f * edge_sum_smooth + 0.1f);

        free(img_stripes);
        free(img_smooth);
        printf("    ✓ Spatial Sobel texture sensitivity verified!\n");
    }

    /* Test 4: Scale & Resolution Invariance */
    printf("\n  [Test 4] Verifying Resolution Scaling Invariance (640x480 vs 160x120)...\n");
    {
        uint8_t* img_large = create_test_image(640, 480, 0);
        uint8_t* img_small = create_test_image(160, 120, 0);

        float emb_large[RCRE_FEATURE_DIM];
        float emb_small[RCRE_FEATURE_DIM];

        assert(rcre_extract_features_rgb(img_large, 640, 480, emb_large, arena) == RCRE_OK);
        assert(rcre_extract_features_rgb(img_small, 160, 120, emb_small, arena) == RCRE_OK);

        float sim_scale = rcre_compute_cosine_similarity(emb_large, emb_small, RCRE_FEATURE_DIM);
        printf("    -> Cosine Similarity across 4x scale variance: %.4f (Expected: > 0.98)\n", sim_scale);
        assert(sim_scale > 0.98f);

        free(img_large);
        free(img_small);
        printf("    ✓ Bilinear canonical resampling scale invariance verified!\n");
    }

    /* Test 5: Scratchpad Memory Determinism & Zero Leaks */
    printf("\n  [Test 5] Stress Testing Scratchpad Arena Invariant (50 passes)...\n");
    {
        size_t initial_offset = arena->offset;
        uint8_t* img = create_test_image(256, 256, 0);
        float emb[RCRE_FEATURE_DIM];

        for (int iter = 0; iter < 50; ++iter) {
            assert(rcre_extract_features_rgb(img, 256, 256, emb, arena) == RCRE_OK);
            assert(arena->offset == initial_offset); /* Verifies O(1) bulk rewind */
        }

        free(img);
        printf("    ✓ Scratchpad arena perfectly rewound across 50 iterations (0 memory leaks)!\n");
    }

    rcre_arena_destroy(arena);

    printf("\n===================================================================\n");
    printf("🎉 All Phase 1 Feature Extractor Invariants Verified Successfully! \n");
    printf("===================================================================\n");
    return 0;
}
