/**
 * @file test_classifier.c
 * @brief Comprehensive Verification of Landmarks, FACS Action Units & Emotion Classifier
 */

#include "landmark_detector.h"
#include "action_units.h"
#include "expression_classifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define TEST_ARENA_SIZE (16 * 1024 * 1024)
#define DIM 128

/* Draw a synthetic canonical face with parametric mouth curvature and brow elevation */
static void draw_parametric_face(uint8_t* rgb, 
                                 float mouth_curve,     /* +: smile, -: frown */
                                 float mouth_open,      /* 0.0: closed, 1.0: wide open */
                                 float brow_elevation,  /* 0.0: normal, +: high, -: low */
                                 float furrow)          /* 0.0: normal, 1.0: angry knit */
{
    for (int y = 0; y < DIM; ++y) {
        for (int x = 0; x < DIM; ++x) {
            size_t idx = ((size_t)y * (size_t)DIM + (size_t)x) * 3;

            /* Base Skin Tone */
            float dx = (float)(x - 64);
            float dy = (float)(y - 64);
            if (dx*dx + dy*dy <= 58.0f * 58.0f) {
                rgb[idx + 0] = 220;
                rgb[idx + 1] = 155;
                rgb[idx + 2] = 120;
            } else {
                rgb[idx + 0] = 30;
                rgb[idx + 1] = 40;
                rgb[idx + 2] = 50;
                continue;
            }

            /* Eyes (Left center at 38, 48; Right center at 90, 48) */
            float l_eye_dist = sqrtf((x - 38)*(x - 38) + (y - 48)*(y - 48));
            float r_eye_dist = sqrtf((x - 90)*(x - 90) + (y - 48)*(y - 48));
            if (l_eye_dist < 6.0f || r_eye_dist < 6.0f) {
                rgb[idx + 0] = 20; rgb[idx + 1] = 15; rgb[idx + 2] = 15;
            }

            /* Eyebrows: y baseline is 32 - brow_elevation * 10.0 */
            float brow_y_base = 32.0f - brow_elevation * 10.0f;
            float l_brow_dist = fabsf((float)y - (brow_y_base + (furrow > 0 ? (x - 38)*0.2f : 0.0f)));
            float r_brow_dist = fabsf((float)y - (brow_y_base - (furrow > 0 ? (x - 90)*0.2f : 0.0f)));

            if (x >= 24 && x <= 52 && l_brow_dist < 3.0f) {
                rgb[idx + 0] = 40; rgb[idx + 1] = 25; rgb[idx + 2] = 20;
            }
            if (x >= 76 && x <= 104 && r_brow_dist < 3.0f) {
                rgb[idx + 0] = 40; rgb[idx + 1] = 25; rgb[idx + 2] = 20;
            }

            /* Glabella furrow lines (between eyes) if furrow > 0 */
            if (furrow > 0.5f && x >= 60 && x <= 68 && y >= 30 && y <= 42 && (x % 3 == 0)) {
                rgb[idx + 0] = 120; rgb[idx + 1] = 70; rgb[idx + 2] = 50;
            }

            /* Nose Tip */
            if (x >= 61 && x <= 67 && y >= 70 && y <= 74) {
                rgb[idx + 0] = 180; rgb[idx + 1] = 110; rgb[idx + 2] = 80;
            }

            /* Mouth Complex: baseline y = 96 */
            float m_dx = (float)(x - 64);
            if (fabsf(m_dx) <= 24.0f) {
                /* Quadratic mouth curve */
                float curve_offset = (m_dx * m_dx / 24.0f) * (-mouth_curve * 0.45f);
                float mouth_center_y = 96.0f + curve_offset;
                float lip_thickness = 4.0f + mouth_open * 12.0f;

                if (fabsf((float)y - mouth_center_y) <= lip_thickness) {
                    if (mouth_open > 0.4f && fabsf((float)y - mouth_center_y) < lip_thickness * 0.6f) {
                        /* Dark open oral cavity */
                        rgb[idx + 0] = 30; rgb[idx + 1] = 10; rgb[idx + 2] = 10;
                    } else {
                        /* Pink-red lips */
                        rgb[idx + 0] = 200; rgb[idx + 1] = 50; rgb[idx + 2] = 60;
                    }
                }
            }
        }
    }
}

int main(void) {
    printf("===================================================================\n");
    printf("🧪 RCRE FaceX: Phase 2 & 3 FACS & Emotion Classification Suite     \n");
    printf("===================================================================\n\n");

    rcre_hal_init();

    RCRE_Arena* arena = rcre_arena_create(TEST_ARENA_SIZE);
    assert(arena != NULL);

    uint8_t face[DIM * DIM * 3];
    RCRE_FacialLandmarks lm;
    RCRE_ActionUnits au;
    RCRE_ExpressionResult res;

    /* ------------------------------------------------------------- */
    /* Test 1: Happy Expression (Smile Curvature > 0)                */
    /* ------------------------------------------------------------- */
    printf("[Test 1/5] Evaluating 'Happy' Expression (Smile Curve = +1.0)...\n");
    draw_parametric_face(face, +1.0f, 0.1f, 0.0f, 0.0f);
    assert(rcre_detect_landmarks(face, &lm, arena) == RCRE_OK);
    assert(rcre_extract_action_units(face, &lm, &au, arena) == RCRE_OK);
    assert(rcre_classify_expression(&au, &res) == RCRE_OK);

    printf("  ✓ Landmark IOD: %.1f px | Smile Curvature: %+.4f\n", lm.inter_ocular_dist, au.smile_curvature);
    printf("  ✓ Classified Emotion: %s (Confidence: %.1f%%)\n", res.emotion_name, res.confidence * 100.0f);
    assert(res.dominant_emotion == EMOTION_HAPPY);
    assert(res.confidence > 0.65f);

    /* ------------------------------------------------------------- */
    /* Test 2: Sad Expression (Frown Curvature < 0)                  */
    /* ------------------------------------------------------------- */
    printf("[Test 2/5] Evaluating 'Sad' Expression (Frown Curve = -1.0)...\n");
    draw_parametric_face(face, -1.0f, 0.0f, -0.1f, 0.0f);
    assert(rcre_detect_landmarks(face, &lm, arena) == RCRE_OK);
    assert(rcre_extract_action_units(face, &lm, &au, arena) == RCRE_OK);
    assert(rcre_classify_expression(&au, &res) == RCRE_OK);

    printf("  ✓ Smile Curvature: %+.4f | MAR: %.4f\n", au.smile_curvature, au.mouth_aspect_ratio);
    printf("  ✓ Classified Emotion: %s (Confidence: %.1f%%)\n", res.emotion_name, res.confidence * 100.0f);
    assert(res.dominant_emotion == EMOTION_SAD);
    assert(res.confidence > 0.60f);

    /* ------------------------------------------------------------- */
    /* Test 3: Panic / Surprise (Jaw Drop MAR > 0.45, High Brows)    */
    /* ------------------------------------------------------------- */
    printf("[Test 3/5] Evaluating 'Panic' Expression (Wide Open Mouth + High Brows)...\n");
    draw_parametric_face(face, 0.0f, 1.0f, +1.0f, 0.0f);
    assert(rcre_detect_landmarks(face, &lm, arena) == RCRE_OK);
    assert(rcre_extract_action_units(face, &lm, &au, arena) == RCRE_OK);
    assert(rcre_classify_expression(&au, &res) == RCRE_OK);

    printf("  ✓ MAR: %.4f | Brow Elevation: %.4f\n", au.mouth_aspect_ratio, (au.brow_elevation_left + au.brow_elevation_right)*0.5f);
    printf("  ✓ Classified Emotion: %s (Confidence: %.1f%%)\n", res.emotion_name, res.confidence * 100.0f);
    assert(res.dominant_emotion == EMOTION_PANIC);
    assert(res.confidence > 0.60f);

    /* ------------------------------------------------------------- */
    /* Test 4: Angry (Lowered/Furrowed Brows + Tight Mouth)          */
    /* ------------------------------------------------------------- */
    printf("[Test 4/5] Evaluating 'Angry' Expression (Furrowed Brows + Glabella Energy)...\n");
    draw_parametric_face(face, 0.0f, 0.0f, -1.0f, 1.0f);
    assert(rcre_detect_landmarks(face, &lm, arena) == RCRE_OK);
    assert(rcre_extract_action_units(face, &lm, &au, arena) == RCRE_OK);
    assert(rcre_classify_expression(&au, &res) == RCRE_OK);

    printf("  ✓ Glabella Energy: %.4f | Brow Elevation: %.4f\n", au.glabella_wrinkle_energy, (au.brow_elevation_left + au.brow_elevation_right)*0.5f);
    printf("  ✓ Classified Emotion: %s (Confidence: %.1f%%)\n", res.emotion_name, res.confidence * 100.0f);
    assert(res.dominant_emotion == EMOTION_ANGRY);
    assert(res.confidence > 0.50f);

    /* ------------------------------------------------------------- */
    /* Test 5: Neutral Expression (Equilibrium Baseline)             */
    /* ------------------------------------------------------------- */
    printf("[Test 5/5] Evaluating 'Neutral' Expression (Equilibrium State)...\n");
    draw_parametric_face(face, 0.0f, 0.0f, 0.0f, 0.0f);
    assert(rcre_detect_landmarks(face, &lm, arena) == RCRE_OK);
    assert(rcre_extract_action_units(face, &lm, &au, arena) == RCRE_OK);
    assert(rcre_classify_expression(&au, &res) == RCRE_OK);

    printf("  ✓ Smile Curvature: %+.4f | MAR: %.4f | Brow Elevation: %.4f\n", 
           au.smile_curvature, au.mouth_aspect_ratio, (au.brow_elevation_left + au.brow_elevation_right)*0.5f);
    printf("  ✓ Classified Emotion: %s (Confidence: %.1f%%)\n", res.emotion_name, res.confidence * 100.0f);
    assert(res.dominant_emotion == EMOTION_NEUTRAL);

    rcre_arena_destroy(arena);

    printf("\n🎉 ALL 5 FACS & EMOTION CLASSIFIER INVARIANTS PASSED!\n");
    printf("===================================================================\n");
    return 0;
}
