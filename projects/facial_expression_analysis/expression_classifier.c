/**
 * @file expression_classifier.c
 * @brief Categorical Facial Expression & Emotion Classification Engine
 */

#include "expression_classifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char* EMOTION_NAMES[RCRE_NUM_EMOTIONS] = {
    "Happy",
    "Sad",
    "Anxious",
    "Panic",
    "Angry",
    "Neutral"
};

const char* rcre_emotion_to_string(RCRE_EmotionType emotion) {
    if ((int)emotion >= 0 && (int)emotion < RCRE_NUM_EMOTIONS) {
        return EMOTION_NAMES[emotion];
    }
    return "Unknown";
}

RCRE_NODISCARD int rcre_classify_expression(const RCRE_ActionUnits* au, 
                                            RCRE_ExpressionResult* out_result) {
    if (!au || !out_result) {
        return RCRE_ERR_INVALID_ARG;
    }

    float raw_scores[RCRE_NUM_EMOTIONS] = {0.0f};

    float mar = au->mouth_aspect_ratio;
    float smile = au->smile_curvature;
    float brow_elev = (au->brow_elevation_left + au->brow_elevation_right) * 0.5f;
    float brow_furrow = au->brow_furrow_distance;
    float glabella = au->glabella_wrinkle_energy;
    float ear = (au->eye_aspect_ratio_left + au->eye_aspect_ratio_right) * 0.5f;

    /* 1. Happy Logit: AU12 (Lip Corner Puller) */
    float smile_active = smile > 0.04f ? (smile - 0.04f) : 0.0f;
    raw_scores[EMOTION_HAPPY] = 25.0f * smile_active + 
                                3.0f * (au->mouth_width_ratio - 0.82f) + 
                                2.0f * au->nasolabial_wrinkle_energy;

    /* 2. Sad Logit: AU15 (Lip Corner Depressor / Frown) */
    float frown_active = smile < -0.03f ? (-0.03f - smile) : 0.0f;
    raw_scores[EMOTION_SAD] = 25.0f * frown_active + 
                              4.0f * (0.33f - brow_elev) - 
                              3.0f * (mar - 0.22f);

    /* 3. Panic / Surprise Logit: AU26/AU27 (Jaw Drop) + AU1+AU2 (High Brows) */
    raw_scores[EMOTION_PANIC] = 16.0f * (mar - 0.32f) + 
                                12.0f * (brow_elev - 0.32f) + 
                                4.0f * (ear - 0.20f);

    /* 4. Anxious / Fear Logit: AU20 (Lip Stretcher) + AU4 (Furrow) + High Brow */
    raw_scores[EMOTION_ANXIOUS] = 12.0f * (brow_elev - 0.33f) + 
                                  12.0f * glabella + 
                                  5.0f * (0.85f - brow_furrow) + 
                                  3.0f * (au->mouth_width_ratio - 0.80f);

    /* 5. Angry Logit: AU4 (Corrugator Furrow) + Lowered Brow + AU23 (Lip Compression) */
    raw_scores[EMOTION_ANGRY] = 18.0f * (0.33f - brow_elev) + 
                                16.0f * glabella + 
                                6.0f * (0.80f - brow_furrow) + 
                                4.0f * (0.24f - mar);

    /* 6. Neutral Logit: Equilibrium state */
    float deviation = fabsf(smile) * 15.0f + 
                      fabsf(mar - 0.23f) * 10.0f + 
                      fabsf(brow_elev - 0.32f) * 12.0f + 
                      glabella * 8.0f;
    raw_scores[EMOTION_NEUTRAL] = 3.5f - deviation;

    /* Softmax Normalization (with max subtraction for numerical stability) */
    float max_s = raw_scores[0];
    for (int i = 1; i < RCRE_NUM_EMOTIONS; ++i) {
        if (raw_scores[i] > max_s) max_s = raw_scores[i];
    }

    float exp_sum = 0.0f;
    float exps[RCRE_NUM_EMOTIONS];
    for (int i = 0; i < RCRE_NUM_EMOTIONS; ++i) {
        exps[i] = expf(raw_scores[i] - max_s);
        exp_sum += exps[i];
    }

    int best_idx = 0;
    float best_prob = 0.0f;
    for (int i = 0; i < RCRE_NUM_EMOTIONS; ++i) {
        out_result->probabilities[i] = exps[i] / (exp_sum > 1e-6f ? exp_sum : 1e-6f);
        if (out_result->probabilities[i] > best_prob) {
            best_prob = out_result->probabilities[i];
            best_idx = i;
        }
    }

    out_result->dominant_emotion = (RCRE_EmotionType)best_idx;
    out_result->emotion_name = EMOTION_NAMES[best_idx];
    out_result->confidence = best_prob;

    return RCRE_OK;
}
