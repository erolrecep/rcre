/**
 * @file expression_classifier.h
 * @brief Categorical Facial Expression & Emotion Classification Engine
 */

#ifndef RCRE_EXPRESSION_CLASSIFIER_H
#define RCRE_EXPRESSION_CLASSIFIER_H

#include "rcre.h"
#include "action_units.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EMOTION_HAPPY    = 0,
    EMOTION_SAD      = 1,
    EMOTION_ANXIOUS  = 2,
    EMOTION_PANIC    = 3,
    EMOTION_ANGRY    = 4,
    EMOTION_NEUTRAL  = 5,
    RCRE_NUM_EMOTIONS = 6
} RCRE_EmotionType;

/**
 * @brief Final expression classification result with categorical probability distribution
 */
typedef struct {
    RCRE_EmotionType dominant_emotion;
    const char*      emotion_name;
    float            confidence;                    /* Confidence of dominant category in [0.0, 1.0] */
    float            probabilities[RCRE_NUM_EMOTIONS]; /* Calibrated probability distribution */
} RCRE_ExpressionResult;

/**
 * @brief Classify facial expression from extracted FACS Action Unit metrics.
 * 
 * @param au Action units and geometric ratios.
 * @param out_result Output classification struct with probabilities and dominant label.
 * @return RCRE_OK (0) on success, or error code.
 */
RCRE_NODISCARD int rcre_classify_expression(const RCRE_ActionUnits* au, 
                                            RCRE_ExpressionResult* out_result);

/**
 * @brief Convert emotion enum type to human-readable string.
 */
const char* rcre_emotion_to_string(RCRE_EmotionType emotion);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_EXPRESSION_CLASSIFIER_H */
