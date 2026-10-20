/**
 * @file landmark_detector.h
 * @brief Anthropometric Facial Landmark Localization & Geometry Engine
 */

#ifndef RCRE_LANDMARK_DETECTOR_H
#define RCRE_LANDMARK_DETECTOR_H

#include "rcre.h"
#include "face_detector.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LANDMARK_LEFT_BROW_ARCH     = 0,
    LANDMARK_RIGHT_BROW_ARCH    = 1,
    LANDMARK_LEFT_EYE_OUTER     = 2,
    LANDMARK_LEFT_EYE_CENTER    = 3,
    LANDMARK_LEFT_EYE_INNER     = 4,
    LANDMARK_RIGHT_EYE_INNER    = 5,
    LANDMARK_RIGHT_EYE_CENTER   = 6,
    LANDMARK_RIGHT_EYE_OUTER    = 7,
    LANDMARK_NOSE_APEX          = 8,
    LANDMARK_MOUTH_LEFT_CORNER  = 9,
    LANDMARK_MOUTH_RIGHT_CORNER = 10,
    LANDMARK_UPPER_LIP_PEAK     = 11,
    LANDMARK_LOWER_LIP_CENTER   = 12,
    RCRE_NUM_LANDMARKS          = 13
} RCRE_LandmarkIndex;

/**
 * @brief 2D subpixel landmark coordinate and confidence
 */
typedef struct {
    float x;
    float y;
    float confidence;
} RCRE_LandmarkPoint;

/**
 * @brief 13-point anthropometric constellation of facial landmarks
 */
typedef struct {
    RCRE_LandmarkPoint points[RCRE_NUM_LANDMARKS];
    float              inter_ocular_dist; /* IOD: Euclidean distance between eye centers */
} RCRE_FacialLandmarks;

/**
 * @brief Localize facial landmarks on a canonically cropped (128x128) face image.
 * 
 * @param canonical_rgb 128x128x3 RGB face crop.
 * @param out_landmarks Output constellation structure.
 * @param scratch_arena Scratchpad arena for intermediate tensors.
 * @return RCRE_OK (0) on success, or error code.
 */
RCRE_NODISCARD int rcre_detect_landmarks(const uint8_t* canonical_rgb, 
                                         RCRE_FacialLandmarks* out_landmarks, 
                                         RCRE_Arena* scratch_arena);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_LANDMARK_DETECTOR_H */
