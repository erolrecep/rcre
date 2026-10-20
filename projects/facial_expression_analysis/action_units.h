/**
 * @file action_units.h
 * @brief Facial Action Coding System (FACS) Biomechanical Action Unit Extractor
 */

#ifndef RCRE_ACTION_UNITS_H
#define RCRE_ACTION_UNITS_H

#include "rcre.h"
#include "landmark_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RCRE_FACS_FEATURE_DIM 16

/**
 * @brief Extracted FACS Action Unit metrics and invariant geometric ratios
 */
typedef struct {
    float mouth_aspect_ratio;        /* MAR: vertical lip opening / mouth width */
    float mouth_width_ratio;         /* Mouth width / IOD */
    float smile_curvature;           /* AU12: elevation of mouth corners (+: smile, -: frown) */
    float eye_aspect_ratio_left;     /* EAR Left: eye aperture ratio */
    float eye_aspect_ratio_right;    /* EAR Right: eye aperture ratio */
    float brow_elevation_left;       /* AU1+AU2: left brow elevation above eye / IOD */
    float brow_elevation_right;      /* AU1+AU2: right brow elevation above eye / IOD */
    float brow_furrow_distance;      /* AU4: inter-eyebrow span / IOD */
    float glabella_wrinkle_energy;   /* Sobel texture energy between eyebrows */
    float nasolabial_wrinkle_energy; /* Sobel texture energy along cheek smile creases */
    float facial_symmetry_index;     /* Bilateral facial activation symmetry */
    float features[RCRE_FACS_FEATURE_DIM]; /* Normalized 16-D vector for ML classification */
} RCRE_ActionUnits;

/**
 * @brief Extract FACS Action Unit metrics from canonical face image and localized landmarks.
 * 
 * @param canonical_rgb 128x128x3 RGB face crop.
 * @param landmarks Detected 13-point facial landmarks constellation.
 * @param out_au Destination Action Units structure.
 * @param scratch_arena Scratchpad memory arena for Sobel filtering.
 * @return RCRE_OK (0) on success, or error code.
 */
RCRE_NODISCARD int rcre_extract_action_units(const uint8_t* canonical_rgb, 
                                            const RCRE_FacialLandmarks* landmarks, 
                                            RCRE_ActionUnits* out_au, 
                                            RCRE_Arena* scratch_arena);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_ACTION_UNITS_H */
