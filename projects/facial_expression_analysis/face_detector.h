/**
 * @file face_detector.h
 * @brief High-Speed, Pure C23 Face Detection & Canonical Normalization Engine
 */

#ifndef RCRE_FACE_DETECTOR_H
#define RCRE_FACE_DETECTOR_H

#include "rcre.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RCRE_CANONICAL_FACE_SIZE 128

/**
 * @brief Bounding box coordinates and confidence score for a detected face
 */
typedef struct {
    int   x;
    int   y;
    int   width;
    int   height;
    float confidence;
} RCRE_FaceBox;

/**
 * @brief Detect human faces in an RGB image using skin-locus saliency and integral cascades.
 * 
 * @param rgb Interleaved RGB image buffer (height x width x 3).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param out_faces Array to populate with detected face boxes.
 * @param max_faces Maximum number of faces to return.
 * @param scratch_arena Scratchpad arena for intermediate tensors.
 * @return Number of detected faces (>= 0), or negative error code.
 */
RCRE_NODISCARD int rcre_detect_faces(const uint8_t* rgb, 
                                     int width, int height, 
                                     RCRE_FaceBox* out_faces, 
                                     size_t max_faces, 
                                     RCRE_Arena* scratch_arena);

/**
 * @brief Load an image from disk and detect faces.
 * 
 * @param filepath Path to .jpg, .png, or .bmp image.
 * @param out_faces Array to populate with detected face boxes.
 * @param max_faces Maximum number of faces to return.
 * @param out_width Pointer to store loaded image width (optional).
 * @param out_height Pointer to store loaded image height (optional).
 * @param scratch_arena Scratchpad arena for memory management.
 * @return Number of detected faces (>= 0), or negative error code.
 */
RCRE_NODISCARD int rcre_detect_faces_file(const char* filepath, 
                                          RCRE_FaceBox* out_faces, 
                                          size_t max_faces, 
                                          int* out_width, 
                                          int* out_height, 
                                          RCRE_Arena* scratch_arena);

/**
 * @brief Crop a detected face bounding box and canonically resample to 128x128 RGB.
 * 
 * @param rgb Source RGB image buffer.
 * @param width Source image width.
 * @param height Source image height.
 * @param face Target face bounding box.
 * @param out_canonical_rgb Output buffer of size (128 x 128 x 3) bytes.
 * @param scratch_arena Scratchpad arena for bilinear interpolation.
 * @return RCRE_OK (0) on success, or error code.
 */
RCRE_NODISCARD int rcre_crop_canonical_face(const uint8_t* rgb, 
                                            int width, int height, 
                                            const RCRE_FaceBox* face, 
                                            uint8_t* out_canonical_rgb, 
                                            RCRE_Arena* scratch_arena);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_FACE_DETECTOR_H */
