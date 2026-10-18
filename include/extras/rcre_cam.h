/**
 * @file rcre_cam.h
 * @brief RCRE Hardware Camera Capture Companion Module
 * 
 * Satisfies FR-07.
 * Segregated from core rcre.h to maintain zero-dependency pure C23 compliance.
 * Supports macOS AVFoundation, Linux Video4Linux2, WASM, and Synthetic Test Pattern fallback.
 */

#ifndef RCRE_CAM_H
#define RCRE_CAM_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RCRE_Camera RCRE_Camera;

typedef enum {
    RCRE_CAM_BACKEND_AUTO    = 0,
    RCRE_CAM_BACKEND_AVF     = 1, /* macOS AVFoundation */
    RCRE_CAM_BACKEND_V4L2    = 2, /* Linux Video4Linux2 */
    RCRE_CAM_BACKEND_WASM    = 3, /* WebRTC / HTML5 Canvas */
    RCRE_CAM_BACKEND_DUMMY   = 4  /* Synthetic / Test Pattern generator */
} RCRE_CamBackend;

typedef struct RCRE_CamConfig {
    size_t          width;           /* Desired frame width (e.g. 640 or 1920) */
    size_t          height;          /* Desired frame height (e.g. 480 or 1080) */
    size_t          fps;             /* Desired framerate (e.g. 30 or 60) */
    int             device_index;    /* Device ID (0 = default webcam) */
    RCRE_CamBackend backend;         /* Selected backend (AUTO defaults to platform-native) */
} RCRE_CamConfig;

/**
 * @brief Default camera configuration helper.
 */
static inline RCRE_CamConfig rcre_cam_default_config(void) {
    RCRE_CamConfig cfg = {
        .width = 640,
        .height = 480,
        .fps = 30,
        .device_index = 0,
        .backend = RCRE_CAM_BACKEND_AUTO
    };
    return cfg;
}

/**
 * @brief Opens and initializes camera device.
 * @param config Camera settings.
 * @param arena Arena for internal camera buffers.
 * @return Opaque pointer to RCRE_Camera instance, or NULL on failure.
 */
RCRE_NODISCARD RCRE_Camera* rcre_cam_open(const RCRE_CamConfig* config, RCRE_Arena* arena);

/**
 * @brief Grabs the next video frame into caller-supplied destination tensor.
 * @param cam Active camera handle.
 * @param out_frame Destination tensor [Height, Width, 3] U8 in HWC layout.
 * @return RCRE_OK on success, or error code if no frame ready.
 */
RCRE_NODISCARD int rcre_cam_grab(RCRE_Camera* cam, RCRE_Tensor* out_frame);

/**
 * @brief Stops capture and releases camera device resources.
 */
void rcre_cam_close(RCRE_Camera* cam);

/**
 * @brief Returns the name of the active camera backend.
 */
const char* rcre_cam_backend_name(const RCRE_Camera* cam);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_CAM_H */
