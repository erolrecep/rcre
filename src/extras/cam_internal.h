/**
 * @file cam_internal.h
 * @brief Internal declarations for RCRE Camera Companion Backends
 */

#ifndef RCRE_CAM_INTERNAL_H
#define RCRE_CAM_INTERNAL_H

#include "extras/rcre_cam.h"

struct RCRE_Camera {
    RCRE_CamConfig config;
    const char*    backend_name;
    void*          backend_handle;
    int (*grab_fn)(RCRE_Camera* cam, RCRE_Tensor* out_frame);
    void (*close_fn)(RCRE_Camera* cam);
};

/* Synthetic / Test Pattern Backend */
RCRE_Camera* rcre_cam_dummy_open(const RCRE_CamConfig* config, RCRE_Arena* arena);

/* AVFoundation Backend (macOS) */
#if defined(__APPLE__)
RCRE_Camera* rcre_cam_avf_open(const RCRE_CamConfig* config, RCRE_Arena* arena);
#endif

/* Video4Linux2 Backend (Linux) */
#if defined(__linux__)
RCRE_Camera* rcre_cam_v4l2_open(const RCRE_CamConfig* config, RCRE_Arena* arena);
#endif

#endif /* RCRE_CAM_INTERNAL_H */
