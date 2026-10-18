/**
 * @file cam_core.c
 * @brief Hardware Camera Capture Core Dispatcher
 * 
 * Satisfies FR-07.
 */

#include "cam_internal.h"

RCRE_NODISCARD RCRE_Camera* rcre_cam_open(const RCRE_CamConfig* config, RCRE_Arena* arena) {
    if (!arena) return NULL;

    RCRE_CamConfig cfg = config ? *config : rcre_cam_default_config();

    switch (cfg.backend) {
        case RCRE_CAM_BACKEND_DUMMY:
            return rcre_cam_dummy_open(&cfg, arena);

#if defined(__APPLE__)
        case RCRE_CAM_BACKEND_AVF:
            return rcre_cam_avf_open(&cfg, arena);
#endif

#if defined(__linux__)
        case RCRE_CAM_BACKEND_V4L2:
            return rcre_cam_v4l2_open(&cfg, arena);
#endif

        case RCRE_CAM_BACKEND_AUTO:
        default:
#if defined(__APPLE__)
            {
                RCRE_Camera* cam = rcre_cam_avf_open(&cfg, arena);
                if (cam) return cam;
            }
#elif defined(__linux__)
            {
                RCRE_Camera* cam = rcre_cam_v4l2_open(&cfg, arena);
                if (cam) return cam;
            }
#endif
            /* Fallback to synthetic dummy camera */
            return rcre_cam_dummy_open(&cfg, arena);
    }
}

RCRE_NODISCARD int rcre_cam_grab(RCRE_Camera* cam, RCRE_Tensor* out_frame) {
    if (!cam || !cam->grab_fn) {
        return RCRE_ERR_INVALID_ARG;
    }
    return cam->grab_fn(cam, out_frame);
}

void rcre_cam_close(RCRE_Camera* cam) {
    if (cam && cam->close_fn) {
        cam->close_fn(cam);
    }
}

const char* rcre_cam_backend_name(const RCRE_Camera* cam) {
    if (!cam) return "None";
    return cam->backend_name ? cam->backend_name : "Unknown Backend";
}
