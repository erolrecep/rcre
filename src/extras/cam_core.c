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
    RCRE_Camera* cam = NULL;

    switch (cfg.backend) {
        case RCRE_CAM_BACKEND_DUMMY:
            cam = rcre_cam_dummy_open(&cfg, arena);
            break;

#if defined(__APPLE__)
        case RCRE_CAM_BACKEND_AVF:
            cam = rcre_cam_avf_open(&cfg, arena);
            break;
#endif

#if defined(__linux__)
        case RCRE_CAM_BACKEND_V4L2:
            cam = rcre_cam_v4l2_open(&cfg, arena);
            break;
#endif

        case RCRE_CAM_BACKEND_AUTO:
        default:
#if defined(__APPLE__)
            cam = rcre_cam_avf_open(&cfg, arena);
#elif defined(__linux__)
            cam = rcre_cam_v4l2_open(&cfg, arena);
#endif
            if (!cam) {
                cam = rcre_cam_dummy_open(&cfg, arena);
            }
            break;
    }

    if (cam) {
        cam->stream_active = 0;
        cam->stream_ring = NULL;
        cam->stream_temp_data = NULL;
    }
    return cam;
}

RCRE_NODISCARD int rcre_cam_grab(RCRE_Camera* cam, RCRE_Tensor* out_frame) {
    if (!cam || !cam->grab_fn) {
        return RCRE_ERR_INVALID_ARG;
    }
    return cam->grab_fn(cam, out_frame);
}

#include <unistd.h>
#include <stdlib.h>

static void* async_capture_worker(void* arg) {
    RCRE_Camera* cam = (RCRE_Camera*)arg;
    size_t frame_bytes = cam->config.width * cam->config.height * 3;
    while (cam->stream_active) {
        int status = cam->grab_fn(cam, &cam->stream_temp_frame);
        if (status == RCRE_OK) {
            rcre_ringbuf_push(cam->stream_ring,
                              cam->stream_temp_frame.data,
                              frame_bytes,
                              cam->config.width,
                              cam->config.height,
                              0);
        } else {
            usleep(5000); /* 5ms backoff on frame wait/timeout */
        }
    }
    return NULL;
}

int rcre_cam_start_async(RCRE_Camera* cam, RCRE_RingBuf* ring) {
    if (!cam || !cam->grab_fn || !ring) return RCRE_ERR_INVALID_ARG;
    if (cam->stream_active) return RCRE_OK; /* Already streaming */

    size_t frame_bytes = cam->config.width * cam->config.height * 3;
    cam->stream_temp_data = (uint8_t*)malloc(frame_bytes);
    if (!cam->stream_temp_data) return RCRE_ERR_OUT_OF_MEMORY;

    size_t shape[3] = {cam->config.height, cam->config.width, 3};
    cam->stream_temp_frame = (RCRE_Tensor){
        .data = cam->stream_temp_data,
        .shape = {shape[0], shape[1], shape[2]},
        .ndim = 3,
        .dtype = RCRE_DTYPE_U8,
        .layout = RCRE_LAYOUT_HWC
    };

    cam->stream_ring = ring;
    cam->stream_active = 1;

    if (pthread_create(&cam->stream_thread, NULL, async_capture_worker, cam) != 0) {
        cam->stream_active = 0;
        free(cam->stream_temp_data);
        cam->stream_temp_data = NULL;
        return RCRE_ERR_IO;
    }
    return RCRE_OK;
}

void rcre_cam_stop_async(RCRE_Camera* cam) {
    if (!cam || !cam->stream_active) return;
    cam->stream_active = 0;
    pthread_join(cam->stream_thread, NULL);
    if (cam->stream_temp_data) {
        free(cam->stream_temp_data);
        cam->stream_temp_data = NULL;
    }
    cam->stream_ring = NULL;
}

int rcre_cam_is_streaming(const RCRE_Camera* cam) {
    return cam ? cam->stream_active : 0;
}

void rcre_cam_close(RCRE_Camera* cam) {
    if (cam) {
        if (cam->stream_active) {
            rcre_cam_stop_async(cam);
        }
        if (cam->close_fn) {
            cam->close_fn(cam);
        }
    }
}

const char* rcre_cam_backend_name(const RCRE_Camera* cam) {
    if (!cam) return "None";
    return cam->backend_name ? cam->backend_name : "Unknown Backend";
}
