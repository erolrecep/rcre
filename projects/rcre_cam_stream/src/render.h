/**
 * @file render.h
 * @brief SDL3 + OpenGL 3.3 Core Renderer for RCRE Camera Stream
 */

#ifndef RCRE_CAM_STREAM_RENDER_H
#define RCRE_CAM_STREAM_RENDER_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    PIXFMT_RGB24 = 0,
    PIXFMT_NV12  = 1,
    PIXFMT_YUYV  = 2,
    PIXFMT_MJPEG = 3
} pixfmt_t;

typedef struct {
    const uint8_t* data;
    size_t         size;
    pixfmt_t       fmt;
    int            width;
    int            height;
    int            stride;
} frame_t;

typedef struct renderer renderer_t;

renderer_t* render_create(int width, int height, const char* title);
void        render_frame(renderer_t* r, const frame_t* f);
int         render_poll_quit(void); /* 1 when user requests quit */
void        render_destroy(renderer_t* r);

#endif /* RCRE_CAM_STREAM_RENDER_H */
