/**
 * @file cam_v4l2.c
 * @brief Linux Video4Linux2 Camera Capture Backend
 * 
 * Satisfies FR-07 on Linux & Raspberry Pi.
 */

#include "cam_internal.h"

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>

typedef struct {
    int   fd;
    void* buffer;
    size_t buffer_len;
} V4L2CamContext;

static int v4l2_grab(RCRE_Camera* cam, RCRE_Tensor* out_frame) {
    V4L2CamContext* ctx = (V4L2CamContext*)cam->backend_handle;
    if (!ctx || ctx->fd < 0 || !out_frame || !out_frame->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0) {
        return RCRE_ERR_TIMEOUT;
    }

    size_t nbytes = cam->config.width * cam->config.height * 3;
    size_t copy_bytes = (buf.bytesused < nbytes) ? buf.bytesused : nbytes;
    memcpy(out_frame->data, ctx->buffer, copy_bytes);

    ioctl(ctx->fd, VIDIOC_QBUF, &buf);
    return RCRE_OK;
}

static void v4l2_close(RCRE_Camera* cam) {
    if (!cam) return;
    V4L2CamContext* ctx = (V4L2CamContext*)cam->backend_handle;
    if (ctx) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
        if (ctx->buffer && ctx->buffer != MAP_FAILED) {
            munmap(ctx->buffer, ctx->buffer_len);
        }
        if (ctx->fd >= 0) {
            close(ctx->fd);
        }
    }
}

RCRE_Camera* rcre_cam_v4l2_open(const RCRE_CamConfig* config, RCRE_Arena* arena) {
    char dev_name[32];
    snprintf(dev_name, sizeof(dev_name), "/dev/video%d", config->device_index);

    int fd = open(dev_name, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) return NULL;

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (uint32_t)config->width;
    fmt.fmt.pix.height = (uint32_t)config->height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        close(fd);
        return NULL;
    }

    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        close(fd);
        return NULL;
    }

    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
        close(fd);
        return NULL;
    }

    void* buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffer == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
        munmap(buffer, buf.length);
        close(fd);
        return NULL;
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        munmap(buffer, buf.length);
        close(fd);
        return NULL;
    }

    RCRE_Camera* cam = (RCRE_Camera*)rcre_arena_alloc(arena, sizeof(RCRE_Camera), 64);
    V4L2CamContext* ctx = (V4L2CamContext*)rcre_arena_alloc(arena, sizeof(V4L2CamContext), 64);
    if (!cam || !ctx) {
        munmap(buffer, buf.length);
        close(fd);
        return NULL;
    }

    ctx->fd = fd;
    ctx->buffer = buffer;
    ctx->buffer_len = buf.length;

    cam->config = *config;
    cam->backend_name = "Linux Video4Linux2 (v4l2)";
    cam->backend_handle = ctx;
    cam->grab_fn = v4l2_grab;
    cam->close_fn = v4l2_close;

    return cam;
}
#endif /* __linux__ */
