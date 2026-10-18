/**
 * @file cam_dummy.c
 * @brief Synthetic Test Pattern Camera Backend for Headless Testing & CI
 * 
 * Generates moving color bars and gradients with zero hardware dependencies.
 */

#include "cam_internal.h"
#include <math.h>

typedef struct {
    size_t frame_index;
} DummyCamContext;

static int dummy_grab(RCRE_Camera* cam, RCRE_Tensor* out_frame) {
    if (!cam || !out_frame || !out_frame->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t H = cam->config.height;
    size_t W = cam->config.width;

    if (out_frame->shape[0] != H || out_frame->shape[1] != W || out_frame->shape[2] != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    DummyCamContext* ctx = (DummyCamContext*)cam->backend_handle;
    uint8_t* pixels = (uint8_t*)out_frame->data;
    size_t frame = ctx->frame_index++;

    /* Generate animated synthetic color bars */
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            size_t idx = (y * W + x) * 3;
            uint8_t r = (uint8_t)((x + frame * 3) % 256);
            uint8_t g = (uint8_t)((y + frame * 2) % 256);
            uint8_t b = (uint8_t)((x + y + frame) % 256);

            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    return RCRE_OK;
}

static void dummy_close(RCRE_Camera* cam) {
    (void)cam;
}

RCRE_Camera* rcre_cam_dummy_open(const RCRE_CamConfig* config, RCRE_Arena* arena) {
    RCRE_Camera* cam = (RCRE_Camera*)rcre_arena_alloc(arena, sizeof(RCRE_Camera), 64);
    DummyCamContext* ctx = (DummyCamContext*)rcre_arena_alloc(arena, sizeof(DummyCamContext), 64);

    if (!cam || !ctx) return NULL;

    cam->config = *config;
    cam->backend_name = "Synthetic Test Pattern (Dummy)";
    cam->backend_handle = ctx;
    cam->grab_fn = dummy_grab;
    cam->close_fn = dummy_close;

    ctx->frame_index = 0;

    return cam;
}
