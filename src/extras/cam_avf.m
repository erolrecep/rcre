/**
 * @file cam_avf.m
 * @brief macOS AVFoundation Camera Capture Backend
 * 
 * Satisfies FR-07 on macOS.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include "cam_internal.h"

@interface RCRE_AVFReceiver : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property (nonatomic, assign) CVImageBufferRef latestPixelBuffer;
@property (nonatomic, strong) dispatch_semaphore_t frameReady;
@end

@implementation RCRE_AVFReceiver
- (instancetype)init {
    self = [super init];
    if (self) {
        _latestPixelBuffer = NULL;
        _frameReady = dispatch_semaphore_create(0);
    }
    return self;
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    (void)output; (void)connection;
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (imageBuffer) {
        @synchronized(self) {
            if (_latestPixelBuffer) {
                CVBufferRelease(_latestPixelBuffer);
            }
            _latestPixelBuffer = (CVImageBufferRef)CVBufferRetain(imageBuffer);
        }
        dispatch_semaphore_signal(_frameReady);
    }
}

- (void)dealloc {
    if (_latestPixelBuffer) {
        CVBufferRelease(_latestPixelBuffer);
    }
}
@end

typedef struct {
    AVCaptureSession*  session;
    RCRE_AVFReceiver*  receiver;
    dispatch_queue_t   queue;
} AVFCamContext;

static int avf_grab(RCRE_Camera* cam, RCRE_Tensor* out_frame) {
    if (!cam || !out_frame || !out_frame->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    AVFCamContext* ctx = (AVFCamContext*)cam->backend_handle;
    if (!ctx) return RCRE_ERR_INVALID_ARG;

    /* Wait for frame with 100ms timeout */
    intptr_t signaled = dispatch_semaphore_wait(ctx->receiver.frameReady, 
                                                dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC));
    if (signaled != 0) {
        return RCRE_ERR_TIMEOUT;
    }

    CVImageBufferRef buffer = NULL;
    @synchronized(ctx->receiver) {
        if (ctx->receiver.latestPixelBuffer) {
            buffer = (CVImageBufferRef)CVBufferRetain(ctx->receiver.latestPixelBuffer);
        }
    }

    if (!buffer) return RCRE_ERR_INVALID_ARG;

    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    uint8_t* base_addr = (uint8_t*)CVPixelBufferGetBaseAddress(buffer);
    size_t in_w = CVPixelBufferGetWidth(buffer);
    size_t in_h = CVPixelBufferGetHeight(buffer);
    size_t bytes_per_row = CVPixelBufferGetBytesPerRow(buffer);
    size_t target_w = cam->config.width;
    size_t target_h = cam->config.height;

    uint8_t* dst = (uint8_t*)out_frame->data;

    /* AVF 32BGRA -> RGB 24-bit with full sensor field-of-view (never crop top-left) */
    if (in_w == target_w && in_h == target_h) {
        for (size_t y = 0; y < target_h; ++y) {
            const uint8_t* src_row = base_addr + y * bytes_per_row;
            uint8_t* dst_row = dst + y * target_w * 3;
            for (size_t x = 0; x < target_w; ++x) {
                dst_row[x * 3 + 0] = src_row[x * 4 + 2]; /* R */
                dst_row[x * 3 + 1] = src_row[x * 4 + 1]; /* G */
                dst_row[x * 3 + 2] = src_row[x * 4 + 0]; /* B */
            }
        }
    } else {
        /* Proportional resample across 100% of the camera sensor view */
        for (size_t y = 0; y < target_h; ++y) {
            size_t src_y = (y * in_h) / target_h;
            if (src_y >= in_h) src_y = in_h - 1;
            const uint8_t* src_row = base_addr + src_y * bytes_per_row;
            uint8_t* dst_row = dst + y * target_w * 3;
            for (size_t x = 0; x < target_w; ++x) {
                size_t src_x = (x * in_w) / target_w;
                if (src_x >= in_w) src_x = in_w - 1;
                dst_row[x * 3 + 0] = src_row[src_x * 4 + 2]; /* R */
                dst_row[x * 3 + 1] = src_row[src_x * 4 + 1]; /* G */
                dst_row[x * 3 + 2] = src_row[src_x * 4 + 0]; /* B */
            }
        }
    }

    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    CVBufferRelease(buffer);

    return RCRE_OK;
}

static void avf_close(RCRE_Camera* cam) {
    if (!cam) return;
    AVFCamContext* ctx = (AVFCamContext*)cam->backend_handle;
    if (ctx && ctx->session) {
        [ctx->session stopRunning];
    }
}

RCRE_Camera* rcre_cam_avf_open(const RCRE_CamConfig* config, RCRE_Arena* arena) {
    @autoreleasepool {
        AVCaptureDevice* device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        if (!device) return NULL;

        NSError* error = nil;
        AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
        if (!input || error) return NULL;

        AVCaptureSession* session = [[AVCaptureSession alloc] init];
        if (![session canAddInput:input]) return NULL;
        [session addInput:input];

        /* Match optimal session preset to target resolution */
        if (config->width >= 1920 && config->height >= 1080 && [session canSetSessionPreset:AVCaptureSessionPreset1920x1080]) {
            session.sessionPreset = AVCaptureSessionPreset1920x1080;
        } else if (config->width >= 1280 && config->height >= 720 && [session canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
            session.sessionPreset = AVCaptureSessionPreset1280x720;
        } else if (config->width <= 640 && config->height <= 480 && [session canSetSessionPreset:AVCaptureSessionPreset640x480]) {
            session.sessionPreset = AVCaptureSessionPreset640x480;
        } else if ([session canSetSessionPreset:AVCaptureSessionPresetHigh]) {
            session.sessionPreset = AVCaptureSessionPresetHigh;
        }

        AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
        output.alwaysDiscardsLateVideoFrames = YES;
        output.videoSettings = @{ (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };

        RCRE_AVFReceiver* receiver = [[RCRE_AVFReceiver alloc] init];
        dispatch_queue_t queue = dispatch_queue_create("com.rcre.camera.queue", DISPATCH_QUEUE_SERIAL);
        [output setSampleBufferDelegate:receiver queue:queue];

        if (![session canAddOutput:output]) return NULL;
        [session addOutput:output];

        [session startRunning];

        RCRE_Camera* cam = (RCRE_Camera*)rcre_arena_alloc(arena, sizeof(RCRE_Camera), 64);
        AVFCamContext* ctx = (AVFCamContext*)rcre_arena_alloc(arena, sizeof(AVFCamContext), 64);
        if (!cam || !ctx) {
            [session stopRunning];
            return NULL;
        }

        ctx->session = session;
        ctx->receiver = receiver;
        ctx->queue = queue;

        cam->config = *config;
        cam->backend_name = "macOS AVFoundation";
        cam->backend_handle = ctx;
        cam->grab_fn = avf_grab;
        cam->close_fn = avf_close;

        return cam;
    }
}
