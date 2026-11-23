/**
 * @file gpu_metal.m
 * @brief Optional Apple Metal Compute Offload Backend for SGEMM (Extras Layer)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "extras/rcre_gpu.h"
#include <string.h>

static id<MTLDevice> g_mtl_device = nil;
static id<MTLCommandQueue> g_mtl_queue = nil;
static id<MTLComputePipelineState> g_gemm_pso = nil;
static char g_gpu_name[128] = "CPU HAL Fallback";
static bool g_mtl_inited = false;

static const char* g_metal_gemm_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Params { uint M; uint N; uint K; uint lda; uint ldb; uint ldc; float alpha; float beta; };\n"
    "kernel void rcre_sgemm(\n"
    "    device const float* A [[buffer(0)]],\n"
    "    device const float* B [[buffer(1)]],\n"
    "    device float* C       [[buffer(2)]],\n"
    "    constant Params& p    [[buffer(3)]],\n"
    "    uint2 gid             [[thread_position_in_grid]]\n"
    ") {\n"
    "    uint row = gid.y;\n"
    "    uint col = gid.x;\n"
    "    if (row >= p.M || col >= p.N) return;\n"
    "    float acc = 0.0f;\n"
    "    for (uint k = 0; k < p.K; ++k) {\n"
    "        acc += A[row * p.lda + k] * B[k * p.ldb + col];\n"
    "    }\n"
    "    float prev = (p.beta != 0.0f) ? C[row * p.ldc + col] * p.beta : 0.0f;\n"
    "    C[row * p.ldc + col] = p.alpha * acc + prev;\n"
    "}\n";

static void rcre_metal_ensure_init(void) {
    if (g_mtl_inited) return;
    g_mtl_inited = true;

    @autoreleasepool {
        g_mtl_device = MTLCreateSystemDefaultDevice();
        if (!g_mtl_device) return;
        g_mtl_queue = [g_mtl_device newCommandQueue];
        if (!g_mtl_queue) { g_mtl_device = nil; return; }

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:g_metal_gemm_src];
        id<MTLLibrary> lib = [g_mtl_device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { g_mtl_device = nil; return; }

        id<MTLFunction> fn = [lib newFunctionWithName:@"rcre_sgemm"];
        if (!fn) { g_mtl_device = nil; return; }

        g_gemm_pso = [g_mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_gemm_pso) { g_mtl_device = nil; return; }

        snprintf(g_gpu_name, sizeof(g_gpu_name), "%s (Metal)", [[g_mtl_device name] UTF8String]);
    }
}

RCRE_NODISCARD bool rcre_gpu_is_available(void) {
    rcre_metal_ensure_init();
    return g_gemm_pso != nil;
}

RCRE_NODISCARD const char* rcre_gpu_device_name(void) {
    rcre_metal_ensure_init();
    return g_gpu_name;
}

RCRE_NODISCARD int rcre_gpu_gemm_f32(
    size_t M, size_t N, size_t K,
    float alpha,
    const float* A, size_t lda,
    const float* B, size_t ldb,
    float beta,
    float* C, size_t ldc
) {
    if (!A || !B || !C || M == 0 || N == 0 || K == 0) return RCRE_ERR_INVALID_ARG;
    rcre_metal_ensure_init();

    /* For tiny matrices or headless VMs without GPU, CPU HAL avoids driver dispatch latency */
    if (!g_gemm_pso || M * N < 4096) {
        rcre_hal_init();
        rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return RCRE_OK;
    }

    @autoreleasepool {
        size_t bytesA = M * lda * sizeof(float);
        size_t bytesB = K * ldb * sizeof(float);
        size_t bytesC = M * ldc * sizeof(float);

        id<MTLBuffer> bufA = [g_mtl_device newBufferWithBytes:A length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [g_mtl_device newBufferWithBytes:B length:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_mtl_device newBufferWithBytes:C length:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) {
            rcre_hal_init();
            rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return RCRE_OK;
        }

        struct {
            uint32_t M, N, K, lda, ldb, ldc;
            float alpha, beta;
        } params = {
            (uint32_t)M, (uint32_t)N, (uint32_t)K,
            (uint32_t)lda, (uint32_t)ldb, (uint32_t)ldc,
            alpha, beta
        };

        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_gemm_pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];

        MTLSize grid = MTLSizeMake(N, M, 1);
        MTLSize tpg  = MTLSizeMake(16, 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(C, [bufC contents], bytesC);
    }
    return RCRE_OK;
}
