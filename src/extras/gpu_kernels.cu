/* RCRE H6 — CUDA kernels (2.3.0)
 *
 * All kernels use FIXED grid/block configurations (no device-query
 * auto-tuning), so a launch is deterministic on a given device:
 *
 *   gemm f32/f64  tiled shared-memory; one thread per output element,
 *                 sequential k inside the thread (no cross-thread
 *                 reduction for the same element)
 *   spmv          one thread per row, sequential row walk
 *   axpy          grid-stride, one element per thread
 *   dot           one block (256 threads): strided sequential
 *                 accumulation + fixed shared-memory tree reduction
 *                 (no atomics — atomics are non-deterministic)
 *   philox4x32    one thread per 4 words; the round function is the
 *                 exact curand _philox4x32round (bit-identical to the
 *                 C core rcre_rng_philox4x32)
 *
 * Compiled by nvcc only when the build has CUDA (RCRE_WITH_CUDA is
 * irrelevant here — this file is only compiled by nvcc at all).
 */
#include <cuda_runtime.h>
#include <stdint.h>

/* C linkage for the host wrappers (called from the C bridge gpu_cuda.c). */
extern "C" {
int rcre_gpu_cuda_gemm_f32(const float* A, size_t lda, const float* B,
                           size_t ldb, float* C, size_t ldc, size_t m,
                           size_t n, size_t k, float alpha, float beta);
int rcre_gpu_cuda_gemm_f64(const double* A, size_t lda, const double* B,
                           size_t ldb, double* C, size_t ldc, size_t m,
                           size_t n, size_t k, double alpha, double beta);
int rcre_gpu_cuda_spmv_f32(const size_t* row_ptr, const size_t* col_idx,
                           const float* values, const float* x, float* y,
                           size_t rows);
int rcre_gpu_cuda_spmv_f64(const size_t* row_ptr, const size_t* col_idx,
                           const double* values, const double* x, double* y,
                           size_t rows);
int rcre_gpu_cuda_axpy_f32(float a, const float* x, float* y, size_t n);
int rcre_gpu_cuda_axpy_f64(double a, const double* x, double* y, size_t n);
int rcre_gpu_cuda_dot_f32(const float* x, const float* y, float* out, size_t n);
int rcre_gpu_cuda_dot_f64(const double* x, const double* y, double* out,
                          size_t n);
int rcre_gpu_cuda_philox_fill(uint32_t* out, size_t n, uint64_t counter0,
                              uint64_t key);
}

#define RCRE_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            return (int)e_;                                                   \
        }                                                                     \
    } while (0)

/* ========================================================================= */
/* GEMM (row-major, C = alpha*A*B + beta*C)                                  */
/* ========================================================================= */

template <typename T, int TM, int TN, int TK>
__global__ void gemm_kernel(const T* A, size_t lda, const T* B, size_t ldb,
                            T* C, size_t ldc, size_t m, size_t n, size_t k,
                            T alpha, T beta) {
    /* thread (tx,ty) owns C[by*TM + ty][bx*TN + tx] */
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t row = (size_t)blockIdx.y * TM + ty;
    const size_t col = (size_t)blockIdx.x * TN + tx;
    __shared__ T As[TM][TK];
    __shared__ T Bs[TK][TN];
    T acc = T(0);
    T c0 = (row < m && col < n) ? C[row * ldc + col] : T(0);
    for (size_t k0 = 0; k0 < k; k0 += TK) {
        /* cooperative tile load (bounds-guarded) */
        for (int kk = 0; kk < TK; ++kk) {
            size_t ki = k0 + kk;
            As[ty][kk] = (row < m && ki < k) ? A[row * lda + ki] : T(0);
        }
        for (int kk = 0; kk < TK; ++kk) {
            size_t ki = k0 + kk;
            Bs[kk][tx] = (ki < k && col < n) ? B[ki * ldb + col] : T(0);
        }
        __syncthreads();
        for (int kk = 0; kk < TK; ++kk) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (row < m && col < n)
        C[row * ldc + col] = alpha * acc + beta * c0;
}

int rcre_gpu_cuda_gemm_f32(const float* A, size_t lda, const float* B,
                           size_t ldb, float* C, size_t ldc, size_t m,
                           size_t n, size_t k, float alpha, float beta) {
    const int TM = 32, TN = 32, TK = 8;
    dim3 block(TN, TM);
    dim3 grid((unsigned)((n + TN - 1) / TN), (unsigned)((m + TM - 1) / TM));
    gemm_kernel<float, TM, TN, TK><<<grid, block>>>(A, lda, B, ldb, C, ldc,
                                                    m, n, k, alpha, beta);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

int rcre_gpu_cuda_gemm_f64(const double* A, size_t lda, const double* B,
                           size_t ldb, double* C, size_t ldc, size_t m,
                           size_t n, size_t k, double alpha, double beta) {
    const int TM = 16, TN = 16, TK = 8;
    dim3 block(TN, TM);
    dim3 grid((unsigned)((n + TN - 1) / TN), (unsigned)((m + TM - 1) / TM));
    gemm_kernel<double, TM, TN, TK><<<grid, block>>>(A, lda, B, ldb, C, ldc,
                                                     m, n, k, alpha, beta);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

/* ========================================================================= */
/* SpMV (CSR, size_t indices)                                                */
/* ========================================================================= */

__global__ void spmv_kernel_f32(const size_t* row_ptr, const size_t* col_idx,
                                const float* values, const float* x,
                                float* y, size_t rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float s = 0.0f;
    for (size_t jj = row_ptr[i]; jj < row_ptr[i + 1]; ++jj)
        s += values[jj] * x[col_idx[jj]];
    y[i] = s;
}

__global__ void spmv_kernel_f64(const size_t* row_ptr, const size_t* col_idx,
                                const double* values, const double* x,
                                double* y, size_t rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    double s = 0.0;
    for (size_t jj = row_ptr[i]; jj < row_ptr[i + 1]; ++jj)
        s += values[jj] * x[col_idx[jj]];
    y[i] = s;
}

int rcre_gpu_cuda_spmv_f32(const size_t* row_ptr, const size_t* col_idx,
                           const float* values, const float* x, float* y,
                           size_t rows) {
    const int TPB = 256;
    spmv_kernel_f32<<<(unsigned)((rows + TPB - 1) / TPB), TPB>>>(
        row_ptr, col_idx, values, x, y, rows);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

int rcre_gpu_cuda_spmv_f64(const size_t* row_ptr, const size_t* col_idx,
                           const double* values, const double* x, double* y,
                           size_t rows) {
    const int TPB = 256;
    spmv_kernel_f64<<<(unsigned)((rows + TPB - 1) / TPB), TPB>>>(
        row_ptr, col_idx, values, x, y, rows);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

/* ========================================================================= */
/* AXPY (grid-stride: one element per thread)                                */
/* ========================================================================= */

__global__ void axpy_kernel_f32(float a, const float* x, float* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void axpy_kernel_f64(double a, const double* x, double* y,
                                size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int rcre_gpu_cuda_axpy_f32(float a, const float* x, float* y, size_t n) {
    const int TPB = 256;
    axpy_kernel_f32<<<(unsigned)((n + TPB - 1) / TPB), TPB>>>(a, x, y, n);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

int rcre_gpu_cuda_axpy_f64(double a, const double* x, double* y, size_t n) {
    const int TPB = 256;
    axpy_kernel_f64<<<(unsigned)((n + TPB - 1) / TPB), TPB>>>(a, x, y, n);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

/* ========================================================================= */
/* DOT — one block, fixed-order tree reduction (deterministic, no atomics)   */
/* ========================================================================= */

template <typename T>
__global__ void dot_kernel(const T* x, const T* y, T* out, size_t n) {
    __shared__ T s[256];
    const int tid = threadIdx.x; /* blockDim.x == 256 */
    T acc = T(0);
    for (size_t i = (size_t)tid; i < n; i += 256) acc += x[i] * y[i];
    s[tid] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}

int rcre_gpu_cuda_dot_f32(const float* x, const float* y, float* out,
                          size_t n) {
    dot_kernel<float><<<1, 256>>>(x, y, out, n);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

int rcre_gpu_cuda_dot_f64(const double* x, const double* y, double* out,
                          size_t n) {
    dot_kernel<double><<<1, 256>>>(x, y, out, n);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}

/* ========================================================================= */
/* Philox-4x32-10 — exact curand round function                              */
/* ========================================================================= */

#define PHILOX_W32_0 0x9E3779B9u
#define PHILOX_W32_1 0xBB67AE85u
#define PHILOX_M4X32_0 0xD2511F53u
#define PHILOX_M4X32_1 0xCD9E8D57u

/* One round, identical to curand _philox4x32round (CUDA 13.3):
 * mulhilo32 split + XOR mix. */
__device__ __forceinline__ void philox_round(uint4* c, uint2* k) {
    uint64_t p0 = (uint64_t)PHILOX_M4X32_0 * (uint64_t)c->x;
    uint64_t p1 = (uint64_t)PHILOX_M4X32_1 * (uint64_t)c->z;
    uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
    uint4 n;
    n.x = hi1 ^ c->y ^ k->x;
    n.y = lo1;
    n.z = hi0 ^ c->w ^ k->y;
    n.w = lo0;
    *c = n;
    k->x += PHILOX_W32_0;
    k->y += PHILOX_W32_1;
}

__global__ void philox_fill_kernel(uint32_t* out, size_t n, uint64_t counter0,
                                   uint64_t key) {
    /* one thread per 4 words */
    size_t w4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = w4 * 4;
    if (base >= n) return;
    uint4 c;
    c.x = (uint32_t)(counter0 + w4);
    c.y = (uint32_t)((counter0 + w4) >> 32);
    c.z = 0u;
    c.w = 0u;
    uint2 k;
    k.x = (uint32_t)(key & 0xFFFFFFFFu);
    k.y = (uint32_t)(key >> 32);
    for (int r = 0; r < 10; ++r) philox_round(&c, &k);
    uint4* o = (uint4*)(out + base);
    if (base + 4 <= n) {
        *o = c;
    } else {
        for (size_t i = base; i < n; ++i) out[i] = ((const uint32_t*)&c)[i - base];
    }
}

int rcre_gpu_cuda_philox_fill(uint32_t* out, size_t n, uint64_t counter0,
                              uint64_t key) {
    const int TPB = 256;
    size_t n4 = (n + 3) / 4;
    philox_fill_kernel<<<(unsigned)((n4 + TPB - 1) / TPB), TPB>>>(
        out, n, counter0, key);
    RCRE_CHECK(cudaGetLastError());
    return 0;
}
