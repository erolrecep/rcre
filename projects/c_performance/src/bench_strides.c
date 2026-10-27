/**
 * @file bench_strides.c
 * @brief Benchmark 02: Strided Tensor Views vs Contiguous Memory Re-layout
 */

#include "rcre_perf.h"
#include <inttypes.h>

void rcre_bench_strided_tensor(const RCRE_PerfConfig* cfg, RCRE_ProfContext* prof, RCRE_Arena* arena) {
    const size_t N = cfg->matrix_size;
    const size_t shape[2] = { N, N };
    const size_t num_elements = N * N;
    const size_t total_bytes = num_elements * sizeof(float);
    const double total_mb = (double)total_bytes / (1024.0 * 1024.0);

    printf("\n===================================================================\n");
    printf(" 📊 [Benchmark 02] Strided Tensor Views vs Contiguous Re-layout   \n");
    printf("===================================================================\n");
    printf(" Tensor Dimensions      : [%zu, %zu] (Rank 2, F32)\n", N, N);
    printf(" Tensor Footprint       : %.2f MB\n", total_mb);
    printf(" Concept                : ShapeTracker Zero-Copy Views vs Materialization\n");
    printf("-------------------------------------------------------------------\n");

    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);

    /* 1. Allocate base contiguous tensor */
    RCRE_Tensor t_base = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* base_data = rcre_tensor_f32_data(&t_base);
    for (size_t i = 0; i < num_elements; ++i) {
        base_data[i] = (float)(i % 100) * 0.01f;
    }

    /* --------------------------------------------------------------------- */
    /* 1. Contiguous Reduction (stride = [N, 1])                             */
    /* --------------------------------------------------------------------- */
    double sum_cont = 0.0;
    uint64_t dur_cont_ns = 0;

    for (size_t iter = 0; iter < cfg->iterations; ++iter) {
        RCRE_PROFILE_SCOPE(prof, "stride_contiguous", z_cont);
        const ptrdiff_t s0 = t_base.strides[0];
        const ptrdiff_t s1 = t_base.strides[1];
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sum_cont += base_data[i * s0 + j * s1];
            }
        }
        dur_cont_ns += RCRE_PROFILE_END(z_cont);
    }
    dur_cont_ns /= cfg->iterations;
    rcre_perf_do_not_optimize(&sum_cont);

    double time_cont_sec = (double)dur_cont_ns * 1e-9;
    double bw_cont = (total_mb / 1024.0) / time_cont_sec;

    /* --------------------------------------------------------------------- */
    /* 2. Zero-Copy Transpose View (stride = [1, N])                         */
    /* Invariant: 0 additional bytes allocated, O(1) stride swap             */
    /* --------------------------------------------------------------------- */
    uint64_t t_view_start = rcre_prof_now_ns();
    RCRE_Tensor t_view = rcre_tensor_transpose(&t_base, 0, 1);
    uint64_t t_view_dur = rcre_prof_now_ns() - t_view_start;

    double sum_view = 0.0;
    uint64_t dur_view_ns = 0;

    for (size_t iter = 0; iter < cfg->iterations; ++iter) {
        RCRE_PROFILE_SCOPE(prof, "stride_zerocopy_view", z_view);
        const ptrdiff_t s0 = t_view.strides[0];
        const ptrdiff_t s1 = t_view.strides[1];
        const float* vdata = rcre_tensor_f32_data(&t_view);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sum_view += vdata[i * s0 + j * s1];
            }
        }
        dur_view_ns += RCRE_PROFILE_END(z_view);
    }
    dur_view_ns /= cfg->iterations;
    rcre_perf_do_not_optimize(&sum_view);

    double time_view_sec = (double)dur_view_ns * 1e-9;
    double bw_view = (total_mb / 1024.0) / time_view_sec;

    /* --------------------------------------------------------------------- */
    /* 3. Physical Materialization / Repack                                  */
    /* Allocates new contiguous buffer and repacks elements in row-major.    */
    /* --------------------------------------------------------------------- */
    uint64_t t_pack_start = rcre_prof_now_ns();
    RCRE_Tensor t_packed = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* pdata = rcre_tensor_f32_data(&t_packed);
    const float* vdata_src = rcre_tensor_f32_data(&t_view);
    const ptrdiff_t vs0 = t_view.strides[0];
    const ptrdiff_t vs1 = t_view.strides[1];
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            pdata[i * N + j] = vdata_src[i * vs0 + j * vs1];
        }
    }
    uint64_t t_pack_dur = rcre_prof_now_ns() - t_pack_start;

    double sum_packed = 0.0;
    uint64_t dur_packed_ns = 0;

    for (size_t iter = 0; iter < cfg->iterations; ++iter) {
        RCRE_PROFILE_SCOPE(prof, "stride_repacked_contiguous", z_packed);
        const ptrdiff_t s0 = t_packed.strides[0];
        const ptrdiff_t s1 = t_packed.strides[1];
        const float* pdata = rcre_tensor_f32_data(&t_packed);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sum_packed += pdata[i * s0 + j * s1];
            }
        }
        dur_packed_ns += RCRE_PROFILE_END(z_packed);
    }
    dur_packed_ns /= cfg->iterations;
    rcre_perf_do_not_optimize(&sum_packed);

    double time_packed_sec = (double)dur_packed_ns * 1e-9;
    double bw_packed = (total_mb / 1024.0) / time_packed_sec;

    /* Report results */
    printf("\n [1] Contiguous Tensor Traversal (Strides [%td, %td]):\n", 
           t_base.strides[0], t_base.strides[1]);
    printf("     Latency    : %7.3f ms\n", time_cont_sec * 1000.0);
    printf("     Throughput : %7.2f GB/s\n", bw_cont);
    printf("     Contiguous : %s\n", rcre_tensor_is_contiguous(&t_base) ? "YES" : "NO");

    printf("\n [2] Zero-Copy Transposed View (Strides [%td, %td]):\n", 
           t_view.strides[0], t_view.strides[1]);
    printf("     Creation   : %" PRIu64 " ns (0 bytes memory allocated!)\n", t_view_dur);
    printf("     Latency    : %7.3f ms\n", time_view_sec * 1000.0);
    printf("     Throughput : %7.2f GB/s\n", bw_view);
    printf("     Contiguous : %s (Zero-Copy View)\n", rcre_tensor_is_contiguous(&t_view) ? "YES" : "NO");

    printf("\n [3] Repacked Physical Contiguity (Strides [%td, %td]):\n", 
           t_packed.strides[0], t_packed.strides[1]);
    printf("     Pack Cost  : %7.3f ms (Allocated %.2f MB in arena)\n", (double)t_pack_dur * 1e-6, total_mb);
    printf("     Latency    : %7.3f ms\n", time_packed_sec * 1000.0);
    printf("     Throughput : %7.2f GB/s\n", bw_packed);
    printf("     Contiguous : %s\n", rcre_tensor_is_contiguous(&t_packed) ? "YES" : "NO");

    printf("\n Architectural Synthesis:\n");
    printf("   • Zero-Copy View Overhead  : Zero allocation cost (%" PRIu64 " ns metadata swap).\n", t_view_dur);
    printf("   • Contiguous vs Strided BW : Contiguous reads are %.2fx faster due to prefetching.\n",
           time_view_sec / (time_cont_sec > 1e-9 ? time_cont_sec : 1e-9));
    printf("   • Break-Even Decision      : If an operation is evaluated > %.1f times, the one-time\n",
           (double)t_pack_dur / (dur_view_ns - dur_packed_ns > 0 ? (double)(dur_view_ns - dur_packed_ns) : 1.0));
    printf("     repack cost pays for itself in amortized runtime performance.\n");

    if (prof) {
        rcre_prof_mark(prof, "STRIDE_CONTIG_GBPS", bw_cont);
        rcre_prof_mark(prof, "STRIDE_VIEW_GBPS", bw_view);
        rcre_prof_mark(prof, "STRIDE_PACKED_GBPS", bw_packed);
    }

    rcre_arena_temp_end(checkpoint);
}
