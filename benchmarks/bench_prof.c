/**
 * @file bench_prof.c
 * @brief RCRE Profiling Workload (rcre_prof) — exercises every telemetry pillar
 *
 * Workload (all scopes recorded + OS/arena/IO samples at stage boundaries):
 *   1. arena_setup      — 256MB dual arena, bulk allocations
 *   2. hal_gemm_512     — 20x NEON/AVX2 GEMM via rcre_hal (GFLOPS marked)
 *   3. cv_grayscale     — 1080p RGB->gray via rcre_hal (MP/s marked)
 *   4. gpt_init         — Micro-GPT (V=256, T=64, C=128, L=4) weights init
 *   5. gpt_inference    — 64-token autoregressive loop
 *                          (forward_token + sample_token; P1: zero-heap sampling)
 *   6. checkpoint_io    — real disk: rcre_gpt_save_checkpoint / load_checkpoint
 *                          (engine I/O recorded in the app-level ledger)
 *
 * Outputs: ANSI terminal report + Chrome/Perfetto JSON + HTML dashboard.
 * Usage:   bin/bench_prof [out.json] [out.html]
 */

#include "rcre.h"
#include "rcre_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    const char* json_path = (argc > 1) ? argv[1] : "rcre_profile.json";
    const char* html_path = (argc > 2) ? argv[2] : "rcre_profile.html";
    const char* ckpt_path = "/tmp/rcre_prof_checkpoint.bin";

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("================================================================\n");
    printf("   RCRE Profiling Workload (rcre_prof) — M-series / x86 portable\n");
    printf("================================================================\n");

    rcre_hal_init();
    RCRE_ProfContext* prof = rcre_prof_init(8192);
    if (!prof) { fprintf(stderr, "prof init failed\n"); return 1; }

    /* ---------- 1. arena setup ---------- */
    RCRE_PROFILE_SCOPE(prof, "arena_setup", z1);
    RCRE_Arena* arena = rcre_arena_create(256 * 1024 * 1024);
    if (!arena) { fprintf(stderr, "arena failed\n"); return 1; }
    float* A = (float*)rcre_arena_alloc(arena, 512 * 512 * sizeof(float), 64);
    float* B = (float*)rcre_arena_alloc(arena, 512 * 512 * sizeof(float), 64);
    float* C = (float*)rcre_arena_alloc(arena, 512 * 512 * sizeof(float), 64);
    RCRE_PROFILE_END(z1);
    rcre_prof_sample_rss(prof);
    rcre_prof_sample_arena(prof, arena);

    /* ---------- 2. HAL GEMM ---------- */
    for (size_t i = 0; i < 512 * 512; i++) { A[i] = (float)((i % 13) - 6) * 0.1f; B[i] = (float)((i % 17) - 8) * 0.1f; }
    {
        const size_t N = 512;
        const int runs = 20;
        RCRE_PROFILE_SCOPE(prof, "hal_gemm_512", z2);
        double t0 = now_sec();
        for (int r = 0; r < runs; r++) {
            rcre_hal.gemm_f32(N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        }
        double dt = now_sec() - t0;
        RCRE_PROFILE_END(z2);
        double gflops = (2.0 * N * N * N) * runs / 1e9 / dt;
        rcre_prof_mark(prof, "GFLOPS", gflops);
        printf("[1/6] HAL GEMM 512^3 x%d (%s): %.2f ms, %.2f GFLOPS\n",
               runs, rcre_hal.backend_name, dt * 1e3, gflops);
    }

    /* ---------- 3. CV grayscale ---------- */
    {
        const size_t W = 1920, H = 1080, px = W * H;
        uint8_t* rgb = (uint8_t*)rcre_arena_alloc(arena, px * 3, 64);
        uint8_t* gray = (uint8_t*)rcre_arena_alloc(arena, px, 64);
        for (size_t i = 0; i < px * 3; i++) rgb[i] = (uint8_t)(i * 33 + 7);
        const int runs = 50;
        RCRE_PROFILE_SCOPE(prof, "cv_grayscale_1080p", z3);
        double t0 = now_sec();
        for (int r = 0; r < runs; r++) rcre_hal.rgb_to_gray_u8(rgb, gray, px);
        double dt = now_sec() - t0;
        RCRE_PROFILE_END(z3);
        double mps = (double)px * runs / 1e6 / dt;
        rcre_prof_mark(prof, "MPS", mps);
        printf("[2/6] CV grayscale 1080p x%d: %.2f ms, %.1f MP/s\n", runs, dt * 1e3, mps);
    }
    rcre_prof_sample_rss(prof);
    rcre_prof_sample_arena(prof, arena);

    /* ---------- 4+5. GPT init + autoregressive inference ---------- */
    RCRE_GPTModel* model = NULL;
    RCRE_KVCache* cache = NULL;
    float* logits_buf = NULL;
    {
        RCRE_GPTConfig cfg;
        cfg.vocab_size = 256;
        cfg.seq_len = 64;
        cfg.embed_dim = 128;
        cfg.num_heads = 4;
        cfg.head_dim = 32;
        cfg.num_layers = 4;
        cfg.mlp_dim = 512;

        RCRE_PROFILE_SCOPE(prof, "gpt_init", z4);
        model = rcre_gpt_create(cfg, arena);
        if (model) {
            uint64_t rng = 0xC0FFEE1234567890ULL;
            rcre_gpt_init_weights(model, &rng);
        }
        RCRE_PROFILE_END(z4);
        if (!model) { fprintf(stderr, "gpt create failed\n"); return 1; }
        printf("[3/6] GPT (V=256,T=64,C=128,L=4) initialized on arena (HWM %.1f MB)\n",
               (double)arena->high_watermark / 1e6);

        cache = rcre_kv_cache_create(cfg, arena);
        logits_buf = (float*)rcre_arena_alloc(arena, 256 * sizeof(float), 64);

        const int T = 64;
        uint64_t rng = 0x9E3779B97F4A7C15ULL;
        RCRE_PROFILE_SCOPE(prof, "gpt_inference_64tok", z5);
        double t0 = now_sec();
        int tok = 1;
        for (int pos = 0; pos < T; pos++) {
            RCRE_PROFILE_SCOPE(prof, "gpt_forward_token", zf);
            if (!rcre_gpt_forward_token(model, cache, tok, pos, logits_buf)) {
                fprintf(stderr, "forward_token failed at pos %d\n", pos);
                return 1;
            }
            RCRE_PROFILE_END(zf);
            RCRE_PROFILE_SCOPE(prof, "gpt_sample_token", zs);
            tok = rcre_gpt_sample_token(logits_buf, 256, 1.0f, 50, &rng);
            if (tok < 0) { fprintf(stderr, "sample failed at pos %d\n", pos); return 1; }
            RCRE_PROFILE_END(zs);
        }
        double dt = now_sec() - t0;
        RCRE_PROFILE_END(z5);
        rcre_prof_mark(prof, "TOK_PER_SEC", T / dt);
        printf("[4/6] GPT inference: %d tokens in %.2f ms (%.1f tok/s, sampling zero-heap)\n",
               T, dt * 1e3, T / dt);
    }
    rcre_prof_sample_rss(prof);
    rcre_prof_sample_arena(prof, arena);

    /* ---------- 6. checkpoint I/O (real disk, engine ledger) ---------- */
    {
        size_t ckpt_bytes = 0;
        RCRE_PROFILE_SCOPE(prof, "checkpoint_save", z6);
        bool ok = rcre_gpt_save_checkpoint(model, ckpt_path);
        RCRE_PROFILE_END(z6);
        if (ok) {
            struct stat st;
            if (stat(ckpt_path, &st) == 0) {
                ckpt_bytes = (size_t)st.st_size;
                rcre_prof_io_record(prof, true, ckpt_bytes);
            }
            rcre_prof_mark(prof, "CKPT_BYTES", (double)ckpt_bytes);
        } else {
            fprintf(stderr, "checkpoint save failed\n");
        }

        RCRE_PROFILE_SCOPE(prof, "checkpoint_load", z7);
        bool ok2 = rcre_gpt_load_checkpoint(model, ckpt_path);
        RCRE_PROFILE_END(z7);
        if (ok2) rcre_prof_io_record(prof, false, ckpt_bytes);
        printf("[5/6] Checkpoint I/O: %zu bytes saved+loaded (engine ledger)\n", ckpt_bytes);
        unlink(ckpt_path);
    }

    rcre_prof_sample_rss(prof);
    rcre_prof_sample_arena(prof, arena);
    rcre_prof_sample_io(prof);

    /* ---------- report ---------- */
    rcre_prof_print_summary(prof, stdout);
    if (rcre_prof_export_perfetto_json(prof, json_path)) {
        printf("\n   Perfetto JSON: %s (open https://ui.perfetto.dev)\n", json_path);
    }
    if (rcre_prof_export_html(prof, html_path)) {
        printf("   HTML dashboard: %s\n", html_path);
    }

    rcre_gpt_free(model);
    rcre_prof_free(prof);
    rcre_arena_destroy(arena);
    printf("\n✅ Profiling workload complete.\n");
    return 0;
}
