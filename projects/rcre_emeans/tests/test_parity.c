/**
 * @file test_parity.c
 * @brief T3/T4 — strict state-machine bit parity vs the Python oracle.
 *
 * The oracle (ref/ref_emeans.py, make refs) runs the documented FP32
 * operation order and emits per-update (labels, sums, counts, objective)
 * streams plus post-update model states and the revival U draws. This
 * suite replays the streams through the C state machine (em_state_update)
 * and requires 0 ULP on every model state, the finalization result and
 * the FP64 pass sums/counts. The C assignment path is deliberately NOT
 * involved (labels are injected) — it is gated separately in
 * test_assign.c.
 */
#include "em_util.h"

#ifdef EM_STREAMS_AVAILABLE
#include "streams/strict_norev.h"
#include "streams/strict_revive.h"

typedef struct EMCaseResult {
    int state_checks;
    int final_checks;
} EMCaseResult;

/* Replay one strict case: read the payload, drive the C state machine with
 * injected streams, compare every post-update state bit-for-bit. */
static void run_strict_case(const struct em_stream_meta* m,
                            const char* bin_path, const char* x_path) {
    size_t xsz = 0;
    float* X = (float*)em_read_file(x_path, &xsz);
    ASSERT_NOT_NULL(X);
    ASSERT_UINT_EQ(m->n * m->d, xsz / sizeof(float));

    size_t bs = 0;
    uint8_t* buf = (uint8_t*)em_read_file(bin_path, &bs);
    ASSERT_NOT_NULL(buf);

    /* Layout: init c | updates | states | U checks | final */
    size_t kd = m->k * m->d;
    size_t off = 0;
    float* init_c = (float*)(buf + off);
    off += kd * sizeof(float);

    size_t* u_off = malloc(m->n_updates * sizeof(size_t));
    for (size_t i = 0; i < m->n_updates; i++) {
        u_off[i] = off;
        uint32_t mm;
        memcpy(&mm, buf + off, 4); off += 4;
        ASSERT_UINT_EQ(m->update_rows[i], mm);
        off += (size_t)mm * sizeof(int32_t)
             + kd * sizeof(float)
             + m->k * sizeof(float);
    }
    size_t state_off = off;
    off += m->n_updates * (kd * sizeof(float) + m->k * sizeof(float));
    size_t U_off = off;
    off += m->n_U_checks * m->k * sizeof(double);
    size_t final_off = off;
    size_t expect_total = final_off
        + kd * sizeof(float) + m->k * sizeof(float)
        + kd * sizeof(double) + m->k * sizeof(double);
    ASSERT_UINT_EQ(expect_total, bs);

    /* Build the state (serial, no pool). */
    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    RCRE_Tensor Xt = em_tensor2(arena, m->n, m->d, RCRE_DTYPE_F32);
    memcpy(Xt.data, X, m->n * m->d * sizeof(float));
    RCRE_Tensor Ct = em_tensor2(arena, m->k, m->d, RCRE_DTYPE_F32);
    memcpy(Ct.data, init_c, kd * sizeof(float));

    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = m->T;
    p.batch_size = m->B;
    p.alpha0 = (float)m->alpha0_full; /* fixtures use F32-exact values */
    p.unused_threshold = m->tau;
    p.shuffle = false;
    p.init_centroids = &Ct;
    p.seed = m->seed;
    p.capture_labels = false;

    EMState st;
    int rc = em_state_init(&st, &Xt, m->k, &p, arena, NULL);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_MEM_EQ(init_c, st.c, kd * sizeof(float));   /* model start exact */

    /* Inject the oracle's revival U stream (no-op when tau == 0). */
    if (m->n_U_checks > 0) {
        st.inject_U = (const double*)(buf + U_off);
        st.inject_U_len = m->n_U_checks * m->k;
        st.inject_U_pos = 0;
    }

    /* Replay, comparing every post-update state. */
    int state_checks = 0;
    size_t epoch_rows_arr[64];
    size_t ne = 0;
    size_t whole = (size_t)(double)m->T >= 1.0 ? (size_t)m->T : 0;
    (void)whole; (void)epoch_rows_arr;
    /* Derive the final-epoch update range the same way the oracle does:
     * epoch rows from T; the last epoch's updates are the tail. */
    size_t rows_arr[64];
    ne = 0;
    {
        double T = (double)m->T;
        size_t w = (size_t)T;
        size_t rem = em_epoch_remainder(m->T, m->n);
        if (w == 0) { rows_arr[0] = rem; ne = 1; }
        else {
            for (size_t i = 0; i + 1 < w; i++) rows_arr[ne++] = m->n;
            if (rem > 0) rows_arr[ne++] = rem;
            rows_arr[ne++] = m->n;
        }
    }
    size_t final_start_updates = 0;
    for (size_t e = 0; e + 1 < ne; e++)
        final_start_updates += (rows_arr[e] + m->B - 1) / m->B;

    for (size_t t = 0; t < m->n_updates; t++) {
        st.last_epoch = (t >= final_start_updates);
        if (t == final_start_updates && ne > 0) {
            memset(st.pass_sums, 0, kd * sizeof(double));
            memset(st.pass_counts, 0, m->k * sizeof(double));
            st.obj_last = 0.0;
            st.obj_last_rows = 0;
        }
        uint8_t* up = buf + u_off[t];
        uint32_t mm;
        memcpy(&mm, up, 4);
        int32_t* labels = (int32_t*)(up + 4);
        float* sums = (float*)(up + 4 + (size_t)mm * sizeof(int32_t));
        float* counts = (float*)(sums + kd);
        (void)labels; /* injected labels are not re-verified here */

        rc = em_state_update(&st, sums, counts, 0.0);
        ASSERT_INT_EQ(RCRE_OK, rc);

        const uint8_t* exp = buf + state_off
            + t * (kd * sizeof(float) + m->k * sizeof(float));
        ASSERT_MEM_EQ(exp, st.c, kd * sizeof(float));      /* 0 ULP, c */
        ASSERT_MEM_EQ(exp + kd * sizeof(float), st.N,
                      m->k * sizeof(float));               /* 0 ULP, N */
        state_checks += 2;
    }

    /* Finalization (T4): FP64 pass sums + FP32 division. */
    rc = em_state_finalize(&st);
    ASSERT_INT_EQ(RCRE_OK, rc);
    const uint8_t* fin = buf + final_off;
    ASSERT_MEM_EQ(fin, st.c, kd * sizeof(float));                  /* 0 ULP */
    ASSERT_MEM_EQ(fin + kd * sizeof(float), st.N, m->k * sizeof(float));
    ASSERT_MEM_EQ(fin + kd * sizeof(float) + m->k * sizeof(float),
                  st.pass_sums, kd * sizeof(double));              /* FP64 */
    ASSERT_MEM_EQ(fin + kd * sizeof(float) + m->k * sizeof(float)
                  + kd * sizeof(double),
                  st.pass_counts, m->k * sizeof(double));          /* FP64 */
    int final_checks = 4;

    printf("  [%s] %zu updates, %d state checks, %d final checks, "
           "%zu U-checks replayed, splits=%d\n",
           bin_path, m->n_updates, state_checks, final_checks,
           m->n_U_checks, st.n_splits);

    free(u_off);
    free(buf);
    free(X);
    rcre_arena_destroy(arena);
}

static void test_strict_norev(void) {
    run_strict_case(&EM_STREAM_STRICT_NOREV,
                    "streams/strict_norev.bin", "streams/X_strict_norev.f32");
}

static void test_strict_revive(void) {
    /* The fixture guarantees splits occurred and the Gumbel key margin is
     * away from the libm-log tie zone (the oracle prints the margin). */
    run_strict_case(&EM_STREAM_STRICT_REVIVE,
                    "streams/strict_revive.bin", "streams/X_strict_revive.f32");
}
#endif /* EM_STREAMS_AVAILABLE */

int main(void) {
    test_suite_begin("e-means T3/T4: strict state-machine bit parity");
#ifdef EM_STREAMS_AVAILABLE
    TEST_RUN(test_strict_norev);
    TEST_RUN(test_strict_revive);
#else
    printf("  SKIP: oracle streams absent (run `make refs`) — "
           "degraded coverage\n");
#endif
    return test_suite_report();
}
