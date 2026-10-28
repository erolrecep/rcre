/**
 * @file bench_xgb.c
 * @brief RCRE-XGB benchmarks: scaling, per-stage timing, RCRE-predictor
 *        parity, arena high-water.
 *
 * Sections:
 *   1. scaling: fixed d=64, n in {10k, 25k, 50k, 100k, 250k}, 30 rounds
 *   2. per-stage breakdown on 100k x 64 (binning / fit / predict), 200 rounds
 *   3. binary task on the same data (AUC)
 *   4. xgb_predict vs RCRE rcre_ml_gbdt_predict parity + throughput
 *
 * Timing via RCRE's profiler (hardware cycle counter) where useful,
 * CLOCK_MONOTONIC (rcre_prof_now_ns) for wall sections.
 */
#include "rcre.h"
#include "xgb.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static uint64_t S_rng = 777;
static double s_frand(void) {
    S_rng = S_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(S_rng >> 11) / 9007199254740992.0; /* 2^53 */
}
static double s_frandn(void) {
    return (s_frand() + s_frand() + s_frand()) / 3.0 * 2.0 - 1.0;
}

static void gen(size_t n, size_t d, float* X, float* y) {
    for (size_t i = 0; i < n; i++) {
        for (size_t c = 0; c < d; c++) X[i * d + c] = (float)s_frandn();
        double t = tanh(1.5 * X[i * d + 0] + 0.8 * X[i * d + 1]
                       - 0.5 * X[i * d + 2] + 0.3 * sin(X[i * d + 3]));
        t += 0.4 * X[i * d + 4] * X[i * d + 5];
        t += 0.15 * X[i * d + 6] * X[i * d + 7];
        t += 0.25 * s_frandn();
        y[i] = (float)t;
    }
}

static int cmpd(const void* p, const void* q) {
    double x = *(const double*)p, y = *(const double*)q;
    return (x > y) - (x < y);
}

int main(void) {
    rcre_hal_init();
    printf("================================================================\n");
    printf("   RCRE-XGB BENCHMARKS  (host: %s)\n", rcre_cpu_get_brand());
    printf("================================================================\n");

    /* ---------------- 1. scaling ---------------- */
    printf("\n[1] scaling (d=64, 30 rounds, depth 6, regression)\n");
    printf("    %8s %10s %12s %12s %12s %10s\n",
           "n", "ms/round", "total ms", "samples/s", "val RMSE", "trees");
    const size_t D = 64;
    for (int s = 0; s < 5; s++) {
        size_t n = (size_t)(10000 * (s == 0 ? 1 : s == 1 ? 2.5 : s == 2 ? 5 : s == 3 ? 10 : 25));
        size_t ntr = (size_t)(n * 0.8), nval = n - ntr;
        float* X = (float*)malloc(n * D * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));
        gen(n, D, X, y);

        RCRE_Arena* a = rcre_arena_create(512 << 20);
        XGBConfig cfg; xgb_config_default(&cfg);
        cfg.task = XGB_TASK_REGRESSION;
        cfg.rounds = 30; cfg.eta = 0.3f; cfg.max_depth = 6;
        cfg.seed = 42;

        XGBDataset* ds = xgb_dataset_create(X, y, ntr, D, &cfg, a);
        XGBDataset* dsv = xgb_dataset_create(&X[ntr * D], &y[ntr], nval, D, &cfg, a);

        uint64_t t0 = rcre_prof_now_ns();
        XGBModel m;
        xgb_train(ds, dsv, &cfg, &m, a);
        uint64_t ms = (rcre_prof_now_ns() - t0) / 1000000;

        float* pv = (float*)malloc(nval * sizeof(float));
        xgb_predict(&m, &X[ntr * D], nval, D, pv);
        float rmsev = xgb_rmse(&y[ntr], pv, nval);

        printf("    %8zu %10.1f %12.1f %12.0f %12.4f %10d\n",
               n, (double)ms / 30, (double)ms,
               (double)ntr / ((double)ms / 1000.0) / 30,
               (double)rmsev, m.num_trees);

        free(pv);
        rcre_arena_destroy(a);
        free(X); free(y);
    }

    /* ---------------- 2. per-stage breakdown ---------------- */
    printf("\n[2] per-stage (n=100k, d=64, 200 rounds, depth 6)\n");
    {
        size_t n = 100000, ntr = 80000, nval = 20000;
        float* X = (float*)malloc(n * D * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));
        gen(n, D, X, y);

        RCRE_Arena* a = rcre_arena_create(512 << 20);
        XGBConfig cfg; xgb_config_default(&cfg);
        cfg.task = XGB_TASK_REGRESSION;
        cfg.rounds = 200; cfg.eta = 0.3f; cfg.max_depth = 6;
        cfg.early_stop_rounds = 20;
        cfg.seed = 42;

        RCRE_ProfContext* prof = rcre_prof_init(1 << 15);
        RCRE_PROFILE_SCOPE(prof, "binning(2x)", zb);
        XGBDataset* ds = xgb_dataset_create(X, y, ntr, D, &cfg, a);
        XGBDataset* dsv = xgb_dataset_create(&X[ntr * D], &y[ntr], nval, D, &cfg, a);
        rcre_prof_scope_end(&zb);

        RCRE_PROFILE_SCOPE(prof, "training", zt);
        XGBModel m;
        xgb_train(ds, dsv, &cfg, &m, a);
        rcre_prof_scope_end(&zt);

        float* p = (float*)malloc(n * sizeof(float));
        RCRE_PROFILE_SCOPE(prof, "predict(100k)", zp);
        xgb_predict(&m, X, n, D, p);
        rcre_prof_scope_end(&zp);
        rcre_prof_sample_arena(prof, a);

        float* pv = (float*)malloc(nval * sizeof(float));
        xgb_predict(&m, &X[ntr * D], nval, D, pv);
        float rmsev = xgb_rmse(&y[ntr], pv, nval);
        float rmsetr = xgb_rmse(y, &p[0], ntr);
        int ni, nl;
        xgb_model_counts(&m, &ni, &nl);

        printf("    trees=%d (best iter %d)  internal=%d leaves=%d\n",
               m.num_trees, m.best_iteration, ni, nl);
        printf("    train RMSE=%.4f  val RMSE=%.4f\n", (double)rmsetr, (double)rmsev);
        rcre_prof_print_summary(prof, stdout);
        rcre_prof_free(prof);
        free(p); free(pv);
        rcre_arena_destroy(a);
        free(X); free(y);
    }

    /* ---------------- 3. binary ---------------- */
    printf("\n[3] binary task (n=100k, d=64, 200 rounds)\n");
    {
        size_t n = 100000, ntr = 80000, nval = 20000;
        float* X = (float*)malloc(n * D * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));
        gen(n, D, X, y);
        double* srt = (double*)malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) srt[i] = (double)y[i];
        qsort(srt, n, sizeof(double), cmpd);
        double med = srt[n / 2];
        float* yf = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) yf[i] = (y[i] > med) ? 1.0f : 0.0f;

        RCRE_Arena* a = rcre_arena_create(512 << 20);
        XGBConfig cfg; xgb_config_default(&cfg);
        cfg.task = XGB_TASK_BINARY;
        cfg.rounds = 200; cfg.eta = 0.3f; cfg.max_depth = 6;
        cfg.early_stop_rounds = 20;
        cfg.seed = 42;
        XGBDataset* ds = xgb_dataset_create(X, yf, ntr, D, &cfg, a);
        XGBDataset* dsv = xgb_dataset_create(&X[ntr * D], &yf[ntr], nval, D, &cfg, a);

        uint64_t t0 = rcre_prof_now_ns();
        XGBModel m;
        xgb_train(ds, dsv, &cfg, &m, a);
        uint64_t ms = (rcre_prof_now_ns() - t0) / 1000000;

        float* pv = (float*)malloc(nval * sizeof(float));
        xgb_predict(&m, &X[ntr * D], nval, D, pv);
        float aucv = xgb_auc(&yf[ntr], pv, nval);
        float llv = xgb_logloss(&yf[ntr], pv, nval);
        int ni, nl;
        xgb_model_counts(&m, &ni, &nl);
        printf("    trees=%d (best iter %d)  internal=%d leaves=%d\n",
               m.num_trees, m.best_iteration, ni, nl);
        printf("    train time %.0f ms  val AUC=%.4f  val logloss=%.4f\n",
               (double)ms, (double)aucv, (double)llv);

        /* ---------------- 4. RCRE predictor parity ---------------- */
        printf("\n[4] RCRE flat-inference parity (same binary model)\n");
        RCRE_TreeNode* rn =
            (RCRE_TreeNode*)rcre_arena_alloc(a, m.num_nodes * sizeof(RCRE_TreeNode), 16);
        size_t* rr = (size_t*)rcre_arena_alloc(a, m.num_trees * sizeof(size_t), 4);
        RCRE_GBDTModel rm;
        xgb_to_rcre_gbdt(&m, &rm, rn, rr);

        size_t shp[2] = {n, D};
        RCRE_Tensor Xt = rcre_tensor_from_buffer(X, shp, 2, RCRE_DTYPE_F32, 0);
        RCRE_Tensor Pt = rcre_tensor_create(shp, 2, RCRE_DTYPE_F32, 0, a);
        float* pmine = (float*)malloc(n * sizeof(float));
        xgb_predict(&m, X, n, D, pmine);

        uint64_t t0a = rcre_prof_now_ns();
        if (rcre_ml_gbdt_predict(&rm, &Xt, &Pt) != RCRE_OK) {
            printf("    rcre_ml_gbdt_predict failed\n");
        }
        uint64_t ms_a = (rcre_prof_now_ns() - t0a) / 1000;
        uint64_t t0b = rcre_prof_now_ns();
        xgb_predict(&m, X, n, D, pmine);
        uint64_t ms_b = (rcre_prof_now_ns() - t0b) / 1000;

        float md = 0;
        for (size_t i = 0; i < n; i++)
            md = fmaxf(md, fabsf(pmine[i] - ((float*)Pt.data)[i]));
        printf("    max |xgb - rcre| = %.3g\n", (double)md);
        printf("    rcre predictor: %.1f us/row   xgb predictor: %.1f us/row\n",
               (double)ms_a / n, (double)ms_b / n);
        printf("    throughput: rcre %.1f Mrows/s   xgb %.1f Mrows/s\n",
               1000.0 * n / (double)ms_a, 1000.0 * n / (double)ms_b);

        free(pv); free(pmine); free(yf); free(srt);
        rcre_arena_destroy(a);
        free(X); free(y);
    }

    printf("\n================================================================\n");
    printf("   done\n");
    return 0;
}

