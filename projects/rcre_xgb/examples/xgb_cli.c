/**
 * @file xgb_cli.c
 * @brief RCRE-XGB command-line interface.
 *
 *   xgb_cli demo                              self-contained end-to-end demo
 *   xgb_cli train  <data.csv> [opts]          train, print metrics, save model
 *   xgb_cli predict <model.bin> <data.csv>    print per-row scores
 *   xgb_cli eval     <model.bin> <data.csv>   metrics vs label column
 *   xgb_cli export   <model.bin>              human-readable tree dump
 *
 * train options:
 *   --task regression|binary|multi  --classes K
 *   --label COL (0-based)  --val <file.csv>
 *   --rounds N  --eta F  --depth D  --min-child F  --gamma F
 *   --lambda F  --alpha F  --subsample F  --colsample F  --bins N
 *   --early-stop N  --seed N  --out model.bin
 */
#include "rcre.h"
#include "xgb.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* arg helpers                                                         */
/* ------------------------------------------------------------------ */

static const char* opt_val(int argc, char** argv, int* i, const char* name) {
    if (strcmp(argv[*i], name) == 0) {
        if (*i + 1 < argc) return argv[++*i];
        fprintf(stderr, "missing value for %s\n", name);
        exit(2);
    }
    return NULL;
}


/* ------------------------------------------------------------------ */
/* synthetic data generator (demo + offline experiments)               */
/* ------------------------------------------------------------------ */

static uint64_t s_rng = 2026;
static double s_frand(void) {
    s_rng = s_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(s_rng >> 11) / (double)0x10000000000000ULL;
}
static double s_frandn(void) {
    return (s_frand() + s_frand() + s_frand()) / 3.0 * 2.0 - 1.0;
}

/* n x d data; response uses a handful of active features + interaction */
static void gen_data(size_t n, size_t d, float* X, double* yreg) {
    for (size_t i = 0; i < n; i++) {
        for (size_t c = 0; c < d; c++) X[i * d + c] = (float)s_frandn();
        double t = tanh(1.5 * X[i * d + 0] + 0.8 * X[i * d + 1]
                       - 0.5 * X[i * d + 2] + 0.3 * sin(X[i * d + 3]));
        t += 0.4 * X[i * d + 4] * X[i * d + 5];
        t += 0.15 * X[i * d + 6] * X[i * d + 7];
        t += 0.25 * s_frandn();          /* irreducible noise */
        yreg[i] = t;
    }
}

static int cmpd(const void* p, const void* q) {
    double x = *(const double*)p, y = *(const double*)q;
    return (x > y) - (x < y);
}

/* ------------------------------------------------------------------ */
/* demo                                                                */
/* ------------------------------------------------------------------ */

static int cmd_demo(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("================================================================\n");
    printf("   RCRE-XGB DEMO — synthetic data -> train -> evaluate\n");
    printf("================================================================\n\n");

    size_t n = 20000, d = 16;
    size_t ntr = (size_t)(n * 0.8), nval = n - ntr;

    float* X = (float*)malloc(n * d * sizeof(float));
    double* yreg = (double*)malloc(n * sizeof(double));
    gen_data(n, d, X, yreg);

    /* deterministic contiguous train/val split */
    float* Xtr = (float*)malloc(ntr * d * sizeof(float));
    float* Xv = (float*)malloc(nval * d * sizeof(float));
    float* ytr = (float*)malloc(ntr * sizeof(float));
    float* yv = (float*)malloc(nval * sizeof(float));
    for (size_t i = 0; i < ntr; i++) {
        memcpy(&Xtr[i * d], &X[i * d], d * sizeof(float));
        ytr[i] = (float)yreg[i];
    }
    for (size_t i = 0; i < nval; i++) {
        size_t s = ntr + i;
        memcpy(&Xv[i * d], &X[s * d], d * sizeof(float));
        yv[i] = (float)yreg[s];
    }

    RCRE_Arena* a = rcre_arena_create(256 << 20);

    /* ---------- regression ---------- */
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_REGRESSION;
    cfg.rounds = 150; cfg.eta = 0.3f; cfg.max_depth = 6;
    cfg.early_stop_rounds = 15; cfg.seed = 42;

    XGBDataset* ds = xgb_dataset_create(Xtr, ytr, ntr, d, &cfg, a);
    XGBDataset* dsv = xgb_dataset_create(Xv, yv, nval, d, &cfg, a);

    RCRE_ProfContext* prof = rcre_prof_init(1 << 14);
    RCRE_PROFILE_SCOPE(prof, "train(regression)", zt);
    XGBModel mreg;
    xgb_train(ds, dsv, &cfg, &mreg, a);
    uint64_t train_ms = rcre_prof_scope_end(&zt) / 1000000;

    float* ptr = (float*)malloc(ntr * sizeof(float));
    float* pv = (float*)malloc(nval * sizeof(float));
    xgb_predict(&mreg, Xtr, ntr, d, ptr);
    xgb_predict(&mreg, Xv, nval, d, pv);
    float rmsetr = xgb_rmse(ytr, ptr, ntr);
    float rmsev = xgb_rmse(yv, pv, nval);
    rcre_prof_sample_arena(prof, a);

    int ni, nl;
    xgb_model_counts(&mreg, &ni, &nl);
    printf("[regression]  n=%zu d=%zu\n", ntr, d);
    printf("  trees=%d (best iter %d of %d attempted)  internal=%d leaves=%d\n",
           mreg.num_trees, mreg.best_iteration, cfg.rounds, ni, nl);
    printf("  train RMSE = %.4f   val RMSE = %.4f\n", (double)rmsetr, (double)rmsev);
    if (mreg.val_curve_len > 0) {
        printf("  val curve (first/last): %.4f -> %.4f\n",
               (double)mreg.val_curve[0], (double)mreg.val_curve[mreg.val_curve_len - 1]);
    }
    printf("  train time = %.0f ms  (%.1f ms/round)\n", (double)train_ms,
           (double)train_ms / (mreg.best_iteration + 1));

    /* ---------- binary ---------- */
    /* threshold the regression target at its median */
    double* sorted = (double*)malloc(n * sizeof(double));
    memcpy(sorted, yreg, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmpd);
    double med = sorted[n / 2];
    for (size_t i = 0; i < ntr; i++) ytr[i] = (ytr[i] > med) ? 1.0f : 0.0f;
    for (size_t i = 0; i < nval; i++) yv[i] = (yv[i] > med) ? 1.0f : 0.0f;

    XGBConfig cb; xgb_config_default(&cb);
    cb.task = XGB_TASK_BINARY;
    cb.rounds = 200; cb.eta = 0.3f; cb.max_depth = 6;
    cb.early_stop_rounds = 15; cb.seed = 42;

    XGBDataset* bds = xgb_dataset_create(Xtr, ytr, ntr, d, &cb, a);
    XGBDataset* bdsv = xgb_dataset_create(Xv, yv, nval, d, &cb, a);

    RCRE_PROFILE_SCOPE(prof, "train(binary)", zb);
    XGBModel mb;
    xgb_train(bds, bdsv, &cb, &mb, a);
    uint64_t trainb_ms = rcre_prof_scope_end(&zb) / 1000000;

    xgb_predict(&mb, Xtr, ntr, d, ptr);
    xgb_predict(&mb, Xv, nval, d, pv);
    float lltr = xgb_logloss(ytr, ptr, ntr);
    float llv = xgb_logloss(yv, pv, nval);
    float aucv = xgb_auc(yv, pv, nval);
    int nbi, nbl;
    xgb_model_counts(&mb, &nbi, &nbl);
    printf("\n[binary]      (y = 1[target > median])\n");
    printf("  trees=%d (best iter %d of %d attempted)  internal=%d leaves=%d\n",
           mb.num_trees, mb.best_iteration, cb.rounds, nbi, nbl);
    printf("  train logloss = %.4f   val logloss = %.4f   val AUC = %.4f\n",
           (double)lltr, (double)llv, (double)aucv);
    printf("  train time = %.0f ms  (%.1f ms/round)\n", (double)trainb_ms,
           (double)trainb_ms / (mb.best_iteration + 1));

    /* ---------- sample predictions ---------- */
    printf("\n[sample rows]  (target | pred_reg | pred_bin)\n");
    for (size_t i = 0; i < 5; i++) {
        float pr, pb;
        xgb_predict(&mreg, &Xtr[i * d], 1, d, &pr);
        xgb_predict(&mb, &Xtr[i * d], 1, d, &pb);
        printf("  %.4f   |  %8.4f  |  %6.3f\n",
               (double)yreg[i * 1], (double)pr, (double)pb);
    }

    /* ---------- model persistence ---------- */
    const char* mp = "/tmp/rcre_xgb_demo_binary.bin";
    xgb_model_save(&mb, mp);
    long mbytes = 0;
    {
        FILE* mf = fopen(mp, "rb");
        if (mf) { fseek(mf, 0, SEEK_END); mbytes = ftell(mf); fclose(mf); }
    }
    XGBModel ml;
    int ok = xgb_model_load(mp, &ml, a) == 0;
    if (ok) {
        float* pr2 = (float*)malloc(ntr * sizeof(float));
        xgb_predict(&ml, Xtr, ntr, d, pr2);
        float md = 0;
        for (size_t i = 0; i < ntr; i++) md = fmaxf(md, fabsf(ptr[i] - pr2[i]));
        ok = (md < 1e-6f);
        free(pr2);
    }
    printf("\n[persistence]  saved %s (%ld bytes)  roundtrip %s\n", mp, mbytes,
           ok ? "identical (ok)" : "MISMATCH (BAD)");

    rcre_prof_print_summary(prof, stdout);
    rcre_prof_free(prof);
    rcre_arena_destroy(a);
    free(X); free(yreg); free(Xtr); free(Xv); free(ytr); free(yv);
    free(ptr); free(pv); free(sorted);
    printf("\n");
    return 0;
}


/* ------------------------------------------------------------------ */
/* train                                                               */
/* ------------------------------------------------------------------ */

static int cmd_train(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: xgb_cli train <data.csv> [opts]\n");
        return 2;
    }
    const char* path = argv[1];
    XGBConfig cfg; xgb_config_default(&cfg);
    int label_col = -1;
    int i = 2;
    const char* valpath = NULL, *outpath = "/tmp/rcre_xgb_model.bin";
    while (i < argc) {
        const char* a = argv[i];
        if (!strcmp(a, "--task")) {
            i++;
            if (!strcmp(argv[i], "regression")) cfg.task = XGB_TASK_REGRESSION;
            else if (!strcmp(argv[i], "binary")) cfg.task = XGB_TASK_BINARY;
            else if (!strcmp(argv[i], "multi")) cfg.task = XGB_TASK_MULTI;
            else { fprintf(stderr, "bad --task %s\n", argv[i]); return 2; }
        } else if (!strcmp(a, "--label"))      { i++; label_col = atoi(argv[i]); }
        else if (!strcmp(a, "--classes"))      { i++; cfg.num_classes = atoi(argv[i]); }
        else if (!strcmp(a, "--val"))          { i++; valpath = argv[i]; }
        else if (!strcmp(a, "--out"))          { i++; outpath = argv[i]; }
        else if (!strcmp(a, "--seed"))         { i++; cfg.seed = (uint64_t)strtoull(argv[i], NULL, 10); }
        else if (!strcmp(a, "--rounds"))       { i++; cfg.rounds = atoi(argv[i]); }
        else if (!strcmp(a, "--eta"))          { i++; cfg.eta = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--depth"))        { i++; cfg.max_depth = atoi(argv[i]); }
        else if (!strcmp(a, "--min-child"))    { i++; cfg.min_child_weight = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--gamma"))        { i++; cfg.gamma = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--lambda"))       { i++; cfg.lambda = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--alpha"))        { i++; cfg.alpha = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--subsample"))    { i++; cfg.subsample = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--colsample"))    { i++; cfg.colsample = strtof(argv[i], NULL); }
        else if (!strcmp(a, "--bins"))         { i++; cfg.max_bins = atoi(argv[i]); }
        else if (!strcmp(a, "--early-stop"))   { i++; cfg.early_stop_rounds = atoi(argv[i]); }
        else { fprintf(stderr, "unknown option: %s\n", a); return 2; }
        i++;
    }

    RCRE_Arena* a = rcre_arena_create(128 << 20);
    float* X, *y;
    size_t n, d;
    if (xgb_csv_load(path, &X, &y, &n, &d, label_col, a) != 0) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }
    printf("loaded %s: n=%zu d=%zu (label col %d)\n", path, n, d, label_col);
    XGBDataset* ds = xgb_dataset_create(X, y, n, d, &cfg, a);
    XGBDataset* dsv = NULL;
    if (valpath) {
        float* Xv, *yv;
        size_t nv, dv;
        if (xgb_csv_load(valpath, &Xv, &yv, &nv, &dv, label_col, a) != 0) {
            fprintf(stderr, "failed to load %s\n", valpath);
            return 1;
        }
        dsv = xgb_dataset_create(Xv, yv, nv, dv, &cfg, a);
    }

    XGBModel m;
    uint64_t t0 = rcre_prof_now_ns();
    int rc = xgb_train(ds, dsv, &cfg, &m, a);
    uint64_t ms = (rcre_prof_now_ns() - t0) / 1000000;
    if (rc != 0) { fprintf(stderr, "training failed\n"); return 1; }

    float* p = (float*)malloc(n * m.num_classes * sizeof(float));
    xgb_predict(&m, X, n, d, p);
    int ni, nl;
    xgb_model_counts(&m, &ni, &nl);
    printf("trained: trees=%d nodes=%zu (internal=%d leaves=%d) eta=%.3f in %lu ms\n",
           m.num_trees, m.num_nodes, ni, nl, (double)m.eta, (unsigned long)ms);
    if (m.num_classes == 1) {
        if (cfg.task == XGB_TASK_BINARY)
            printf("train: logloss=%.4f auc=%.4f\n",
                   (double)xgb_logloss(y, p, n), (double)xgb_auc(y, p, n));
        else
            printf("train: rmse=%.4f mae=%.4f\n",
                   (double)xgb_rmse(y, p, n), (double)xgb_mae(y, p, n));
    } else {
        printf("train: accuracy=%.4f\n", (double)xgb_accuracy(y, p, n, m.num_classes));
    }
    if (dsv)
        printf("best_iteration=%d (val curve len %d)\n", m.best_iteration, m.val_curve_len);

    if (xgb_model_save(&m, outpath) != 0) {
        fprintf(stderr, "save failed\n");
        return 1;
    }
    printf("saved model -> %s\n", outpath);
    rcre_arena_destroy(a);
    free(p);
    return 0;
}

/* ------------------------------------------------------------------ */
/* predict / eval / export                                             */
/* ------------------------------------------------------------------ */

static int load_model(const char* path, RCRE_Arena* a, XGBModel* m) {
    if (xgb_model_load(path, m, a) != 0) {
        fprintf(stderr, "failed to load model %s\n", path);
        return -1;
    }
    return 0;
}

static int cmd_predict(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: xgb_cli predict <model.bin> <data.csv>\n"); return 2; }
    RCRE_Arena* a = rcre_arena_create(128 << 20);
    XGBModel m;
    if (load_model(argv[1], a, &m) != 0) return 1;
    float* X, *y;
    size_t n, d;
    int label_col = -1;
    int i = 3;
    while (i < argc) {
        const char* v = opt_val(argc, argv, &i, "--label");
        if (v) label_col = atoi(v);
        else i++;
    }
    if (xgb_csv_load(argv[2], &X, &y, &n, &d, label_col, a) != 0) {
        fprintf(stderr, "failed to load %s\n", argv[2]);
        return 1;
    }
    float* p = (float*)malloc(n * m.num_classes * sizeof(float));
    xgb_predict(&m, X, n, d, p);
    for (size_t r = 0; r < n; r++) {
        if (m.num_classes == 1) printf("%.6f\n", (double)p[r]);
        else {
            for (int k = 0; k < m.num_classes; k++)
                printf("%s%.6f", k ? " " : "", (double)p[r * m.num_classes + k]);
            printf("\n");
        }
    }
    free(p);
    rcre_arena_destroy(a);
    return 0;
}

static int cmd_eval(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: xgb_cli eval <model.bin> <data.csv> [--label COL]\n"); return 2; }
    RCRE_Arena* a = rcre_arena_create(128 << 20);
    XGBModel m;
    if (load_model(argv[1], a, &m) != 0) return 1;
    int label_col = -1;
    int i = 3;
    while (i < argc) {
        const char* v = opt_val(argc, argv, &i, "--label");
        if (v) label_col = atoi(v);
        else i++;
    }
    if (label_col < 0) { fprintf(stderr, "eval requires --label COL\n"); return 2; }
    float* X, *y;
    size_t n, d;
    if (xgb_csv_load(argv[2], &X, &y, &n, &d, label_col, a) != 0) {
        fprintf(stderr, "failed to load %s\n", argv[2]);
        return 1;
    }
    float* p = (float*)malloc(n * m.num_classes * sizeof(float));
    xgb_predict(&m, X, n, d, p);
    if (m.num_classes == 1) {
        if (m.task == XGB_TASK_BINARY) {
            double acc = 0.0;
            for (size_t i = 0; i < n; i++)
                if ((p[i] > 0.5f ? 1 : 0) == (int)y[i]) acc++;
            printf("logloss=%.4f  auc=%.4f  accuracy=%.4f\n",
                   (double)xgb_logloss(y, p, n), (double)xgb_auc(y, p, n),
                   acc / (double)n);
        } else
            printf("rmse=%.4f  mae=%.4f\n",
                   (double)xgb_rmse(y, p, n), (double)xgb_mae(y, p, n));
    } else {
        printf("accuracy=%.4f\n", (double)xgb_accuracy(y, p, n, m.num_classes));
    }
    free(p);
    rcre_arena_destroy(a);
    return 0;
}

static int cmd_export(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: xgb_cli export <model.bin>\n"); return 2; }
    RCRE_Arena* a = rcre_arena_create(128 << 20);
    XGBModel m;
    if (load_model(argv[1], a, &m) != 0) return 1;
    xgb_model_print(&m, NULL);
    rcre_arena_destroy(a);
    return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    rcre_hal_init();
    if (argc < 2) {
        printf("RCRE-XGB — XGBoost-style gradient boosted trees on RCRE\n\n");
        printf("usage:\n");
        printf("  xgb_cli demo\n");
        printf("  xgb_cli train <data.csv> [--task binary|regression|multi] [--label COL]\n");
        printf("      [--rounds N] [--eta F] [--depth D] [--min-child F] [--gamma F]\n");
        printf("      [--lambda F] [--alpha F] [--subsample F] [--colsample F] [--bins N]\n");
        printf("      [--val <file.csv>] [--early-stop N] [--seed N] [--out model.bin]\n");
        printf("  xgb_cli predict <model.bin> <data.csv> [--label COL]\n");
        printf("  xgb_cli eval <model.bin> <data.csv> --label COL\n");
        printf("  xgb_cli export <model.bin>\n");
        return 2;
    }
    if (!strcmp(argv[1], "demo")) return cmd_demo(argc - 1, argv + 1);
    if (!strcmp(argv[1], "train")) return cmd_train(argc - 1, argv + 1);
    if (!strcmp(argv[1], "predict")) return cmd_predict(argc - 1, argv + 1);
    if (!strcmp(argv[1], "eval")) return cmd_eval(argc - 1, argv + 1);
    if (!strcmp(argv[1], "export")) return cmd_export(argc - 1, argv + 1);
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 2;
}
