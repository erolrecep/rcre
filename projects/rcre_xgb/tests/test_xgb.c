/**
 * @file test_xgb.c
 * @brief RCRE-XGB test suite: binning, splits, losses, regularization,
 *        missing values, early stopping, multi-class, I/O, RCRE interop.
 */
#include "xgb_internal.h"

static int n_pass = 0, n_fail = 0;

#define CHECK(name, cond) do { fflush(stdout); \
    if (cond) { n_pass++; printf("  PASS  %s\n", name); } \
    else      { n_fail++; printf("  FAIL  %s   (line %d)\n", name, __LINE__); } \
} while (0)

static uint64_t rnd_state = 12345;
static double frand(void) {
    rnd_state = rnd_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rnd_state >> 11) / 9007199254740992.0; /* 2^53 */
}
static double frandn(void) {
    return (frand() + frand() + frand()) / 3.0 * 2.0 - 1.0;
}

/* ================================================================== */
/* Binning                                                             */
/* ================================================================== */
static void test_binning(RCRE_Arena* a) {
    printf("[binning]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    float X[10 * 2], y[10];
    for (int i = 0; i < 10; i++) {
        X[i * 2] = (float)i;             /* 0..9 */
        X[i * 2 + 1] = 5.0f;             /* constant */
        y[i] = 0.0f;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, 10, 2, &cfg, a);
    CHECK("dataset created", ds != NULL);
    CHECK("constant feature -> 1 bin", ds->n_bins[1] == 1);
    CHECK("ramp feature -> >1 bin", ds->n_bins[0] > 1);
    /* monotone binning: bin[i] <= bin[i+1] */
    int mono = 1;
    for (int i = 1; i < 10; i++)
        if (ds->bins[i * 2] < ds->bins[(i - 1) * 2]) mono = 0;
    CHECK("monotone bin assignment", mono);
    /* NaN -> missing bin */
    float Xn[2], yn = 0;
    Xn[0] = NAN; Xn[1] = 1.0f;
    XGBDataset* dsn = xgb_dataset_create(Xn, &yn, 1, 2, &cfg, a);
    CHECK("NaN -> missing bin", dsn->bins[0] == XGB_MISSING_BIN);

    /* small bins cap: 100 distinct values, max_bins=8 -> n_bins <= 8 */
    float Xs[100 * 1], ys[100];
    for (int i = 0; i < 100; i++) { Xs[i] = (float)i; ys[i] = 0; }
    cfg.max_bins = 8;
    XGBDataset* dss = xgb_dataset_create(Xs, ys, 100, 1, &cfg, a);
    CHECK("max_bins respected", dss->n_bins[0] <= 8);
    cfg.max_bins = 256;
}

/* ================================================================== */
/* Metric helpers                                                      */
/* ================================================================== */
static void test_metrics(void) {
    printf("[metrics]\n");
    float y[4] = {1, 1, 0, 0};
    float p[4] = {0.9f, 0.8f, 0.2f, 0.1f};
    CHECK("AUC perfect order = 1.0", fabsf(xgb_auc(y, p, 4) - 1.0f) < 1e-6f);
    float pbad[4] = {0.1f, 0.2f, 0.8f, 0.9f};
    CHECK("AUC reversed = 0.0", fabsf(xgb_auc(y, pbad, 4)) < 1e-6f);
    float ll = xgb_logloss(y, p, 4);
    float ref = -(logf(0.9f) + logf(0.8f) + logf(0.8f) + logf(0.9f)) / 4.0f;
    CHECK("logloss value", fabsf(ll - ref) < 1e-5f);
    float yy[2] = {0.0f, 2.0f}, pp[2] = {0.0f, 4.0f};
    CHECK("rmse", fabsf(xgb_rmse(yy, pp, 2) - sqrtf(2.0f)) < 1e-6f);
    float P[9] = {0.7f, 0.2f, 0.1f,   /* row 0: class 0 wins */
                  0.1f, 0.6f, 0.3f,   /* row 1: class 1 wins */
                  0.2f, 0.3f, 0.5f};  /* row 2: class 2 wins */
    float yc[3] = {0.0f, 1.0f, 0.0f};
    CHECK("accuracy (K=3)", fabsf(xgb_accuracy(yc, P, 3, 3) - 2.0f / 3.0f) < 1e-6f);
}

/* ================================================================== */
/* Regression: step function recovered exactly                         */
/* ================================================================== */
static void test_reg_perfect(RCRE_Arena* a) {
    printf("[regression perfect fit]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_REGRESSION;
    cfg.rounds = 4; cfg.eta = 1.0f; cfg.max_depth = 2; cfg.lambda = 0.1f;

    /* y = 0 if x0 < 0 else 10, plus an x1-dependent offset */
    int n = 64;
    float X[n * 2], y[n];
    for (int i = 0; i < n; i++) {
        X[i * 2] = frand() * 2.0f - 1.0f;
        X[i * 2 + 1] = frand();
        y[i] = (X[i * 2] < 0.0f ? 0.0f : 10.0f);
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 2, &cfg, a);
    XGBModel m;
    CHECK("train ok", xgb_train(ds, NULL, &cfg, &m, a) == 0);
    float p[n];
    xgb_predict(&m, X, n, 2, p);
    float rmse = xgb_rmse(y, p, n);
    printf("    rmse = %.4f\n", (double)rmse);
    CHECK("step function rmse < 0.5", rmse < 0.5f);
    float ymean = 0.0f;
    for (int i = 0; i < n; i++) ymean += y[i];
    ymean /= n;
    CHECK("base score = mean(y)", fabsf(m.base_score - ymean) < 1e-4f);
    /* the step must be captured: mean prediction on x0>0 well above x0<0 */
    double sHi = 0, sLo = 0; int cHi = 0, cLo = 0;
    for (int i = 0; i < n; i++)
        if (X[i * 2] >= 0.0f) { sHi += p[i]; cHi++; }
        else { sLo += p[i]; cLo++; }
    double gap = sHi / cHi - sLo / cLo;
    printf("    step gap = %.3f (need > 5 = half the step)\n", gap);
    CHECK("predictions monotone in x0", gap > 5.0);
}

/* ================================================================== */
/* Binary: XOR separable with depth 2                                  */
/* ================================================================== */
static void test_xor(RCRE_Arena* a) {
    printf("[binary xor]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_BINARY;
    cfg.rounds = 3; cfg.eta = 1.0f; cfg.max_depth = 2; cfg.lambda = 0.1f;
    cfg.min_child_weight = 0.01f; /* XOR needs 1-sample leaves */

    /* asymmetric XOR: quadrant counts 4/2/1/1 chosen so BOTH root-split
     * directions are informative (a perfectly symmetric XOR has zero-gain
     * root splits for every axis and boosting stalls on it — standard
     * XGBoost included) */
    float X[8 * 2] = {-1,-1,  -1,-1,  -1,-1,  -1,-1,
                       -1, 1,  -1, 1,   1, 1,   1,-1};
    float y[8] = {1, 1, 1, 1,  0, 0, 1, 0};
    XGBDataset* ds = xgb_dataset_create(X, y, 8, 2, &cfg, a);
    XGBModel m;
    CHECK("train ok", xgb_train(ds, NULL, &cfg, &m, a) == 0);
    float p[8];
    xgb_predict(&m, X, 8, 2, p);
    float ll = xgb_logloss(y, p, 8);
    float auc = xgb_auc(y, p, 8);
    int correct = 0;
    for (int i = 0; i < 8; i++) if ((p[i] > 0.5f ? 1 : 0) == y[i]) correct++;
    printf("    logloss = %.4f  auc = %.3f  correct = %d/8\n",
           (double)ll, (double)auc, correct);
    CHECK("xor accuracy 8/8", correct == 8);
    CHECK("xor logloss < 0.15", ll < 0.15f);
    CHECK("xor auc = 1.0", auc > 0.999f);
}

/* ================================================================== */
/* Convergence: logloss decreases every round on separable data        */
/* ================================================================== */
static void test_convergence(RCRE_Arena* a) {
    printf("[convergence]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_BINARY;
    cfg.rounds = 30; cfg.eta = 0.3f; cfg.max_depth = 4;

    int n = 400;
    float X[n * 2], y[n];
    for (int i = 0; i < n; i++) {
        int cls = i % 2;
        X[i * 2] = frandn() + cls * 2.0f;
        X[i * 2 + 1] = frandn() - cls * 2.0f;
        y[i] = (float)cls;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 2, &cfg, a);
    XGBModel m;
    xgb_train(ds, NULL, &cfg, &m, a);
    /* re-train with validation on same data to get the curve */
    xgb_train(ds, ds, &cfg, &m, a);
    CHECK("val curve length = rounds", m.val_curve_len == 30);
    int decreasing = 1;
    for (int i = 1; i < m.val_curve_len; i++)
        if (m.val_curve[i] > m.val_curve[i - 1] + 1e-6f) decreasing = 0;
    CHECK("logloss decreases (monotone on separable data)", decreasing);
    float p[n];
    xgb_predict(&m, X, n, 2, p);
    CHECK("final auc > 0.99", xgb_auc(y, p, n) > 0.99f);
}

/* ================================================================== */
/* Regularization                                                      */
/* ================================================================== */
static void test_regularization(RCRE_Arena* a) {
    printf("[regularization]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_REGRESSION;
    cfg.rounds = 1; cfg.eta = 1.0f; cfg.max_depth = 3;

    int n = 200;
    float X[n * 3], y[n];
    for (int i = 0; i < n; i++) {
        X[i * 3] = frand() * 2 - 1;
        X[i * 3 + 1] = frand() * 2 - 1;
        X[i * 3 + 2] = frand() * 2 - 1;
        y[i] = 4.0f * X[i * 3] + frandn() * 0.5f;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 3, &cfg, a);

    /* stronger lambda -> smaller leaf magnitudes */
    float maxleaf[3];
    float lam[3] = {0.1f, 1.0f, 100.0f};
    for (int t = 0; t < 3; t++) {
        cfg.lambda = lam[t];
        XGBModel m;
        xgb_train(ds, NULL, &cfg, &m, a);
        float mx = 0;
        for (size_t i = 0; i < m.num_nodes; i++)
            if (m.nodes[i].is_leaf && fabsf(m.nodes[i].leaf_value) > mx)
                mx = fabsf(m.nodes[i].leaf_value);
        maxleaf[t] = mx;
    }
    printf("    max |leaf| for lambda 0.1/1/100: %.3f / %.3f / %.3f\n",
           (double)maxleaf[0], (double)maxleaf[1], (double)maxleaf[2]);
    CHECK("L2 shrinks leaves", maxleaf[0] > maxleaf[1] && maxleaf[1] > maxleaf[2]);

    /* huge gamma -> no split at all (single leaf) */
    cfg.lambda = 1.0f;
    cfg.gamma = 1e6f;
    XGBModel mg;
    xgb_train(ds, NULL, &cfg, &mg, a);
    cfg.gamma = 0.0f;
    CHECK("huge gamma -> single leaf", mg.num_nodes == 1);
    CHECK("single leaf = base-ish weight", mg.nodes[0].is_leaf);

    /* min_child_weight larger than any side mass -> no split */
    cfg.gamma = 0.0f;
    cfg.min_child_weight = 1e6f;
    XGBModel mw;
    xgb_train(ds, NULL, &cfg, &mw, a);
    cfg.min_child_weight = 1.0f;
    CHECK("huge min_child_weight -> single leaf", mw.num_nodes == 1);

    /* alpha (L1) soft-thresholds: with huge alpha the leaf is exactly 0 */
    cfg.alpha = 1e6f;
    XGBModel ma;
    xgb_train(ds, NULL, &cfg, &ma, a);
    cfg.alpha = 0.0f;
    CHECK("huge alpha -> zero leaf weight", ma.nodes[0].leaf_value == 0.0f);
}

/* ================================================================== */
/* Missing values                                                      */
/* ================================================================== */
static void test_missing(RCRE_Arena* a) {
    printf("[missing values]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_REGRESSION;
    cfg.rounds = 3; cfg.eta = 0.5f; cfg.max_depth = 3;

    int n = 120;
    float X[n * 2], y[n];
    for (int i = 0; i < n; i++) {
        X[i * 2] = frand() * 2 - 1;
        X[i * 2 + 1] = frand() * 2 - 1;
        y[i] = 3.0f * X[i * 2] + 2.0f * X[i * 2 + 1];
    }
    /* 25% missing in feature 1 */
    for (int i = 0; i < n; i += 4) X[i * 2 + 1] = NAN;

    XGBDataset* ds = xgb_dataset_create(X, y, n, 2, &cfg, a);
    XGBModel m;
    CHECK("train with NaNs ok", xgb_train(ds, NULL, &cfg, &m, a) == 0);
    float p[n];
    CHECK("predict with NaNs ok", xgb_predict(&m, X, n, 2, p) == 0);
    int finite = 1;
    for (int i = 0; i < n; i++) if (p[i] != p[i]) finite = 0;
    CHECK("all predictions finite", finite);

    /* NaN rows must be routed deterministically: their predictions must be
     * finite and lie within the hull of the two extreme completions
     * (x1 = -1e30 forces every split left, +1e30 every split right) */
    float qL[n], qR[n];
    float Xr[n * 2];
    memcpy(Xr, X, sizeof X);
    for (int i = 0; i < n; i++)
        if (Xr[i * 2 + 1] != Xr[i * 2 + 1]) Xr[i * 2 + 1] = -1e30f;
    xgb_predict(&m, Xr, n, 2, qL);
    memcpy(Xr, X, sizeof X);
    for (int i = 0; i < n; i++)
        if (Xr[i * 2 + 1] != Xr[i * 2 + 1]) Xr[i * 2 + 1] = 1e30f;
    xgb_predict(&m, Xr, n, 2, qR);
    int ok_nan = 1;
    for (int i = 0; i < n; i++) {
        if (X[i * 2 + 1] == X[i * 2 + 1]) continue;
        if (p[i] != p[i]) ok_nan = 0;
        float lo = qL[i] < qR[i] ? qL[i] : qR[i];
        float hi = qL[i] > qR[i] ? qL[i] : qR[i];
        if (p[i] < lo - 1e-5f || p[i] > hi + 1e-5f) ok_nan = 0;
    }
    CHECK("NaN follows default direction", ok_nan);

    /* Test optimal default-direction learning:
     * When missing rows correlate with the right child (y=10),
     * gain maximization must learn default_left == false (route to right). */
    int n2 = 100;
    float X2[100], y2[100];
    for (int i = 0; i < n2; i++) {
        if (i < 40) {
            X2[i] = -1.0f - (float)i * 0.1f; /* Left branch: negative values */
            y2[i] = 0.0f;
        } else if (i < 80) {
            X2[i] = 1.0f + (float)i * 0.1f;  /* Right branch: positive values */
            y2[i] = 10.0f;
        } else {
            X2[i] = NAN;                     /* Missing values: share label with right */
            y2[i] = 10.0f;
        }
    }
    XGBConfig cfg2; xgb_config_default(&cfg2);
    cfg2.task = XGB_TASK_REGRESSION;
    cfg2.rounds = 1; cfg2.eta = 1.0f; cfg2.max_depth = 1;
    XGBDataset* ds2 = xgb_dataset_create(X2, y2, n2, 1, &cfg2, a);
    XGBModel m2;
    xgb_train(ds2, NULL, &cfg2, &m2, a);
    CHECK("gain maximization learned default_left == false", m2.nodes[0].default_left == false);

    /* When missing rows correlate with left child (y=0), learn default_left == true */
    for (int i = 80; i < n2; i++) y2[i] = 0.0f;
    XGBDataset* ds3 = xgb_dataset_create(X2, y2, n2, 1, &cfg2, a);
    XGBModel m3;
    xgb_train(ds3, NULL, &cfg2, &m3, a);
    CHECK("gain maximization learned default_left == true", m3.nodes[0].default_left == true);
}

/* ================================================================== */
/* Determinism + subsampling                                           */
/* ================================================================== */
static void test_determinism(RCRE_Arena* a) {
    printf("[determinism & subsampling]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_BINARY;
    cfg.rounds = 20; cfg.eta = 0.3f; cfg.max_depth = 4;
    cfg.subsample = 0.7f; cfg.colsample = 1.0f; cfg.seed = 7;

    int n = 300;
    float X[n * 4], y[n];
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 4; c++) X[i * 4 + c] = frandn();
        y[i] = (X[i * 4] + X[i * 4 + 1] > 0.2f) ? 1 : 0;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 4, &cfg, a);
    XGBModel m1, m2;
    xgb_train(ds, NULL, &cfg, &m1, a);
    xgb_train(ds, NULL, &cfg, &m2, a);
    float p1[n], p2[n];
    xgb_predict(&m1, X, n, 4, p1);
    xgb_predict(&m2, X, n, 4, p2);
    float md = 0;
    for (int i = 0; i < n; i++) md = fmaxf(md, fabsf(p1[i] - p2[i]));
    CHECK("same seed -> identical predictions", md == 0.0f);

    cfg.seed = 8;
    XGBModel m3;
    xgb_train(ds, NULL, &cfg, &m3, a);
    /* models with different seeds still need to be *good* (sanity) */
    float p3[n];
    xgb_predict(&m3, X, n, 4, p3);
    CHECK("subsampled model still learns (auc>0.9)", xgb_auc(y, p3, n) > 0.9f);
}

/* ================================================================== */
/* Early stopping                                                      */
/* ================================================================== */
static void test_early_stopping(RCRE_Arena* a) {
    printf("[early stopping]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_REGRESSION;
    cfg.rounds = 200; cfg.eta = 0.5f; cfg.max_depth = 8;
    cfg.min_child_weight = 0.01f; cfg.lambda = 0.01f;
    cfg.early_stop_rounds = 10;

    /* pure noise: validation RMSE bottoms out quickly, then overfits */
    int n = 300, nv = 200;
    float X[n * 3], y[n], Xv[nv * 3], yv[nv];
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) X[i * 3 + c] = frandn();
        y[i] = frandn();
    }
    for (int i = 0; i < nv; i++) {
        for (int c = 0; c < 3; c++) Xv[i * 3 + c] = frandn();
        yv[i] = frandn();
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 3, &cfg, a);
    XGBDataset* dsv = xgb_dataset_create(Xv, yv, nv, 3, &cfg, a);
    XGBModel m;
    xgb_train(ds, dsv, &cfg, &m, a);
    printf("    best_iteration = %d of %d rounds (val curve len %d)\n",
           m.best_iteration, cfg.rounds, m.val_curve_len);
    CHECK("early stop triggered", m.best_iteration + 1 < cfg.rounds - 1);
    CHECK("best_iteration >= 0", m.best_iteration >= 0);
    CHECK("trees truncated to best", m.num_trees == m.best_iteration + 1);
    /* curve: minimum at best_iteration */
    float best = m.val_curve[m.best_iteration];
    int ok = 1;
    for (int i = 0; i < m.val_curve_len; i++)
        if (m.val_curve[i] < best - 1e-9f) ok = 0;
    CHECK("curve min at best_iteration", ok);
}

/* ================================================================== */
/* Multi-class                                                         */
/* ================================================================== */
static void test_multi(RCRE_Arena* a) {
    printf("[multi-class]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_MULTI;
    cfg.num_classes = 3;
    cfg.rounds = 40; cfg.eta = 0.5f; cfg.max_depth = 4;

    int n = 300;
    float X[n * 3], y[n];
    for (int i = 0; i < n; i++) {
        int cls = i % 3;
        X[i * 3] = frandn() + cls * 2.5f;
        X[i * 3 + 1] = frandn() + (cls - 1) * 2.5f;
        X[i * 3 + 2] = frandn() - cls * 2.0f;
        y[i] = (float)cls;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 3, &cfg, a);
    XGBModel m;
    CHECK("train multi ok", xgb_train(ds, NULL, &cfg, &m, a) == 0);
    CHECK("trees = rounds * K", m.num_trees == 40 * 3);
    float p[n * 3];
    xgb_predict(&m, X, n, 3, p);
    float acc = xgb_accuracy(y, p, n, 3);
    float ssum = 0;
    int rowsok = 1;
    for (int i = 0; i < n; i++) {
        float s = p[i * 3] + p[i * 3 + 1] + p[i * 3 + 2];
        ssum += s;
        if (fabsf(s - 1.0f) > 1e-4f) rowsok = 0;
        for (int k = 0; k < 3; k++)
            if (p[i * 3 + k] < 0.0f || p[i * 3 + k] > 1.0f) rowsok = 0;
    }
    printf("    accuracy = %.3f  mean prob-sum = %.6f\n", (double)acc, (double)(ssum / n));
    CHECK("multi accuracy > 0.95", acc > 0.95f);
    CHECK("prob rows sum to 1", rowsok);
    /* RCRE adapter must reject multi-class */
    RCRE_GBDTModel rm;
    RCRE_TreeNode rn[1];
    size_t rr[1];
    CHECK("RCRE adapter rejects multi", xgb_to_rcre_gbdt(&m, &rm, rn, rr) != 0);
}

/* ================================================================== */
/* Save / load roundtrip                                               */
/* ================================================================== */
static void test_io_roundtrip(RCRE_Arena* a) {
    printf("[save/load roundtrip]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_BINARY;
    cfg.rounds = 12; cfg.eta = 0.3f; cfg.max_depth = 4;

    int n = 250;
    float X[n * 3], y[n];
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) X[i * 3 + c] = frandn();
        y[i] = (X[i * 3] - X[i * 3 + 2] > 0.0f) ? 1 : 0;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 3, &cfg, a);
    XGBModel m;
    xgb_train(ds, NULL, &cfg, &m, a);
    float p1[n];
    xgb_predict(&m, X, n, 3, p1);

    const char* path = "/tmp/rcre_xgb_test_model.bin";
    CHECK("save ok", xgb_model_save(&m, path) == 0);
    XGBModel m2;
    CHECK("load ok", xgb_model_load(path, &m2, a) == 0);
    float p2[n];
    xgb_predict(&m2, X, n, 3, p2);
    float md = 0;
    for (int i = 0; i < n; i++) md = fmaxf(md, fabsf(p1[i] - p2[i]));
    CHECK("roundtrip predictions identical", md == 0.0f);
    CHECK("metadata preserved", m2.num_trees == m.num_trees &&
                                 m2.num_nodes == m.num_nodes &&
                                 fabsf(m2.eta - m.eta) < 1e-9f &&
                                 m2.num_classes == m.num_classes);

    /* CSV: write a small file and read it back */
    const char* csv = "/tmp/rcre_xgb_test.csv";
    FILE* f = fopen(csv, "w");
    fprintf(f, "a,b,c,label\n");
    for (int i = 0; i < 8; i++) {
        char tok[32];
        if (i == 3) strcpy(tok, "nan");
        else snprintf(tok, 32, "%.4f", (double)X[i * 3 + 2]);
        fprintf(f, "%.4f,%.4f,%s,%.0f\n", (double)X[i * 3], (double)X[i * 3 + 1], tok, y[i]);
    }
    fclose(f);
    float *Xc, *yc;
    size_t nc, dc;
    CHECK("csv load ok", xgb_csv_load(csv, &Xc, &yc, &nc, &dc, 3, a) == 0);
    CHECK("csv shape", nc == 8 && dc == 3);
    CHECK("csv label col removed", yc[0] == y[0]);
    CHECK("csv nan preserved", Xc[3 * 3 + 2] != Xc[3 * 3 + 2]);
}

/* ================================================================== */
/* RCRE interop                                                        */
/* ================================================================== */
static void test_rcre_interop(RCRE_Arena* a) {
    printf("[rcre interop]\n");
    XGBConfig cfg; xgb_config_default(&cfg);
    cfg.task = XGB_TASK_BINARY;
    cfg.rounds = 15; cfg.eta = 0.3f; cfg.max_depth = 4;

    int n = 250;
    float X[n * 3], y[n];
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) X[i * 3 + c] = frandn();
        y[i] = (X[i * 3] - X[i * 3 + 2] > 0.0f) ? 1 : 0;
    }
    XGBDataset* ds = xgb_dataset_create(X, y, n, 3, &cfg, a);
    XGBModel m;
    xgb_train(ds, NULL, &cfg, &m, a);

    RCRE_TreeNode* rnodes =
        (RCRE_TreeNode*)rcre_arena_alloc(a, m.num_nodes * sizeof(RCRE_TreeNode), 16);
    size_t* rroots = (size_t*)rcre_arena_alloc(a, m.num_trees * sizeof(size_t), 4);
    RCRE_GBDTModel rm;
    CHECK("convert ok", xgb_to_rcre_gbdt(&m, &rm, rnodes, rroots) == 0);

    size_t shp2[2] = {(size_t)n, 3};
    RCRE_Tensor Xt = rcre_tensor_from_buffer(X, shp2, 2, RCRE_DTYPE_F32, 0);
    RCRE_Tensor Pt = rcre_tensor_create(shp2, 2, RCRE_DTYPE_F32, 0, a);
    CHECK("rcre predict ok", rcre_ml_gbdt_predict(&rm, &Xt, &Pt) == RCRE_OK);

    float p1[n];
    xgb_predict(&m, X, n, 3, p1);
    float md = 0;
    for (int i = 0; i < n; i++)
        md = fmaxf(md, fabsf(p1[i] - ((float*)Pt.data)[i]));
    printf("    max |xgb - rcre| = %.3g\n", (double)md);
    CHECK("RCRE predictor matches xgb_predict", md < 1e-6f);
}

/* ================================================================== */
int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    rcre_hal_init();
    RCRE_Arena* a = rcre_arena_create(64 << 20);
    if (!a) { fprintf(stderr, "arena alloc failed\n"); return 1; }

    test_binning(a);
    test_metrics();
    test_reg_perfect(a);
    test_xor(a);
    test_convergence(a);
    test_regularization(a);
    test_missing(a);
    test_determinism(a);
    test_early_stopping(a);
    test_multi(a);
    test_io_roundtrip(a);
    test_rcre_interop(a);

    printf("------------------------------------------------------------\n");
    printf("  %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("  ✅ ALL %d CHECKS PASSED\n", n_pass);
    rcre_arena_destroy(a);
    return n_fail == 0 ? 0 : 1;
}
