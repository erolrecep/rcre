# rcre-xgb

XGBoost-style gradient boosted trees (training + inference) on the RCRE C23
framework. Quantile binning, regularized second-order histogram splits,
missing-value default directions, row/column subsampling, early stopping,
regression / binary / multi-class, CSV + binary model I/O, and a zero-copy
adapter into RCRE's flat GBDT predictor.

**~2.8k lines, zero external dependencies, single Makefile, arena-allocated.**

Full design + benchmark + bug-log documentation: **[xgboost_report.md](xgboost_report.md)**

## Quickstart

```sh
make all          # lib + CLI + tests + bench
make test         # 55 checks
make bench        # scaling / per-stage / AUC / RCRE parity
make demo         # synthetic regression + binary + persistence demo

bin/xgb_cli train data.csv --task binary --label 3 \
      --rounds 50 --depth 4 --eta 0.3 --out model.rcxg
bin/xgb_cli eval model.rcxg data.csv --label 3
bin/xgb_cli export model.rcxg
```

## C API

```c
#include "rcre.h"   /* must precede xgb.h */
#include "xgb.h"

rcre_hal_init();
RCRE_Arena* arena = rcre_arena_create(256 << 20);

XGBConfig cfg; xgb_config_default(&cfg);
cfg.task = XGB_TASK_BINARY; cfg.rounds = 100; cfg.eta = 0.3f;
cfg.max_depth = 5; cfg.early_stop_rounds = 10; cfg.seed = 1;

XGBDataset* ds = xgb_dataset_create(Xtr, ytr, ntr, d, &cfg, arena);
XGBModel m;
xgb_train(ds, ds_val, &cfg, &m, arena);

float p[nva];
xgb_predict(&m, Xva, nva, d, p);
float auc = xgb_auc(yva, p, nva);

xgb_model_save(&m, "model.rcxg");   /* bit-exact roundtrip via xgb_model_load */
```

NaNs in `X` are routed by each node's learned default direction.

## Layout

```
include/xgb.h      public API
src/xgb_core.c     quantile binning + histogram tree fitting
src/xgb_boost.c    boosting loop, base scores, early stopping
src/xgb_predict.c  inference + RCRE adapter
src/xgb_io.c       CSV loader + "RCXG" binary save/load
src/xgb_metrics.c  logloss / AUC / RMSE / MAE / accuracy
examples/          CLI (demo/train/predict/eval/export)
tests/             53-check suite
benchmarks/        scaling, per-stage, binary AUC, RCRE parity
third_party/       vendored RCRE amalgamation (fallback when ../rcre/dist is absent)
```

Build flags: `-std=c2x -O3 -Wall -Wextra`. Verified clean under
ASan + UBSan (tests and full benchmark battery).
