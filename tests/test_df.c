/**
 * @file test_df.c
 * @brief Unit tests for RCRE Columnar DataFrame Engine (Phase 1)
 */

#include "rcre.h"
#include "rcre_df.h"
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void test_df_create_and_nulls(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"id", "value"};
    RCRE_DType types[] = {RCRE_DTYPE_I32, RCRE_DTYPE_F32};
    size_t nrows = 16;

    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 2, nrows, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(2, df.num_cols);
    ASSERT_UINT_EQ(16, df.num_rows);

    int32_t* id_data = (int32_t*)df.columns[0].data;
    float* val_data = (float*)df.columns[1].data;

    for (size_t i = 0; i < nrows; ++i) {
        id_data[i] = (int32_t)(i + 1);
        val_data[i] = (float)(i * 2.5f);
        ASSERT_FALSE(rcre_col_is_null(&df.columns[0], i));
        ASSERT_FALSE(rcre_col_is_null(&df.columns[1], i));
    }

    /* Set nulls on rows 3 and 10 of column 1 */
    rcre_col_set_valid(&df.columns[1], 3, false);
    rcre_col_set_valid(&df.columns[1], 10, false);

    ASSERT_TRUE(rcre_col_is_null(&df.columns[1], 3));
    ASSERT_TRUE(rcre_col_is_null(&df.columns[1], 10));
    ASSERT_FALSE(rcre_col_is_null(&df.columns[1], 2));
    ASSERT_FALSE(rcre_col_is_null(&df.columns[1], 4));
    ASSERT_UINT_EQ(2, df.columns[1].null_count);

    rcre_arena_destroy(arena);
}

static void test_df_from_to_tensor(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[] = {5, 3};
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* data = (float*)t.data;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            data[i * 3 + j] = (float)(i * 10.0f + j + 1.0f);
        }
    }

    const char* col_names[] = {"feat_a", "feat_b", "feat_c"};
    RCRE_DataFrame df;
    int rc = rcre_df_from_tensor(&t, col_names, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(5, df.num_rows);
    ASSERT_UINT_EQ(3, df.num_cols);
    ASSERT_STR_EQ("feat_b", df.columns[1].name);

    float* col1 = (float*)df.columns[1].data;
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_FLOAT_NEAR(i * 10.0f + 2.0f, col1[i], 1e-5f);
    }

    /* Round-trip back to tensor */
    size_t indices[] = {0, 1, 2};
    RCRE_Tensor t_back;
    rc = rcre_df_to_tensor(&df, indices, 3, RCRE_DTYPE_F32, arena, &t_back);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(2, t_back.ndim);
    ASSERT_UINT_EQ(5, t_back.shape[0]);
    ASSERT_UINT_EQ(3, t_back.shape[1]);
    const float* back_data = (const float*)t_back.data;
    for (size_t i = 0; i < 15; ++i) {
        ASSERT_FLOAT_NEAR(data[i], back_data[i], 1e-5f);
    }

    rcre_arena_destroy(arena);
}

static void test_df_csv_parsing(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* csv_path = "/tmp/rcre_test_ds.csv";
    FILE* fp = fopen(csv_path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "id,score,category\n");
    fprintf(fp, "1,98.5,10\n");
    fprintf(fp, "2,75.0,20\n");
    fprintf(fp, "3,82.5,10\n");
    fprintf(fp, "4,91.0,20\n");
    fclose(fp);

    RCRE_DataFrame df;
    int rc = rcre_df_from_csv(csv_path, true, ',', arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(4, df.num_rows);
    ASSERT_UINT_EQ(3, df.num_cols);
    ASSERT_STR_EQ("id", df.columns[0].name);
    ASSERT_STR_EQ("score", df.columns[1].name);
    ASSERT_STR_EQ("category", df.columns[2].name);

    const float* scores = (const float*)df.columns[1].data;
    ASSERT_FLOAT_NEAR(98.5f, scores[0], 1e-5f);
    ASSERT_FLOAT_NEAR(75.0f, scores[1], 1e-5f);
    ASSERT_FLOAT_NEAR(82.5f, scores[2], 1e-5f);
    ASSERT_FLOAT_NEAR(91.0f, scores[3], 1e-5f);

    rcre_arena_destroy(arena);
}

static void test_df_select_and_slice(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"a", "b", "c", "d"};
    RCRE_DType types[] = {RCRE_DTYPE_F32, RCRE_DTYPE_F32, RCRE_DTYPE_F32, RCRE_DTYPE_F32};
    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 4, 10, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);

    for (size_t c = 0; c < 4; ++c) {
        float* d = (float*)df.columns[c].data;
        for (size_t r = 0; r < 10; ++r) {
            d[r] = (float)(c * 100 + r);
        }
    }

    const char* sel_names[] = {"c", "a"};
    RCRE_DataFrame df_sel;
    rc = rcre_df_select(&df, sel_names, 2, arena, &df_sel);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(2, df_sel.num_cols);
    ASSERT_UINT_EQ(10, df_sel.num_rows);
    ASSERT_STR_EQ("c", df_sel.columns[0].name);
    ASSERT_STR_EQ("a", df_sel.columns[1].name);

    RCRE_DataFrame df_slice;
    rc = rcre_df_slice(&df_sel, 2, 5, arena, &df_slice);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(5, df_slice.num_rows);
    const float* c_sliced = (const float*)df_slice.columns[0].data;
    ASSERT_FLOAT_NEAR(202.0f, c_sliced[0], 1e-5f);
    ASSERT_FLOAT_NEAR(206.0f, c_sliced[4], 1e-5f);

    rcre_arena_destroy(arena);
}

static void test_df_filter_mask(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"id", "score"};
    RCRE_DType types[] = {RCRE_DTYPE_I32, RCRE_DTYPE_F32};
    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 2, 4, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);

    int32_t* ids = (int32_t*)df.columns[0].data;
    float* scores = (float*)df.columns[1].data;
    ids[0] = 1; scores[0] = 98.5f;
    ids[1] = 2; scores[1] = 75.0f;
    ids[2] = 3; scores[2] = 82.5f;
    ids[3] = 4; scores[3] = 91.0f;

    /* Filter mask: score > 85.0 -> rows 0 and 3 */
    uint8_t mask[4] = {1, 0, 0, 1};
    RCRE_DataFrame filtered;
    rc = rcre_df_filter(&df, mask, arena, &filtered);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(2, filtered.num_rows);

    const int32_t* f_ids = (const int32_t*)filtered.columns[0].data;
    const float* f_scores = (const float*)filtered.columns[1].data;
    ASSERT_INT_EQ(1, f_ids[0]);
    ASSERT_FLOAT_NEAR(98.5f, f_scores[0], 1e-5f);
    ASSERT_INT_EQ(4, f_ids[1]);
    ASSERT_FLOAT_NEAR(91.0f, f_scores[1], 1e-5f);

    rcre_arena_destroy(arena);
}

static void test_df_radix_sort(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"val"};
    RCRE_DType types[] = {RCRE_DTYPE_F32};
    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 1, 6, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);

    float* vals = (float*)df.columns[0].data;
    vals[0] = 42.0f;
    vals[1] = -10.5f;
    vals[2] = 0.0f;
    vals[3] = 100.25f;
    vals[4] = -50.0f;
    vals[5] = 15.0f;

    RCRE_DataFrame sorted_asc;
    rc = rcre_df_sort(&df, 0, true, arena, &sorted_asc);
    ASSERT_INT_EQ(RCRE_OK, rc);
    const float* asc = (const float*)sorted_asc.columns[0].data;
    ASSERT_FLOAT_NEAR(-50.0f, asc[0], 1e-5f);
    ASSERT_FLOAT_NEAR(-10.5f, asc[1], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, asc[2], 1e-5f);
    ASSERT_FLOAT_NEAR(15.0f, asc[3], 1e-5f);
    ASSERT_FLOAT_NEAR(42.0f, asc[4], 1e-5f);
    ASSERT_FLOAT_NEAR(100.25f, asc[5], 1e-5f);

    RCRE_DataFrame sorted_desc;
    rc = rcre_df_sort(&df, 0, false, arena, &sorted_desc);
    ASSERT_INT_EQ(RCRE_OK, rc);
    const float* desc = (const float*)sorted_desc.columns[0].data;
    ASSERT_FLOAT_NEAR(100.25f, desc[0], 1e-5f);
    ASSERT_FLOAT_NEAR(42.0f, desc[1], 1e-5f);
    ASSERT_FLOAT_NEAR(15.0f, desc[2], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, desc[3], 1e-5f);
    ASSERT_FLOAT_NEAR(-10.5f, desc[4], 1e-5f);
    ASSERT_FLOAT_NEAR(-50.0f, desc[5], 1e-5f);

    rcre_arena_destroy(arena);
}

static void test_df_groupby_aggregation(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"cat", "amount"};
    RCRE_DType types[] = {RCRE_DTYPE_I32, RCRE_DTYPE_F32};
    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 2, 4, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);

    int32_t* cat = (int32_t*)df.columns[0].data;
    float* amount = (float*)df.columns[1].data;
    cat[0] = 100; amount[0] = 10.0f;
    cat[1] = 200; amount[1] = 25.0f;
    cat[2] = 100; amount[2] = 40.0f;
    cat[3] = 200; amount[3] = 75.0f;

    RCRE_AggSpec specs[1] = {
        {.target_col_idx = 1, .op = RCRE_AGG_SUM, .out_name = "amount_sum"}
    };
    RCRE_DataFrame grp_sum;
    rc = rcre_df_groupby(&df, 0, specs, 1, arena, &grp_sum);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(2, grp_sum.num_rows);
    ASSERT_UINT_EQ(2, grp_sum.num_cols);

    const int32_t* g_keys = (const int32_t*)grp_sum.columns[0].data;
    const float* g_vals = (const float*)grp_sum.columns[1].data;

    for (size_t i = 0; i < 2; ++i) {
        if (g_keys[i] == 100) {
            ASSERT_FLOAT_NEAR(50.0f, g_vals[i], 1e-4f);
        } else if (g_keys[i] == 200) {
            ASSERT_FLOAT_NEAR(100.0f, g_vals[i], 1e-4f);
        } else {
            ASSERT_TRUE(false);
        }
    }

    rcre_arena_destroy(arena);
}

static void test_df_arrow_c_interface(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"f_col", "i_col"};
    RCRE_DType types[] = {RCRE_DTYPE_F32, RCRE_DTYPE_I32};
    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 2, 8, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);

    RCRE_ArrowArray arrow_arr;
    RCRE_ArrowSchema arrow_sch;

    rc = rcre_col_to_arrow(&df.columns[0], arena, &arrow_sch, &arrow_arr);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_STR_EQ("f", arrow_sch.format);
    ASSERT_STR_EQ("f_col", arrow_sch.name);
    ASSERT_UINT_EQ(8, arrow_arr.length);
    ASSERT_UINT_EQ(2, arrow_arr.n_buffers);
    ASSERT_NOT_NULL(arrow_arr.buffers[1]);

    /* Release Arrow objects via exported release callbacks */
    arrow_arr.release(&arrow_arr);
    arrow_sch.release(&arrow_sch);
    ASSERT_NULL(arrow_arr.release);
    ASSERT_NULL(arrow_sch.release);

    /* Test print function output */
    rcre_df_print(&df, 4);

    rcre_arena_destroy(arena);
}

typedef struct {
    int64_t* partial_sums;
} IndexedTestCtx;

static void test_indexed_sum_task(void* arg, size_t task_id, size_t worker_id) {
    IndexedTestCtx* ctx = (IndexedTestCtx*)arg;
    ctx->partial_sums[worker_id] += (int64_t)(task_id + 1U);
}

static void test_threadpool_indexed_reduction(void) {
    RCRE_ThreadPool* pool = rcre_threadpool_create_ex(3, RCRE_QOS_DEFAULT);
    ASSERT_NOT_NULL(pool);

    size_t nworkers = rcre_threadpool_worker_count(pool);
    ASSERT_UINT_EQ(4, nworkers); /* 3 workers + participating main thread */

    int64_t partials[4] = {0, 0, 0, 0};
    IndexedTestCtx ctx = { .partial_sums = partials };

    constexpr size_t N = 10000U;
    rcre_threadpool_parallel_for_indexed(pool, N, test_indexed_sum_task, &ctx);

    int64_t total = 0;
    for (size_t w = 0; w < nworkers; ++w) {
        total += partials[w];
    }
    int64_t expected = (int64_t)(N * (N + 1U) / 2U);
    ASSERT_TRUE(total == expected);

    rcre_threadpool_destroy(pool);
}

static void test_df_str_id_column_and_groupby(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* names[] = {"city", "temp"};
    RCRE_DType types[] = {RCRE_DTYPE_STR_ID, RCRE_DTYPE_F32};
    RCRE_DataFrame df;
    int rc = rcre_df_create(names, types, 2, 5, arena, &df);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_NOT_NULL(df.columns[0].str_pool);

    ASSERT_INT_EQ(RCRE_OK, rcre_col_set_str(&df.columns[0], 0, RCRE_STR("Hamburg")));
    ASSERT_INT_EQ(RCRE_OK, rcre_col_set_str(&df.columns[0], 1, RCRE_STR("Abéché")));
    ASSERT_INT_EQ(RCRE_OK, rcre_col_set_str(&df.columns[0], 2, RCRE_STR("Hamburg")));
    ASSERT_INT_EQ(RCRE_OK, rcre_col_set_str(&df.columns[0], 3, RCRE_STR("Abéché")));
    ASSERT_INT_EQ(RCRE_OK, rcre_col_set_str(&df.columns[0], 4, RCRE_STR("Tokyo")));

    float* temps = (float*)df.columns[1].data;
    temps[0] = 10.0f; temps[1] = 30.0f; temps[2] = 20.0f; temps[3] = 40.0f; temps[4] = 15.0f;

    ASSERT_TRUE(rcre_slice_eq(rcre_col_get_str(&df.columns[0], 1), RCRE_STR("Abéché")));

    RCRE_AggSpec specs[2] = {
        { .target_col_idx = 1, .op = RCRE_AGG_MEAN,  .out_name = "mean_temp" },
        { .target_col_idx = 1, .op = RCRE_AGG_COUNT, .out_name = "n" }
    };
    RCRE_DataFrame grp;
    rc = rcre_df_groupby(&df, 0, specs, 2, arena, &grp);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(3, grp.num_rows);
    ASSERT_UINT_EQ(3, grp.num_cols);

    for (size_t r = 0; r < grp.num_rows; ++r) {
        RCRE_Slice city = rcre_col_get_str(&grp.columns[0], r);
        float mean_v = ((const float*)grp.columns[1].data)[r];
        int32_t cnt  = ((const int32_t*)grp.columns[2].data)[r];
        if (rcre_slice_eq(city, RCRE_STR("Hamburg"))) {
            ASSERT_FLOAT_NEAR(15.0f, mean_v, 1e-4f);
            ASSERT_INT_EQ(2, cnt);
        } else if (rcre_slice_eq(city, RCRE_STR("Abéché"))) {
            ASSERT_FLOAT_NEAR(35.0f, mean_v, 1e-4f);
            ASSERT_INT_EQ(2, cnt);
        } else if (rcre_slice_eq(city, RCRE_STR("Tokyo"))) {
            ASSERT_FLOAT_NEAR(15.0f, mean_v, 1e-4f);
            ASSERT_INT_EQ(1, cnt);
        } else {
            ASSERT_TRUE(false);
        }
    }

    rcre_df_print(&grp, 5);
    rcre_arena_destroy(arena);
}

static void test_df_stream_csv_groupby(void) {
    rcre_hal_init();
    const char* path2 = "/tmp/rcre_test_stream_2col.csv";
    const char* data2 =
        "Hamburg;10.0\n"
        "Abéché;-5.5\n"
        "Hamburg;20.0\r\n"
        "Abéché;15.5\n"
        "Tokyo;12.3\n"
        "Hamburg;30.0\n";
    ASSERT_INT_EQ(RCRE_OK, rcre_io_write_atomic(path2, data2, strlen(data2)));

    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    RCRE_ThreadPool* pool = rcre_threadpool_create_ex(3, RCRE_QOS_DEFAULT);
    ASSERT_NOT_NULL(pool);

    RCRE_AggSpec specs[6] = {
        { .target_col_idx = 1, .op = RCRE_AGG_MIN,   .out_name = "min"   },
        { .target_col_idx = 1, .op = RCRE_AGG_MEAN,  .out_name = "mean"  },
        { .target_col_idx = 1, .op = RCRE_AGG_MAX,   .out_name = "max"   },
        { .target_col_idx = 1, .op = RCRE_AGG_SUM,   .out_name = "sum"   },
        { .target_col_idx = 1, .op = RCRE_AGG_COUNT, .out_name = "count" },
        { .target_col_idx = 1, .op = RCRE_AGG_STD,   .out_name = "std"   }
    };

    /* Test both mmap and bounded_ram streaming modes with tiny 32B slabs */
    for (int mode = 0; mode < 2; ++mode) {
        rcre_arena_reset(arena);
        RCRE_StreamGroupByOpts opts = {
            .delimiter      = ';',
            .has_header     = false,
            .key_col_idx    = 0,
            .val_col_idx    = 1,
            .specs          = specs,
            .num_specs      = 6,
            .pool           = pool,
            .slab_bytes     = 32,
            .table_cap      = 256,
            .bounded_ram    = (mode == 1),
            .sort_by_key    = true,
            .use_f64_output = true,
            .key_col_name   = "station"
        };
        RCRE_DataFrame out;
        int rc = rcre_df_stream_csv_groupby(path2, &opts, arena, &out);
        ASSERT_INT_EQ(RCRE_OK, rc);
        ASSERT_UINT_EQ(3, out.num_rows);
        ASSERT_UINT_EQ(7, out.num_cols);

        /* Sorted lexicographically: Abéché, Hamburg, Tokyo */
        ASSERT_TRUE(rcre_slice_eq(rcre_col_get_str(&out.columns[0], 0), RCRE_STR("Abéché")));
        ASSERT_TRUE(rcre_slice_eq(rcre_col_get_str(&out.columns[0], 1), RCRE_STR("Hamburg")));
        ASSERT_TRUE(rcre_slice_eq(rcre_col_get_str(&out.columns[0], 2), RCRE_STR("Tokyo")));

        const double* mins   = (const double*)out.columns[1].data;
        const double* means  = (const double*)out.columns[2].data;
        const double* maxs   = (const double*)out.columns[3].data;
        const double* sums   = (const double*)out.columns[4].data;
        const int64_t* cnts  = (const int64_t*)out.columns[5].data;
        const double* stds   = (const double*)out.columns[6].data;

        ASSERT_FLOAT_NEAR(-5.5f, (float)mins[0],  1e-5f);
        ASSERT_FLOAT_NEAR(5.0f,  (float)means[0], 1e-5f);
        ASSERT_FLOAT_NEAR(15.5f, (float)maxs[0],  1e-5f);
        ASSERT_TRUE(cnts[0] == 2);

        ASSERT_FLOAT_NEAR(10.0f, (float)mins[1],  1e-5f);
        ASSERT_FLOAT_NEAR(20.0f, (float)means[1], 1e-5f);
        ASSERT_FLOAT_NEAR(30.0f, (float)maxs[1],  1e-5f);
        ASSERT_FLOAT_NEAR(60.0f, (float)sums[1],  1e-5f);
        ASSERT_TRUE(cnts[1] == 3);
        ASSERT_FLOAT_NEAR(10.0f, (float)stds[1],  1e-4f); /* sample std of {10,20,30} == 10 */

        ASSERT_FLOAT_NEAR(12.3f, (float)means[2], 1e-5f);
        ASSERT_TRUE(cnts[2] == 1);
    }

    /* Also test multi-column CSV with header and non-adjacent key/val columns */
    const char* path4 = "/tmp/rcre_test_stream_4col.csv";
    const char* data4 =
        "id,region,notes,revenue\n"
        "1,EMEA,ok,100.5\n"
        "2,APAC,ok,250.25\n"
        "3,EMEA,ok,49.5\n";
    ASSERT_INT_EQ(RCRE_OK, rcre_io_write_atomic(path4, data4, strlen(data4)));

    rcre_arena_reset(arena);
    RCRE_AggSpec rev_specs[2] = {
        { .target_col_idx = 3, .op = RCRE_AGG_SUM,   .out_name = "total_rev" },
        { .target_col_idx = 3, .op = RCRE_AGG_COUNT, .out_name = "orders"    }
    };
    RCRE_StreamGroupByOpts opts4 = {
        .delimiter      = ',',
        .has_header     = true,
        .key_col_idx    = 1,
        .val_col_idx    = 3,
        .specs          = rev_specs,
        .num_specs      = 2,
        .pool           = pool,
        .slab_bytes     = 64,
        .table_cap      = 256,
        .bounded_ram    = false,
        .sort_by_key    = true,
        .use_f64_output = true
    };
    RCRE_DataFrame out4;
    ASSERT_INT_EQ(RCRE_OK, rcre_df_stream_csv_groupby(path4, &opts4, arena, &out4));
    ASSERT_STR_EQ("region", out4.columns[0].name);
    ASSERT_UINT_EQ(2, out4.num_rows);
    ASSERT_TRUE(rcre_slice_eq(rcre_col_get_str(&out4.columns[0], 0), RCRE_STR("APAC")));
    ASSERT_TRUE(rcre_slice_eq(rcre_col_get_str(&out4.columns[0], 1), RCRE_STR("EMEA")));
    ASSERT_FLOAT_NEAR(250.25f, (float)((const double*)out4.columns[1].data)[0], 1e-4f);
    ASSERT_FLOAT_NEAR(150.0f,  (float)((const double*)out4.columns[1].data)[1], 1e-4f);

    /* Test hash table saturation guard (tiny table_cap vs many distinct keys) */
    {
        rcre_arena_reset(arena);
        const char* path_overflow = "/tmp/rcre_test_overflow.csv";
        char buf[2048] = {0};
        size_t off = 0;
        for (int k = 0; k < 120; ++k) {
            off += snprintf(buf + off, sizeof(buf) - off, "city_%03d;%d.0\n", k, k);
        }
        ASSERT_INT_EQ(RCRE_OK, rcre_io_write_atomic(path_overflow, buf, off));

        RCRE_StreamGroupByOpts opts_overflow = {
            .delimiter   = ';',
            .has_header  = false,
            .key_col_idx = 0,
            .val_col_idx = 1,
            .table_cap   = 64, /* max 64 slots, 75% load = 48 entries, 120 keys must fail gracefully */
            .pool        = nullptr,
            .bounded_ram = false
        };
        RCRE_DataFrame out_of;
        int rc_overflow = rcre_df_stream_csv_groupby(path_overflow, &opts_overflow, arena, &out_of);
        ASSERT_INT_EQ(RCRE_ERR_OUT_OF_MEMORY, rc_overflow);
        remove(path_overflow);
    }

    remove(path2);
    remove(path4);
    rcre_threadpool_destroy(pool);
    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE High-Performance Columnar DataFrame Engine (Phase 1)");

    TEST_RUN(test_df_create_and_nulls);
    TEST_RUN(test_df_from_to_tensor);
    TEST_RUN(test_df_csv_parsing);
    TEST_RUN(test_df_select_and_slice);
    TEST_RUN(test_df_filter_mask);
    TEST_RUN(test_df_radix_sort);
    TEST_RUN(test_df_groupby_aggregation);
    TEST_RUN(test_df_arrow_c_interface);
    TEST_RUN(test_threadpool_indexed_reduction);
    TEST_RUN(test_df_str_id_column_and_groupby);
    TEST_RUN(test_df_stream_csv_groupby);

    return test_suite_report();
}
