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

    return test_suite_report();
}
