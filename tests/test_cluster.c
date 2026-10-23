/**
 * @file test_cluster.c
 * @brief Unit tests for RCRE Clustering (k-Means) & Dimensionality Reduction (PCA)
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-2f

static void test_kmeans(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 6 points: 3 around (1, 1), 3 around (10, 10) */
    const size_t N = 6;
    const size_t D = 2;
    const size_t K = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[12] = {
        0.9f, 1.1f,
        1.0f, 0.9f,
        1.1f, 1.0f,
        9.9f, 10.1f,
        10.0f, 9.9f,
        10.1f, 10.0f
    };
    for (size_t i = 0; i < N * D; ++i) {
        ((float*)X.data)[i] = X_raw[i];
    }

    size_t C_shape[2] = {K, D};
    RCRE_Tensor centroids = rcre_tensor_create(C_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t L_shape[2] = {N, 1};
    RCRE_Tensor labels = rcre_tensor_create(L_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_ml_kmeans_fit(&X, K, 50, &centroids, &labels, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* C_data = (float*)centroids.data;

    /* Sort or identify which centroid is near (1, 1) and which is near (10, 10) */
    size_t idx_low = (C_data[0] < C_data[2]) ? 0 : 1;
    size_t idx_high = 1 - idx_low;

    ASSERT_FLOAT_NEAR(1.0f, C_data[idx_low * 2 + 0], 0.2f);
    ASSERT_FLOAT_NEAR(1.0f, C_data[idx_low * 2 + 1], 0.2f);
    ASSERT_FLOAT_NEAR(10.0f, C_data[idx_high * 2 + 0], 0.2f);
    ASSERT_FLOAT_NEAR(10.0f, C_data[idx_high * 2 + 1], 0.2f);

    /* Test prediction on unseen points */
    float test_raw[4] = {
        1.2f, 0.8f,   /* Should be cluster low */
        10.2f, 9.8f   /* Should be cluster high */
    };
    size_t test_shape[2] = {2, D};
    RCRE_Tensor test_X = rcre_tensor_create(test_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    memcpy(test_X.data, test_raw, sizeof(test_raw));

    size_t pred_shape[2] = {2, 1};
    RCRE_Tensor pred_labels = rcre_tensor_create(pred_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    status = rcre_ml_kmeans_predict(&test_X, &centroids, &pred_labels);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* preds = (float*)pred_labels.data;
    ASSERT_UINT_EQ(idx_low, (size_t)preds[0]);
    ASSERT_UINT_EQ(idx_high, (size_t)preds[1]);

    rcre_arena_destroy(arena);
}

static void test_kmeans_ex(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t N = 6;
    const size_t D = 2;
    const size_t K = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[12] = {
        0.9f, 1.1f,  1.0f, 0.9f,  1.1f, 1.0f,
        9.9f, 10.1f, 10.0f, 9.9f, 10.1f, 10.0f
    };
    memcpy(X.data, X_raw, sizeof(X_raw));

    size_t C_shape[2] = {K, D};
    RCRE_Tensor centroids = rcre_tensor_create(C_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t L_shape[2] = {N, 1};
    RCRE_Tensor labels = rcre_tensor_create(L_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    RCRE_KMeansParams params = {
        .max_iter = 100,
        .n_init = 5,
        .tol = 1e-5f,
        .seed = 987654321ULL
    };

    float inertia = 0.0f;
    int status = rcre_ml_kmeans_fit_ex(&X, K, &params, &centroids, &labels, &inertia, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_TRUE(inertia > 0.0f && inertia < 1.0f); /* Tight clusters have tiny WCSS */

    float* C_data = (float*)centroids.data;
    size_t idx_low = (C_data[0] < C_data[2]) ? 0 : 1;
    size_t idx_high = 1 - idx_low;
    ASSERT_FLOAT_NEAR(1.0f, C_data[idx_low * 2 + 0], 0.2f);
    ASSERT_FLOAT_NEAR(10.0f, C_data[idx_high * 2 + 0], 0.2f);

    rcre_arena_destroy(arena);
}

static void test_dbscan(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 7 points: 3 in cluster A, 3 in cluster B, 1 outlier */
    const size_t N = 7;
    const size_t D = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[14] = {
        1.0f, 1.0f,   1.1f, 1.0f,   1.0f, 1.1f,   /* Cluster 0 */
        10.0f, 10.0f, 10.1f, 10.0f, 10.0f, 10.1f, /* Cluster 1 */
        50.0f, 50.0f                               /* Isolated Outlier */
    };
    memcpy(X.data, X_raw, sizeof(X_raw));

    size_t L_shape[2] = {N, 1};
    RCRE_Tensor labels = rcre_tensor_create(L_shape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);

    RCRE_DBSCANParams params = {
        .eps = 0.5f,
        .min_samples = 2
    };

    size_t num_clusters = 0;
    int status = rcre_ml_dbscan_fit(&X, &params, &labels, &num_clusters, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_UINT_EQ(2, num_clusters);

    int32_t* l_data = (int32_t*)labels.data;
    /* First 3 points must share same cluster ID */
    ASSERT_TRUE(l_data[0] >= 0);
    ASSERT_INT_EQ(l_data[0], l_data[1]);
    ASSERT_INT_EQ(l_data[0], l_data[2]);

    /* Points 3, 4, 5 must share same cluster ID, different from first */
    ASSERT_TRUE(l_data[3] >= 0);
    ASSERT_TRUE(l_data[3] != l_data[0]);
    ASSERT_INT_EQ(l_data[3], l_data[4]);
    ASSERT_INT_EQ(l_data[3], l_data[5]);

    /* Outlier point 6 must be labeled as noise (-1) */
    ASSERT_INT_EQ(-1, l_data[6]);

    rcre_arena_destroy(arena);
}

static void test_pca(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Points along line y = 2x with small noise */
    const size_t N = 5;
    const size_t D = 2;
    const size_t K = 1;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[10] = {
        -2.0f, -4.0f,
        -1.0f, -2.0f,
         0.0f,  0.0f,
         1.0f,  2.0f,
         2.0f,  4.0f
    };
    memcpy(X.data, X_raw, sizeof(X_raw));

    size_t W_shape[2] = {K, D};
    RCRE_Tensor components = rcre_tensor_create(W_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t M_shape[2] = {D, 1};
    RCRE_Tensor mean = rcre_tensor_create(M_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t P_shape[2] = {N, K};
    RCRE_Tensor X_proj = rcre_tensor_create(P_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_ml_pca_fit(&X, K, &components, &mean, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* W_data = (float*)components.data;
    /* First principal component should align with direction [1, 2] / sqrt(5) = [0.4472, 0.8944] */
    float expected_x = 1.0f / sqrtf(5.0f);
    float expected_y = 2.0f / sqrtf(5.0f);

    /* Sign ambiguity in eigenvectors: either [e_x, e_y] or [-e_x, -e_y] */
    bool match_pos = (fabsf(W_data[0] - expected_x) < EPSILON && fabsf(W_data[1] - expected_y) < EPSILON);
    bool match_neg = (fabsf(W_data[0] - (-expected_x)) < EPSILON && fabsf(W_data[1] - (-expected_y)) < EPSILON);
    ASSERT_TRUE(match_pos || match_neg);

    status = rcre_ml_pca_transform(&X, &components, &mean, &X_proj);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Test inverse transformation & exact 1D line reconstruction */
    size_t R_shape[2] = {N, D};
    RCRE_Tensor X_recon = rcre_tensor_create(R_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    status = rcre_ml_pca_inverse_transform(&X_proj, &components, &mean, &X_recon);
    ASSERT_INT_EQ(RCRE_OK, status);

    const float* X_orig = (const float*)X.data;
    const float* X_rec = (const float*)X_recon.data;
    ASSERT_FLOAT_ARRAY_NEAR(X_orig, X_rec, N * D, 1e-4f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Clustering & Dimensionality Reduction (ML-REQ-02)");
    TEST_RUN(test_kmeans);
    TEST_RUN(test_kmeans_ex);
    TEST_RUN(test_dbscan);
    TEST_RUN(test_pca);
    return test_suite_report();
}
