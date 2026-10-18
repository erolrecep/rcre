/**
 * @file gbdt_infer.c
 * @brief Flat-Array Gradient Boosted Decision Tree (GBDT) Inference Engine
 * 
 * Satisfies ML-REQ-03.
 * Zero-allocation, cache-friendly flat-array tree traversal.
 */

#include "rcre.h"
#include <math.h>

#ifndef RCRE_FAST_SIGMOID_DEFINED
#define RCRE_FAST_SIGMOID_DEFINED
static inline float rcre_fast_sigmoidf(float z) {
    if (z > 16.0f) return 1.0f;
    if (z < -16.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}
#endif

RCRE_NODISCARD int rcre_ml_gbdt_predict(const RCRE_GBDTModel* model, 
                                        const RCRE_Tensor* X, 
                                        RCRE_Tensor* preds) {
    if (!model || !model->nodes || !model->tree_roots || !X || !X->data || !preds || !preds->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (X->ndim != 2 || preds->ndim < 1) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    if (preds->shape[0] != N) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* X_data = (const float*)X->data;
    float* p_data = (float*)preds->data;

    for (size_t i = 0; i < N; ++i) {
        const float* row = &X_data[i * D];
        float score = model->base_score;

        for (size_t t = 0; t < model->num_trees; ++t) {
            uint32_t curr = (uint32_t)model->tree_roots[t];

            while (true) {
                const RCRE_TreeNode* node = &model->nodes[curr];
                if (node->is_leaf) {
                    score += node->leaf_value;
                    break;
                }

                if (node->feature_idx < D && row[node->feature_idx] <= node->threshold) {
                    curr = node->left_child;
                } else {
                    curr = node->right_child;
                }
            }
        }

        if (model->is_binary_classification) {
            p_data[i] = rcre_fast_sigmoidf(score);
        } else {
            p_data[i] = score;
        }
    }

    return RCRE_OK;
}
