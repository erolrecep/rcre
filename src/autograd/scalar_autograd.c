/**
 * @file scalar_autograd.c
 * @brief ISO C23 Micrograd-style scalar reverse-mode automatic differentiation engine.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static uint32_t g_visited_epoch = 1;

static RCRE_ScalarVal* rcre_scalar_alloc_internal(RCRE_GraphCtx* ctx, bool is_param) {
    RCRE_Arena* target_arena = (ctx != nullptr)
        ? (is_param ? ctx->persistent_arena : ctx->tape_arena)
        : nullptr;

    RCRE_ScalarVal* v = nullptr;
    if (target_arena != nullptr) {
        v = (RCRE_ScalarVal*)rcre_arena_alloc(target_arena, sizeof(RCRE_ScalarVal), alignof(RCRE_ScalarVal));
    } else {
        v = (RCRE_ScalarVal*)malloc(sizeof(RCRE_ScalarVal));
    }
    if (!v) return nullptr;

    v->data = 0.0;
    v->grad = 0.0;
    v->backward = nullptr;
    v->prev[0] = nullptr;
    v->prev[1] = nullptr;
    v->op = RCRE_SCALAR_OP_CONST;
    v->label = nullptr;
    v->visited_id = 0;
    return v;
}

RCRE_ScalarVal* rcre_scalar_val_new(RCRE_GraphCtx* ctx, double data, const char* label) {
    RCRE_ScalarVal* v = rcre_scalar_alloc_internal(ctx, false);
    if (!v) return nullptr;
    v->data = data;
    v->label = label;
    v->op = RCRE_SCALAR_OP_CONST;
    return v;
}

RCRE_ScalarVal* rcre_scalar_val_param(RCRE_GraphCtx* ctx, double data, const char* label) {
    RCRE_ScalarVal* v = rcre_scalar_alloc_internal(ctx, true);
    if (!v) return nullptr;
    v->data = data;
    v->label = label;
    v->op = RCRE_SCALAR_OP_PARAM;
    return v;
}

RCRE_ScalarVal* rcre_scalar_val_const(RCRE_GraphCtx* ctx, double data) {
    return rcre_scalar_val_new(ctx, data, nullptr);
}

/* Backward callbacks */
static void scalar_backward_add(RCRE_ScalarVal* self) {
    if (self->prev[0]) self->prev[0]->grad += self->grad;
    if (self->prev[1]) self->prev[1]->grad += self->grad;
}

static void scalar_backward_sub(RCRE_ScalarVal* self) {
    if (self->prev[0]) self->prev[0]->grad += self->grad;
    if (self->prev[1]) self->prev[1]->grad -= self->grad;
}

static void scalar_backward_mul(RCRE_ScalarVal* self) {
    if (self->prev[0] && self->prev[1]) {
        self->prev[0]->grad += self->prev[1]->data * self->grad;
        self->prev[1]->grad += self->prev[0]->data * self->grad;
    }
}

static void scalar_backward_div(RCRE_ScalarVal* self) {
    if (self->prev[0] && self->prev[1]) {
        double d = self->prev[1]->data;
        self->prev[0]->grad += (1.0 / d) * self->grad;
        self->prev[1]->grad += (-self->prev[0]->data / (d * d)) * self->grad;
    }
}

static void scalar_backward_pow(RCRE_ScalarVal* self) {
    if (self->prev[0] && self->prev[1]) {
        double base = self->prev[0]->data;
        double exp_val = self->prev[1]->data;
        self->prev[0]->grad += (exp_val * pow(base, exp_val - 1.0)) * self->grad;
        if (base > 0.0) {
            self->prev[1]->grad += (pow(base, exp_val) * log(base)) * self->grad;
        }
    }
}

static void scalar_backward_relu(RCRE_ScalarVal* self) {
    if (self->prev[0]) {
        self->prev[0]->grad += (self->data > 0.0 ? 1.0 : 0.0) * self->grad;
    }
}

static void scalar_backward_tanh(RCRE_ScalarVal* self) {
    if (self->prev[0]) {
        double t = self->data;
        self->prev[0]->grad += (1.0 - t * t) * self->grad;
    }
}

static void scalar_backward_sigmoid(RCRE_ScalarVal* self) {
    if (self->prev[0]) {
        double s = self->data;
        self->prev[0]->grad += s * (1.0 - s) * self->grad;
    }
}

static void scalar_backward_exp(RCRE_ScalarVal* self) {
    if (self->prev[0]) {
        self->prev[0]->grad += self->data * self->grad;
    }
}

static void scalar_backward_log(RCRE_ScalarVal* self) {
    if (self->prev[0]) {
        self->prev[0]->grad += (1.0 / (self->prev[0]->data + 1e-15)) * self->grad;
    }
}

static void scalar_backward_neg(RCRE_ScalarVal* self) {
    if (self->prev[0]) {
        self->prev[0]->grad -= self->grad;
    }
}

RCRE_ScalarVal* rcre_scalar_add(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = a->data + b->data;
    out->prev[0] = a;
    out->prev[1] = b;
    out->op = RCRE_SCALAR_OP_ADD;
    out->backward = scalar_backward_add;
    return out;
}

RCRE_ScalarVal* rcre_scalar_sub(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = a->data - b->data;
    out->prev[0] = a;
    out->prev[1] = b;
    out->op = RCRE_SCALAR_OP_SUB;
    out->backward = scalar_backward_sub;
    return out;
}

RCRE_ScalarVal* rcre_scalar_mul(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = a->data * b->data;
    out->prev[0] = a;
    out->prev[1] = b;
    out->op = RCRE_SCALAR_OP_MUL;
    out->backward = scalar_backward_mul;
    return out;
}

RCRE_ScalarVal* rcre_scalar_div(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = a->data / (b->data + 1e-15);
    out->prev[0] = a;
    out->prev[1] = b;
    out->op = RCRE_SCALAR_OP_DIV;
    out->backward = scalar_backward_div;
    return out;
}

RCRE_ScalarVal* rcre_scalar_pow(RCRE_GraphCtx* ctx, RCRE_ScalarVal* base, RCRE_ScalarVal* exponent) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = pow(base->data, exponent->data);
    out->prev[0] = base;
    out->prev[1] = exponent;
    out->op = RCRE_SCALAR_OP_POW;
    out->backward = scalar_backward_pow;
    return out;
}

RCRE_ScalarVal* rcre_scalar_relu(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = a->data > 0.0 ? a->data : 0.0;
    out->prev[0] = a;
    out->op = RCRE_SCALAR_OP_RELU;
    out->backward = scalar_backward_relu;
    return out;
}

RCRE_ScalarVal* rcre_scalar_tanh(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = tanh(a->data);
    out->prev[0] = a;
    out->op = RCRE_SCALAR_OP_TANH;
    out->backward = scalar_backward_tanh;
    return out;
}

RCRE_ScalarVal* rcre_scalar_sigmoid(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = 1.0 / (1.0 + exp(-a->data));
    out->prev[0] = a;
    out->op = RCRE_SCALAR_OP_SIGMOID;
    out->backward = scalar_backward_sigmoid;
    return out;
}

RCRE_ScalarVal* rcre_scalar_exp(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = exp(a->data);
    out->prev[0] = a;
    out->op = RCRE_SCALAR_OP_EXP;
    out->backward = scalar_backward_exp;
    return out;
}

RCRE_ScalarVal* rcre_scalar_log(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = log(a->data > 1e-15 ? a->data : 1e-15);
    out->prev[0] = a;
    out->op = RCRE_SCALAR_OP_LOG;
    out->backward = scalar_backward_log;
    return out;
}

RCRE_ScalarVal* rcre_scalar_neg(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a) {
    RCRE_ScalarVal* out = rcre_scalar_alloc_internal(ctx, false);
    if (!out) return nullptr;
    out->data = -a->data;
    out->prev[0] = a;
    out->op = RCRE_SCALAR_OP_NEG;
    out->backward = scalar_backward_neg;
    return out;
}

/* Topological sort vector */
typedef struct {
    RCRE_ScalarVal** items;
    size_t count;
    size_t capacity;
} RCRE_ValVec;

static void scalar_vec_push(RCRE_ValVec* vec, RCRE_ScalarVal* v) {
    if (vec->count >= vec->capacity) {
        vec->capacity = vec->capacity == 0 ? 128 : vec->capacity * 2;
        vec->items = (RCRE_ScalarVal**)realloc(vec->items, vec->capacity * sizeof(RCRE_ScalarVal*));
    }
    if (vec->items) {
        vec->items[vec->count++] = v;
    }
}

static void build_topo(RCRE_ScalarVal* v, uint32_t epoch, RCRE_ValVec* topo) {
    if (v == nullptr || v->visited_id == epoch) {
        return;
    }
    v->visited_id = epoch;
    if (v->prev[0] != nullptr) build_topo(v->prev[0], epoch, topo);
    if (v->prev[1] != nullptr) build_topo(v->prev[1], epoch, topo);
    scalar_vec_push(topo, v);
}

void rcre_scalar_backward(RCRE_ScalarVal* root) {
    if (root == nullptr) return;

    uint32_t epoch = ++g_visited_epoch;
    RCRE_ValVec topo = { .items = nullptr, .count = 0, .capacity = 0 };
    build_topo(root, epoch, &topo);

    root->grad = 1.0;

    for (size_t i = topo.count; i > 0; --i) {
        RCRE_ScalarVal* curr = topo.items[i - 1];
        if (curr->backward != nullptr) {
            curr->backward(curr);
        }
    }

    if (topo.items != nullptr) {
        free(topo.items);
    }
}

void rcre_scalar_zero_grad(RCRE_ScalarVal* root) {
    if (root == nullptr) return;
    uint32_t epoch = ++g_visited_epoch;
    RCRE_ValVec topo = { .items = nullptr, .count = 0, .capacity = 0 };
    build_topo(root, epoch, &topo);
    for (size_t i = 0; i < topo.count; ++i) {
        topo.items[i]->grad = 0.0;
    }
    if (topo.items != nullptr) {
        free(topo.items);
    }
}

bool rcre_scalar_export_dot(RCRE_ScalarVal* root, const char* filename) {
    if (root == nullptr || filename == nullptr) return false;

    FILE* f = fopen(filename, "w");
    if (!f) return false;

    fprintf(f, "digraph AutogradGraph {\n");
    fprintf(f, "  rankdir=LR;\n");
    fprintf(f, "  node [shape=record, style=filled, fillcolor=\"#F0F4F8\", fontname=\"Helvetica\"];\n");

    uint32_t epoch = ++g_visited_epoch;
    RCRE_ValVec topo = { .items = nullptr, .count = 0, .capacity = 0 };
    build_topo(root, epoch, &topo);

    for (size_t i = 0; i < topo.count; ++i) {
        RCRE_ScalarVal* n = topo.items[i];
        const char* lbl = n->label ? n->label : "";
        fprintf(f, "  node_%p [label=\"{%s|data: %.4f|grad: %.4f}\"];\n", (void*)n, lbl, n->data, n->grad);
        if (n->prev[0] != nullptr) {
            fprintf(f, "  node_%p -> node_%p;\n", (void*)n->prev[0], (void*)n);
        }
        if (n->prev[1] != nullptr) {
            fprintf(f, "  node_%p -> node_%p;\n", (void*)n->prev[1], (void*)n);
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    if (topo.items != nullptr) free(topo.items);
    return true;
}
