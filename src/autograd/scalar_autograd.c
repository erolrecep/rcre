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

/* Per-thread visited-epoch counter (C11 thread_local). Marking is done by
 * epoch comparison instead of clearing `visited_id` after every traversal,
 * and each thread owns its own counter, so concurrent scalar backward()
 * calls on DISJOINT graphs from different threads are safe. Traversing the
 * same graph from two threads simultaneously is still not supported (grad
 * accumulation is not atomic). Wraparound needs 2^32 traversals per thread.
 */
static thread_local uint32_t g_visited_epoch = 1;

static RCRE_ScalarVal* rcre_scalar_alloc_internal(RCRE_GraphCtx* ctx, bool is_param) {
    if (ctx == nullptr) return nullptr;
    RCRE_Arena* target_arena = (is_param ? ctx->persistent_arena : ctx->tape_arena);
    if (target_arena == nullptr) return nullptr;

    RCRE_ScalarVal* v = (RCRE_ScalarVal*)rcre_arena_alloc(target_arena, sizeof(RCRE_ScalarVal), alignof(RCRE_ScalarVal));
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
        size_t new_cap = (vec->capacity == 0) ? 128 : vec->capacity * 2;
        /* Realloc via temporary pointer: on failure the old buffer stays
         * intact and is returned; callers treat a missing node as OOM. */
        RCRE_ScalarVal** new_items = (RCRE_ScalarVal**)realloc(vec->items, new_cap * sizeof(RCRE_ScalarVal*));
        if (!new_items) return;
        vec->items = new_items;
        vec->capacity = new_cap;
    }
    vec->items[vec->count++] = v;
}

/* Iterative postorder topological sort (explicit stack).
 *
 * The previous recursive DFS consumed ~100 bytes of C stack per node, so a
 * chain of ~50k nodes (plausible for unrolled RNNs or JIT-generated DAGs)
 * overflowed an 8 MB stack. The explicit frame stack lives on the heap and
 * supports arbitrarily deep chains. Cycles are not detected: as before, a
 * back-edge is simply treated as an already-visited node.
 */
typedef struct {
    RCRE_ScalarVal* node;
    uint8_t         stage; /* 0 = entering, 1 = children done, emit node */
} topo_frame_t;

typedef struct {
    topo_frame_t* frames;
    size_t        count;
    size_t        capacity;
} RCRE_FrameStack;

static bool frame_push(RCRE_FrameStack* fs, RCRE_ScalarVal* node, uint8_t stage) {
    if (fs->count >= fs->capacity) {
        size_t new_cap = (fs->capacity == 0) ? 256 : fs->capacity * 2;
        topo_frame_t* nf = (topo_frame_t*)realloc(fs->frames, new_cap * sizeof(topo_frame_t));
        if (!nf) return false; /* OOM: old stack intact; traversal stops early */
        fs->frames = nf;
        fs->capacity = new_cap;
    }
    fs->frames[fs->count].node = node;
    fs->frames[fs->count].stage = stage;
    fs->count++;
    return true;
}

static bool build_topo(RCRE_ScalarVal* root, uint32_t epoch, RCRE_ValVec* topo) {
    if (root == nullptr) return true;

    RCRE_FrameStack fs = { .frames = nullptr, .count = 0, .capacity = 0 };
    if (!frame_push(&fs, root, 0)) return false;

    while (fs.count > 0) {
        topo_frame_t* top = &fs.frames[fs.count - 1];
        RCRE_ScalarVal* v = top->node;

        if (top->stage == 0) {
            if (v->visited_id == epoch) {
                fs.count--; /* fully processed already (DAG sharing) */
                continue;
            }
            v->visited_id = epoch;
            top->stage = 1;
            /* Push prev[1] first so prev[0]'s subtree is emitted first,
             * matching the old recursive child order exactly. A failed
             * push aborts the traversal (caller bails: partial topo would
             * produce silently wrong gradients). */
            if (v->prev[1] != nullptr && v->prev[1]->visited_id != epoch) {
                if (!frame_push(&fs, v->prev[1], 0)) { free(fs.frames); return false; }
            }
            if (v->prev[0] != nullptr && v->prev[0]->visited_id != epoch) {
                if (!frame_push(&fs, v->prev[0], 0)) { free(fs.frames); return false; }
            }
        } else {
            fs.count--;
            scalar_vec_push(topo, v);
        }
    }

    free(fs.frames);
    return true;
}

void rcre_scalar_backward(RCRE_ScalarVal* root) {
    if (root == nullptr) return;

    uint32_t epoch = ++g_visited_epoch;
    RCRE_ValVec topo = { .items = nullptr, .count = 0, .capacity = 0 };
    if (!build_topo(root, epoch, &topo)) {
        free(topo.items); /* OOM: no gradient update rather than a wrong one */
        return;
    }

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
    if (!build_topo(root, epoch, &topo)) {
        free(topo.items);
        return;
    }
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
    if (!build_topo(root, epoch, &topo)) {
        fclose(f);
        remove(filename); /* leave no partial artifact behind */
        return false;
    }

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
