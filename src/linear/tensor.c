/**
 * @file tensor.c
 * @brief Minimal tensor abstraction for 1D/2D numeric data.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "linear/lehmer.h"
#include "linear/quant.h"
#include "linear/tensor.h"

/**
 * Tensor shape
 * @{
 */

static inline size_t shape_count(const Shape* s) {
    size_t product = 1;
    for (size_t i = 0; i < s->id; ++i) {
        product *= s->dims[i];
    }
    return product;
}

static inline Shape shape_vec(size_t len) {
    return (Shape) {{len, 0}, SHAPE_VEC};
}

static inline Shape shape_mat(size_t rows, size_t cols) {
    return (Shape) {{rows, cols}, SHAPE_MAT};
}

/** @} */

/**
 * Tensor life-cycle
 * @{
 */

/**
 * private methods
 */

void tensor_assert_q8(Tensor* t) {
    if (t->shape.id == SHAPE_VEC && t->shape.dims[0] < Q8_BLOCK_SIZE) {
        fprintf(
            stderr,
            "tensor_new_q8: SHAPE_VEC dims[0]=%zu is less than Q8_BLOCK_SIZE=%d\n",
            t->shape.dims[0],
            Q8_BLOCK_SIZE
        );
        abort();
    }
}

void tensor_new_q8(Tensor* t) {
    tensor_assert_q8(t);

    // Q8 is an array of quant8_t for each row
    switch (t->shape.id) {
        case SHAPE_VEC: {
            // 1D: single quant8_t
            quant8_t* q = malloc(sizeof(quant8_t));
            *q = q8_vec_new(t->shape.dims[0]);
            t->data = q;
            break;
        }
        case SHAPE_MAT: {
            // 2D: array of quant8_t, one per row
            size_t rows = t->shape.dims[0];
            size_t cols = t->shape.dims[1];
            quant8_t* q = q8_mat_new(rows, cols);  // Already allocates and returns array
            t->data = q;
            break;
        }
    }
}

void tensor_free_q8(Tensor* t) {
    if (t) {
        switch (t->shape.id) {
            case SHAPE_VEC:
                q8_vec_free((quant8_t*) t->data);
                free(t->data);
                break;
            case SHAPE_MAT:
                q8_mat_free((quant8_t*) t->data, t->shape.dims[0]);
                break;
        }
    }
}

void tensor_new_em(Tensor* t) {
    t->data = malloc(shape_count(&t->shape) * type_size(t->id));
}

/**
 * public methods
 */

/**
 * @brief Create a new tensor with shape (1D or 2D) and type id.
 *        Allocates data and sets workspace to NULL.
 * @param shape Shape object (dims/n must be set)
 * @param id TypeId for storage (e.g. TYPE_F32, TYPE_Q8)
 * @return Tensor struct, .data is allocated, .workspace=NULL.
 */
Tensor tensor_new(Shape shape, TypeId id) {
    Tensor t = {0};
    t.shape = shape;
    t.id = id;
    if (id == TYPE_Q8) {
        tensor_new_q8(&t);
    } else {
        tensor_new_em(&t);
    }
    t.workspace = NULL;
    return t;
}

/**
 * @brief Free tensor data and workspace.
 */
void tensor_free(Tensor* t) {
    if (t) {
        if (t->data) {
            if (t->id == TYPE_Q8) {
                tensor_free_q8(t);
            } else {
                free(t->data);
            }
            t->data = NULL;
        }
        if (t->workspace) {
            free(t->workspace);
            t->workspace = NULL;
        }
    }
}

/** @} */

/**
 * Tensor fill
 * @{
 */

void tensor_fill(Tensor* t, float value) {
    switch (t->shape.id) {
        case SHAPE_VEC: {
            size_t len = t->shape.dims[0];
            float* src = malloc(len * sizeof(float));
            for (size_t i = 0; i < len; i++) {
                src[i] = value;
            }

            quant_vec(t->data, src, len, t->id);

            free(src);
            break;
        }
        case SHAPE_MAT: {
            size_t rows = t->shape.dims[0];
            size_t cols = t->shape.dims[1];
            float* src = malloc(cols * sizeof(float));
            for (size_t i = 0; i < cols; i++) {
                src[i] = value;
            }

            size_t stride = type_size(t->id);
            for (size_t r = 0; r < rows; r++) {
                void* dst;  // Each row's destination pointer
                if (t->id == TYPE_Q8) {
                    // Q8: one quant8_t per row
                    dst = (uint8_t*) t->data + r * stride;
                } else {
                    // Dense: row-major, one element per col
                    dst = (uint8_t*) t->data + r * cols * stride;
                }
                quant_vec(dst, src, cols, t->id);
            }
            free(src);
            break;
        }
    }
}

void tensor_zeros(Tensor* t) {
    tensor_fill(t, 0.0f);
}

void tensor_ones(Tensor* t) {
    tensor_fill(t, 1.0f);
}

/** @} */

/**
 * Tensor initialization
 * @{
 */

// note that this is internal use only
void tensor_init(Tensor* t, LehmerFn prng, void* args) {
    switch (t->shape.id) {
        case SHAPE_VEC: {
            size_t len = t->shape.dims[0];
            float* src = malloc(len * sizeof(float));
            for (size_t i = 0; i < len; i++) {
                src[i] = prng(args);
            }
            quant_vec(t->data, src, len, t->id);
            free(src);
            break;
        }
        case SHAPE_MAT: {
            size_t rows = t->shape.dims[0];
            size_t cols = t->shape.dims[1];
            float* src = malloc(cols * sizeof(float));
            size_t stride = type_size(t->id);
            for (size_t r = 0; r < rows; r++) {
                // fill row buffer
                for (size_t c = 0; c < cols; c++) {
                    src[c] = prng(args);
                }

                // get current vec from mat
                void* dst;
                if (t->id == TYPE_Q8) {
                    dst = (uint8_t*) t->data + r * stride;
                } else {
                    dst = (uint8_t*) t->data + r * cols * stride;
                }

                // populate current vec with row buffer
                quant_vec(dst, src, cols, t->id);
            }
            free(src);
        }
    }
}

// note that these are public
void tensor_lehmer(Tensor* t) {
    tensor_init(t, lehmer_float_cb, NULL);
}

void tensor_xavier(Tensor* t) {
    size_t rows = t->shape.dims[0];
    size_t cols = t->shape.dims[1];
    LehmerArgs args = {rows, cols};
    tensor_init(t, lehmer_xavier_cb, &args);
}

void tensor_muller(Tensor* t) {
    size_t rows = t->shape.dims[0];
    size_t cols = t->shape.dims[1];
    LehmerArgs args = {rows, cols};
    tensor_init(t, lehmer_muller_cb, &args);
}

/** @} */
