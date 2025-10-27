/**
 * @file tensor.c
 * @brief Minimal tensor abstraction for 1D/2D numeric data.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "linear/lehmer.h"
#include "linear/quant.h"
#include "linear/tensor.h"

/**
 * Tensor shape
 * @{
 */

size_t shape_count(const Shape* s) {
    size_t product = 1;
    for (size_t i = 0; i < s->id; ++i) {
        product *= s->dims[i];
    }
    return product;
}

Shape shape_vec(size_t len) {
    return (Shape) {{len, 0}, SHAPE_VEC};
}

Shape shape_mat(size_t rows, size_t cols) {
    return (Shape) {{rows, cols}, SHAPE_MAT};
}

/** @} */

/**
 * Tensor dimensions
 */

bool tensor_is_vec(const Tensor* t) {
    return t->shape.id == SHAPE_VEC;
}

bool tensor_is_mat(const Tensor* t) {
    return t->shape.id == SHAPE_MAT;
}

size_t tensor_cols(const Tensor* t) {
    if (tensor_is_vec(t)) {
        return t->shape.dims[0];
    }
    assert(tensor_is_mat(t) && "tensor_cols: unknown tensor type");
    return t->shape.dims[1];
}

size_t tensor_rows(const Tensor* t) {
    assert(tensor_is_mat(t) && "tensor_rows: not a matrix");
    return t->shape.dims[0];
}

bool tensor_cols_match(const Tensor* a, const Tensor* b) {
    return tensor_cols(a) == tensor_cols(b);
}

bool tensor_cols_match_rows(const Tensor* a, const Tensor* b) {
    return tensor_cols(a) == tensor_rows(b);
}

bool tensor_rows_match(const Tensor* a, const Tensor* b) {
    return tensor_rows(a) == tensor_rows(b);
}

/** @} */

/**
 * Tensor life-cycle
 * @{
 */

/**
 * private methods
 */

void tensor_assert_q8(size_t cols) {
    if (cols < Q8_BLOCK_SIZE) {
        fprintf(
            stderr,
            "tensor_new_q8: SHAPE_VEC dims[0]=%zu is less than Q8_BLOCK_SIZE=%d\n",
            cols,
            Q8_BLOCK_SIZE
        );
        abort();
    }
}

void tensor_new_q8(Tensor* t) {
    size_t cols = tensor_cols(t);
    tensor_assert_q8(cols);

    // Q8 is an array of quant8_t for each row
    switch (t->shape.id) {
        case SHAPE_VEC: {
            // 1D: single quant8_t
            quant8_t* q = malloc(sizeof(quant8_t));
            *q = q8_vec_new(cols);
            t->data = q;
            break;
        }
        case SHAPE_MAT: {
            // 2D: array of quant8_t, one per row
            size_t rows = tensor_rows(t);
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
                q8_mat_free((quant8_t*) t->data, tensor_rows(t));
                break;
        }
    }
}

void tensor_new_data(Tensor* t) {
    size_t stride = type_size(t->id);
    size_t len = shape_count(&t->shape);
    t->data = malloc(len * stride);
}

/**
 * public methods
 */

Tensor tensor_new(Shape shape, TypeId id) {
    Tensor t = {0};
    t.shape = shape;
    t.id = id;
    if (id == TYPE_Q8) {
        tensor_new_q8(&t);
    } else {
        tensor_new_data(&t);
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (t && t->data) {
        if (t->id == TYPE_Q8) {
            tensor_free_q8(t);
        } else {
            free(t->data);
        }
        t->data = NULL;
    }
}

/** @} */

/**
 * Tensor quantization
 */

void tensor_quant_vec(Tensor* dst, float* src, size_t len) {
    assert(tensor_is_vec(dst));
    assert(tensor_cols(dst) == len);
    if (dst->id == TYPE_F32) {
        return;  // skip float
    }
    quant_vec(dst->data, src, len, dst->id);
}

void tensor_dequant_vec(float* dst, const Tensor* src, size_t len) {
    assert(tensor_is_vec(src));
    assert(tensor_cols(src) == len);
    if (src->id == TYPE_F32) {
        memcpy(dst, src->data, len * sizeof(float));
    } else {
        dequant_vec(dst, src->data, len, src->id);
    }
}

/** @} */

/**
 * Tensor view
 */

void* tensor_view(const Tensor* t, size_t offset) {
    size_t stride = type_size(t->id);
    if (t->id == TYPE_Q8 && t->shape.id == SHAPE_VEC) {
        return (uint8_t*) t->data;  // single container with a vector
    }
    return (uint8_t*) t->data + offset * stride;
}

void* tensor_view_row(const Tensor* t, size_t row) {
    assert(tensor_is_mat(t));
    // Q8: one quant8_t per row
    if (t->id == TYPE_Q8) {
        return (uint8_t*) tensor_view(t, row);
    }
    // Dense: row-major, one element per col
    return (uint8_t*) tensor_view(t, row * tensor_cols(t));
}

/** @} */

/**
 * Tensor fill
 * @{
 */

void tensor_fill(Tensor* t, float value) {
    switch (t->shape.id) {
        case SHAPE_VEC: {
            size_t len = tensor_cols(t);
            ;
            float* src = malloc(len * sizeof(float));
            for (size_t i = 0; i < len; i++) {
                src[i] = value;
            }

            quant_vec(t->data, src, len, t->id);

            free(src);
            break;
        }
        case SHAPE_MAT: {
            size_t rows = tensor_rows(t);
            size_t cols = tensor_cols(t);
            float* src = malloc(cols * sizeof(float));
            for (size_t i = 0; i < cols; i++) {
                src[i] = value;
            }

            for (size_t r = 0; r < rows; r++) {
                void* dst = tensor_view_row(t, r);
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
            size_t len = tensor_cols(t);
            ;
            float* src = malloc(len * sizeof(float));
            for (size_t i = 0; i < len; i++) {
                src[i] = prng(args);
            }
            quant_vec(t->data, src, len, t->id);
            free(src);
            break;
        }
        case SHAPE_MAT: {
            size_t rows = tensor_rows(t);
            size_t cols = tensor_cols(t);
            float* src = malloc(cols * sizeof(float));
            for (size_t r = 0; r < rows; r++) {
                // fill row buffer
                for (size_t c = 0; c < cols; c++) {
                    src[c] = prng(args);
                }

                void* dst = tensor_view_row(t, r);
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
    size_t rows = tensor_rows(t);
    size_t cols = tensor_cols(t);
    LehmerArgs args = {rows, cols};
    tensor_init(t, lehmer_xavier_cb, &args);
}

void tensor_muller(Tensor* t) {
    size_t rows = tensor_rows(t);
    size_t cols = tensor_cols(t);
    LehmerArgs args = {rows, cols};
    tensor_init(t, lehmer_muller_cb, &args);
}

/** @} */

/**
 * Tensor log
 */

void tensor_log(const Tensor* t) {
    printf("Tensor [%s] shape(", type_name(t->id));
    for (size_t i = 0; i < (tensor_is_mat(t) ? 2 : 1); ++i) {
        printf("%zu%s", t->shape.dims[i], (i ? "" : ", "));
    }
    printf("):\n");
    switch (t->shape.id) {
        case SHAPE_VEC: {
            size_t len = tensor_cols(t);
            ;
            float* x = calloc(len, sizeof(float));
            dequant_vec(x, t->data, len, t->id);
            printf("[");
            for (size_t i = 0; i < len; ++i) {
                printf(" % .5f", (double) x[i]);
            }
            printf(" ]\n");
            free(x);
            break;
        }
        case SHAPE_MAT: {
            size_t rows = tensor_rows(t);
            size_t cols = tensor_cols(t);
            float* dst = calloc(cols, sizeof(float));
            for (size_t r = 0; r < rows; ++r) {
                const void* src = tensor_view_row(t, r);
                dequant_vec(dst, src, cols, t->id);

                printf("[");
                for (size_t c = 0; c < cols; ++c) {
                    printf(" % .5f", (double) dst[c]);
                }
                printf(" ]\n");
            }
            free(dst);
            break;
        }
    }
}

/** @} */
