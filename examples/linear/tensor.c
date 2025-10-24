/**
 * @file bin/tensor.c
 * @brief Driver for experimental Tensor API.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define Q8_BLOCK_SIZE 8
#include "linear/lehmer.h"
#include "linear/type.h"
#include "linear/quant.h"

typedef enum ShapeId {
    SHAPE_VEC = 1,  // 1D (len, 0)
    SHAPE_MAT = 2,  // 2D (rows, cols)
} ShapeId;

typedef struct Shape {
    size_t dims[2];  // Up to 2D: [rows, cols] or [len, 0]
    ShapeId id;  // 0 for vector, 1 for matrix
} Shape;

typedef struct Tensor {
    void* data;  // Owned, type per TypeId
    float* workspace;  // Optional, for (de)quant, owned
    Shape shape;
    TypeId id;
} Tensor;

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

/**
 * Matrix multiplication
 */

void matmul(Tensor* y, Tensor* W, Tensor* x) {
    assert(y && W && x);
    assert(y->shape.id == SHAPE_VEC);
    assert(W->shape.id == SHAPE_MAT);
    assert(x->shape.id == SHAPE_VEC);

    size_t y_cols = y->shape.dims[0];  // out dim
    size_t W_rows = W->shape.dims[0];
    size_t W_cols = W->shape.dims[1];
    size_t x_cols = x->shape.dims[0];  // in dim
    size_t W_stride = type_size(W->id);
    assert(W_rows == y_cols);  // match out
    assert(W_cols == x_cols);  // match in
    assert(W_stride > 0);  // at least 1 byte

    // Convert input to float
    float* xf = calloc(x->shape.dims[0], type_size(x->id));
    dequant_vec(xf, x->data, x->shape.dims[0], x->id);

    // Temporary buffer for each row of W
    float* wf = malloc(W_cols * sizeof(float));
    float* yf = malloc(y_cols * sizeof(float));

    for (size_t r = 0; r < W_rows; r++) {
        // Compute source row pointer
        const void* wsrc;
        if (W->id == TYPE_Q8) {
            wsrc = (const uint8_t*) W->data + r * W_stride;
        } else {
            wsrc = (const uint8_t*) W->data + r * W_cols * W_stride;
        }
        dequant_vec(wf, wsrc, W_cols, W->id);

        // Compute dot product
        float sum = 0.0f;
        for (size_t c = 0; c < W_cols; c++) {
            sum += wf[c] * xf[c];
        }

        yf[r] = sum;
    }

    // Write result
    quant_vec(y->data, yf, y_cols, y->id);

    // Clean up
    free(wf);
    free(xf);
    free(yf);
}

/** @} */

/**
 * Tensor log
 */

void tensor_print(const Tensor* t) {
    printf("Tensor [%s] shape(", type_name(t->id));
    for (size_t i = 0; i < (t->shape.id == SHAPE_MAT ? 2 : 1); ++i) {
        printf("%zu%s", t->shape.dims[i], (i ? "" : ", "));
    }
    printf("):\n");
    switch (t->shape.id) {
        case SHAPE_VEC: {
            size_t len = t->shape.dims[0];
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
            size_t rows = t->shape.dims[0];
            size_t cols = t->shape.dims[1];
            size_t stride = type_size(t->id);
            float* dst = calloc(cols, sizeof(float));
            for (size_t r = 0; r < rows; ++r) {
                const void* src;
                if (t->id == TYPE_Q8) {
                    src = (const uint8_t*) t->data + r * stride;
                } else {
                    src = (const uint8_t*) t->data + r * cols * stride;
                }
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

int main(void) {
    lehmer_init(42);

    int rows = 4;
    int cols = 8;

    // Create tensors for y = W * x
    Tensor y = tensor_new(shape_vec(rows), TYPE_F32);
    Tensor x = tensor_new(shape_vec(cols), TYPE_F32);
    Tensor W = tensor_new(shape_mat(rows, cols), TYPE_F32);

    tensor_lehmer(&x);
    tensor_xavier(&W);
    matmul(&y, &W, &x);

    printf("x -> ");
    tensor_print(&x);
    printf("W -> ");
    tensor_print(&W);
    printf("y -> ");
    tensor_print(&y);

    tensor_free(&y);
    tensor_free(&x);
    tensor_free(&W);
    return 0;
}
