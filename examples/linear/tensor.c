/**
 * @file bin/tensor.c
 * @brief Driver for experimental Tensor API.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define Q8_BLOCK_SIZE 8
#include "linear/lehmer.h"
#include "linear/type.h"
#include "linear/quant.h"
#include "linear/tensor.h"

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
    float* xf = calloc(x_cols, type_size(x->id));
    dequant_vec(xf, x->data, x_cols, x->id);

    // Temporary buffer for each row of W
    float* wf = malloc(W_cols * sizeof(float));
    float* yf = malloc(y_cols * sizeof(float));

    for (size_t r = 0; r < W_rows; r++) {
        // Compute source row pointer
        const void* wsrc = tensor_row(W, r);
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
    tensor_log(&x);
    printf("W -> ");
    tensor_log(&W);
    printf("y -> ");
    tensor_log(&y);

    tensor_free(&y);
    tensor_free(&x);
    tensor_free(&W);
    return 0;
}
