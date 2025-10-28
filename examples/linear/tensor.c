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
 * @brief Matrix-vector multiply with quantization-aware dequantization.
 *
 * y = W @ x
 *
 * W: (rows, cols) matrix (any type)
 * x: (cols,) vector (any type)
 * y: (rows,) output vector (float only)
 * @ref https://understandinglinearalgebra.org/sec-matrices-lin-combs.html
 * @note The only way to sanely resolve the compute buffers is a graph.
 */
void matmul(Tensor* y, Tensor* W, Tensor* x) {
    // Assert valid tensors
    assert(y && W && x);
    // Assert output type (W and x may be any type)
    assert(y->id == TYPE_F32);  // output must be float
    // Assert shape types
    assert(tensor_is_vec(y));
    assert(tensor_is_mat(W));
    assert(tensor_is_vec(x));
    // Assert dims match y (r,) = W (r, c) @ x (c,)
    assert(tensor_cols_match(x, W));  // match input
    assert(tensor_cols_match_rows(y, W));  // match output
    // Extract input dimensions
    const size_t W_rows = tensor_rows(W);
    const size_t W_cols = tensor_cols(W);
    const size_t x_cols = tensor_cols(x);  // in dim

    // Alias output buffer
    float* yf = (float*) y->data;

    // Convert input to float
    float* xf = calloc(x_cols, sizeof(float));  // scratch buffer
    dequant_vec(xf, x->data, x_cols, x->id);

#pragma omp parallel for
    for (size_t r = 0; r < W_rows; r++) {
        // Compute source row pointer
        float* wdst = calloc(W_cols, sizeof(float));  // scratch buffer
        const void* wsrc = tensor_view_row(W, r);
        dequant_vec(wdst, wsrc, W_cols, W->id);

        // Compute dot product
        float sum = 0.0f;
        for (size_t c = 0; c < W_cols; c++) {
            sum += wdst[c] * xf[c];
        }

        yf[r] = sum;
        free(wdst);
    }

    free(xf);
}

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
