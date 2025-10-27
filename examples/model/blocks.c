/**
 * @file examples/model/blocks.c
 * @brief Driver to exeriment with Tensors in model blocks.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "linear/quant.h"
#include "linear/tensor.h"
#include "model/valerie.h"

// @ref https://arxiv.org/abs/1910.07467
void rmsnorm(Tensor* y, Tensor* w, Tensor* x) {
    // Assert each tensor has a float buffer
    assert(y->buffer);
    assert(w->buffer);
    assert(x->buffer);

    // Assert shapes are vectors
    assert(tensor_is_vec(y));
    assert(tensor_is_vec(w));
    assert(tensor_is_vec(x));

    // Assert shapes match
    size_t n = y->shape.dims[0];
    assert(n == w->shape.dims[0]);
    assert(n == x->shape.dims[0]);

    // Dequantize to buffer
    float* yf = tensor_vec_dequant(y);  // alias to buffer
    float* wf = tensor_vec_dequant(w);
    float* xf = tensor_vec_dequant(x);

    // calculate sum of squares
    float sos = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sos += xf[i] * xf[i];
    }
    sos = 1.0f / sqrtf((sos / n) + 1e-6f);

    // normalize and scale
    for (size_t i = 0; i < n; i++) {
        yf[i] = wf[i] * (sos * xf[i]);
    }

    // Quantize output
    tensor_vec_quant(y);
}

// @ref https://arxiv.org/abs/2104.09864
void rotary(float* x, Rotary* rope, size_t pos) {
    // Pre-computed rope frequencies
    const Tensor* cos = &rope->cos;
    const Tensor* sin = &rope->sin;
    // Rotary must always be TYPE_F32
    assert(cos->id == TYPE_F32);
    assert(sin->id == TYPE_F32);
    // Rotary must be same shape (seq_len, head_dim / 2)
    assert(cos->shape.dims[0] == sin->shape.dims[0]);
    assert(cos->shape.dims[1] == sin->shape.dims[1]);
    // Column space is always half-dim
    size_t half_dim = sin->shape.dims[1];

    const float* cos_t = (float*) cos->data + pos * half_dim;
    const float* sin_t = (float*) sin->data + pos * half_dim;

    for (size_t i = 0; i < half_dim; i++) {
        float c = cos_t[i];
        float s = sin_t[i];

        float real = x[i];
        float imag = x[i + half_dim];

        x[i] = real * c - imag * s;
        x[i + half_dim] = real * s + imag * c;
    }
}

// @ref https://deeplearningbook.org/contents/mlp.html#pf11
void softmax(Tensor* x, size_t pos) {
    assert(x->buffer);
    assert(x->shape.id == SHAPE_VEC);
    size_t n = x->shape.dims[0];
    assert(n == pos);

    dequant_vec(x->buffer, x->data, n, x->id);
    float* xf = x->buffer;

    float max_score = xf[0];
    for (size_t i = 1; i < n; i++) {
        if (xf[i] > max_score) {
            max_score = xf[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        xf[i] = expf(xf[i] - max_score);
        sum += xf[i];
    }

    for (size_t i = 0; i < n; i++) {
        xf[i] /= sum;
    }

    quant_vec(x->data, xf, n, x->id);
}

// @ref https://arxiv.org/abs/1512.03385
void residual(Tensor* y, Tensor* x) {
    assert(y->buffer);
    assert(x->buffer);

    assert(y->shape.id == SHAPE_VEC);
    assert(x->shape.id == SHAPE_VEC);

    size_t n = y->shape.dims[0];
    assert(n == x->shape.dims[0]);

    float* yf = tensor_vec_dequant(y);
    float* xf = tensor_vec_dequant(x);

    for (size_t i = 0; i < n; i++) {
        yf[i] += xf[i];
    }

    tensor_vec_quant(y);
}

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
        const void* wsrc = tensor_mat_row(W, r);
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

int main(void) {
    return 0;
}
