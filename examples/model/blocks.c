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
void rmsnorm(float* y, float* w, float* x, size_t len) {
    // calculate sum of squares
    float sos = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sos += x[i] * x[i];
    }
    sos = 1.0f / sqrtf((sos / len) + 1e-6f);

    // normalize and scale
    for (size_t i = 0; i < len; i++) {
        y[i] = w[i] * (sos * x[i]);
    }
}

void matmul(float* y, Tensor* W, Tensor* x, size_t len) {
    assert(y && W && x);
    assert(tensor_is_mat(W));
    assert(tensor_is_vec(x));
    assert(tensor_cols_match(W, x));

    const size_t W_rows = tensor_rows(W);
    assert(W_rows == len && "y (r,) != W (r, c) @ x (c,)");  // match out

    // Convert input to float
    const size_t x_cols = tensor_cols(x);  // in dim
    float xf[x_cols];  // scratch buffer
    dequant_vec(xf, x->data, x_cols, x->id);

    const size_t W_cols = tensor_cols(W);
    for (size_t r = 0; r < W_rows; r++) {
        // Compute source row pointer
        float wdst[W_cols];  // scratch buffer
        const void* wsrc = tensor_mat_row(W, r);
        dequant_vec(wdst, wsrc, W_cols, W->id);

        // Compute dot product
        float sum = 0.0f;
        for (size_t c = 0; c < W_cols; c++) {
            sum += wdst[c] * xf[c];
        }

        y[r] = sum;
    }
}

// @ref https://arxiv.org/abs/2104.09864
void rotary(float* x, Rotary* rope, size_t pos, size_t len) {
    // Pre-computed rope frequencies
    const Tensor* cos = &rope->cos;
    const Tensor* sin = &rope->sin;
    // Rotary must always be TYPE_F32
    assert(cos->id == TYPE_F32);
    assert(sin->id == TYPE_F32);
    // Rotary must be shape (seq_len, head_dim / 2)
    assert(tensor_cols_match(cos, sin));
    assert(tensor_rows_match(cos, sin));
    // Column space is always half-dim
    size_t half_dim = tensor_cols(cos);
    assert(half_dim == len / 2); // half_dim == head_dim / 2

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
void softmax(float* x, size_t len) {
    float max_score = x[0];
    for (size_t i = 1; i < len; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        x[i] = expf(x[i] - max_score);
        sum += x[i];
    }

    for (size_t i = 0; i < len; i++) {
        x[i] /= sum;
    }
}

// @ref https://arxiv.org/abs/1512.03385
void residual(float* dst, float* src, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] += src[i];
    }
}

int main(void) {
    return 0;
}
