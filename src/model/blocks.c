/**
 * @file blocks.c
 */

#include <assert.h>
#include <math.h>

#include "model/blocks.h"

void one_hot(float* x, size_t label, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (label == i) {
            x[i] = 1.0f;
        } else {
            x[i] = 0.0f;
        }
    }
}

// y_pred: predicted probabilities (softmax output), shape (n,)
// y_true: target one-hot vector, shape (n,)
// n: number of classes
float cross_entropy(const float* y_pred, const float* y_true, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (y_true[i] == 1.0f) {
            return -logf(fmaxf(y_pred[i], 1e-8f));
        }
    }
    return 0.0f;  // fallback if not one-hot
}

// @ref https://arxiv.org/abs/1910.07467
void rmsnorm(float* y, float* w, float* x, size_t n) {
    // Avoid division by 0
    assert(n > 0 && "Division by zero!");

    // calculate sum of squares
    float sos = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sos += x[i] * x[i];
    }
    sos = 1.0f / sqrtf((sos / n) + 1e-6f);

    // normalize and scale
    for (size_t i = 0; i < n; i++) {
        y[i] = w[i] * (sos * x[i]);
    }
}

// @ref https://arxiv.org/abs/2104.09864
void rotary(float* x, int pos, size_t head_dim, const float* cos, const float* sin) {
    size_t half_dim = head_dim / 2;

    const float* cos_t = cos + pos * half_dim;
    const float* sin_t = sin + pos * half_dim;

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
void softmax(float* x, size_t n) {
    float max_score = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_score);
        sum += x[i];
    }

    for (size_t i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// @ref https://arxiv.org/abs/1512.03385
void residual(float* y, float* x, size_t n) {
    assert(n > 0);

    for (size_t i = 0; i < n; i++) {
        y[i] += x[i];
    }
}
