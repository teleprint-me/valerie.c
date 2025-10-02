/**
 * @file model/blocks.c
 */

#include <assert.h>
#include <math.h>

#include "model/blocks.h"

void one_hot(float* x, unsigned label, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
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
float cross_entropy(const float* y_pred, const float* y_true, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        if (y_true[i] == 1.0f) {
            return -logf(fmaxf(y_pred[i], 1e-8f));
        }
    }
    return 0.0f;  // fallback if not one-hot
}

// Applied to feed-forward method.
void rmsnorm(float* y, float* w, float* x, unsigned n) {
    // Avoid division by 0
    assert(n > 0 && "Division by zero!");

    // calculate sum of squares
    float sos = 0.0f;
    for (unsigned i = 0; i < n; i++) {
        sos += x[i] * x[i];
    }
    sos = 1.0f / sqrtf((sos / n) + 1e-6f);

    // normalize and scale
    for (unsigned i = 0; i < n; i++) {
        y[i] = w[i] * (sos * x[i]);
    }
}

/// @todo Add precomputed rotary cache.
/// Applied to feed-forward method.
void rotary(float* x, int pos, unsigned head_dim) {
    unsigned half_dim = head_dim / 2;

    for (unsigned i = 0; i < half_dim; i++) {
        float angle = pos * powf(1e6f, -(float) i / half_dim);
        float cos_a = cosf(angle), sin_a = sinf(angle);

        float real = x[i];
        float imag = x[i + half_dim];

        x[i] = real * cos_a - imag * sin_a;
        x[i + half_dim] = real * sin_a + imag * cos_a;
    }
}

// Applied to multi-head self-attention
void softmax(float* x, unsigned n) {
    float max_score = x[0];
    for (unsigned i = 1; i < n; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (unsigned i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_score);
        sum += x[i];
    }

    for (unsigned i = 0; i < n; i++) {
        x[i] /= sum;
    }
}
