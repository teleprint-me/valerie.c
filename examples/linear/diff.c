/**
 * @file examples/linear/diff.c
 * @brief driver for experimenting with automated differentiation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Function pointer type for a function R -> R
typedef float (*UnaryFn)(float);

float sine(float x) {
    return sinf(x);
}

float cosine(float x) {
    return cosf(x);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// f(x) = f(g(x)) = sin(σ(x))
float composite(float x) {
    return sine(sigmoid(x));
}

// normalized input [0, 1]
float prng(void) {
    return (float) rand() / (float) RAND_MAX;
}

// mean squared error (loss fn)
float mse(float* y_pred, float* y_true, size_t len) {
    float loss = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss;
}

// Numerical derivative: df/dx at x = a, with step h
float derivative(UnaryFn f, float a, float h) {
    // Guard: h must not be zero!
    if (h == 0.0f) {
        // Handle error (simple print for now)
        fprintf(stderr, "Error: h must not be zero.\n");
        return NAN;
    }
    return (f(a + h) - f(a)) / h;  // first order

    // ignore this for now, but keep it around for reference.
    // return (f(a + h) - f(a - h)) / (2 * h);  // second order
}

int main(void) {
    srand(73);  // the best number ever

    // hyperparameters
    float h = 0.01;  // step size
    float lr = 0.1f;  // learning rate

    // dimensions
    size_t cols = 5;  // inputs
    size_t rows = 3;  // outputs

    // model parameters
    float* x = malloc(cols * sizeof(float));  // inputs
    float* W = malloc(cols * rows * sizeof(float));  // weights
    float* a = malloc(cols * sizeof(float));  // activations
    float* y = malloc(rows * sizeof(float));  // outputs
    float* dx = malloc(cols * sizeof(float));  // d(sigma)/dx
    float* dW = malloc(cols * rows * sizeof(float));  // dL/dW
    float* dy = malloc(rows * sizeof(float));  // dL/dy
    float* target = malloc(rows * sizeof(float));  // pseudo targets

    // feature vector
    for (size_t i = 0; i < cols; i++) {
        x[i] = prng();
    }

    // weights (i/o units)
    for (size_t i = 0; i < cols * rows; i++) {
        W[i] = prng();
    }

    // targets
    for (size_t i = 0; i < rows; i++) {
        target[i] = prng();
    }

    // forward pass (activate the inputs; ignore weights for simplicity)
    for (size_t i = 0; i < cols; i++) {
        a[i] = sigmoid(x[i]);  // store the activation
    }

    /** backward passes are composed of 2 steps */

    // compute derivatives (this is numerically unstable at scale)
    for (size_t i = 0; i < cols; i++) {
        dx[i] = derivative(sigmoid, x[i], h);
    }

    // Weight update: W[j * cols + i] -= lr * error * activation
    for (size_t j = 0; j < rows; j++) {
        for (size_t i = 0; i < cols; i++) {
            W[j * cols + i] -= lr * e[j] * a[i];
        }
    }

    // clean up
    free(x);
    free(W);
    free(a);
    free(y);
    free(dx);
    free(dW);
    free(dy);
    free(target);

    // exit
    return 0;
}
