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

    float h = 0.01;  // step size

    // initialize the input vector
    size_t x_len = 5;
    float* x = malloc(x_len * sizeof(float));
    for (size_t i = 0; i < 5; i++) {
        x[i] = prng();
    }

    // forward pass (activate the inputs; ignore weights for simplicity)
    float* a = malloc(x_len * sizeof(float));
    for (size_t i = 0; i < x_len; i++) {
        a[i] = sigmoid(x[i]);  // store the activation
    }

    /** backward passes are composed of 2 steps */

    // compute derivatives (this is numerically unstable at scale)
    float* dx = malloc(x_len * sizeof(float));
    for (size_t i = 0; i < x_len; i++) {
        dx[i] = derivative(sigmoid, x[i], h);
    }

    // update parameters
    for (size_t i = 0; i < x_len; i++) {
        float y = composite(x[i]);  // f(x) = f(g(x)) = sin(σ(x))
        float dy = derivative(composite, x[i], h);
        printf(
            "x[%zu] = %.5f, sigmoid = %.5f, d/dx = %.5f\n",
            i,
            (double) x[i],
            (double) y,
            (double) dy
        );
    }

    free(dx);
    free(a);
    free(x);
    return 0;
}
