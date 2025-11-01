/**
 * @file examples/linear/diff.c
 * @brief driver for experimenting with automated differentiation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Function pointer type for a function R -> R
typedef float (*UnaryFn)(float);

// Example: our function f(x) = x^2
float square(float x) {
    return x * x;
}

float cube(float x) {
    return x * x * x;
}

float sine(float x) {
    return sinf(x);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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
    return (f(a + h) - f(a)) / h;
}

float derivative_central(UnaryFn f, float a, float h) {
    return (f(a + h) - f(a - h)) / (2 * h);
}

int main(void) {
    srand(73);  // the best number ever

    float h = 0.01;  // step size
    // input
    size_t x_len = 5;
    float* x = malloc(x_len * sizeof(float));
    for (size_t i = 0; i < 5; i++) {
        x[i] = prng();
    }

    for (size_t i = 0; i < x_len; i++) {
        // standard
        float dy = derivative(sigmoid, x[i], h);
        printf(
            "sigmoid (%zu): dy = %.5f, dx = %.5f, step = %.5f\n",
            i,
            (double) dy,
            (double) x[i],
            (double) h
        );
    }

    free(x);
    return 0;
}
