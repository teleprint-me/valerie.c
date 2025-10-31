/**
 * @file examples/linear/diff.c
 * @brief driver for experimenting with automated differentiation.
 */

#include <stdio.h>
#include <math.h>

// Function pointer type for a function R -> R
typedef double (*UnaryFn)(double);

// Example: our function f(x) = x^2
double square(double x) {
    return x * x;
}

double cube(double x) {
    return x * x * x;
}

double sine(double x) {
    return sin(x);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Numerical derivative: df/dx at x = a, with step h
double derivative(UnaryFn f, double a, double h) {
    // Guard: h must not be zero!
    if (h == 0.0) {
        // Handle error (simple print for now)
        fprintf(stderr, "Error: h must not be zero.\n");
        return NAN;
    }
    return (f(a + h) - f(a)) / h;
}

double derivative_central(UnaryFn f, double a, double h) {
    return (f(a + h) - f(a - h)) / (2 * h);
}

int main(void) {
    double a = 2.0;  // input
    double h = 0.01;  // step size

    // note that this is just analytic differentiation. not automatic.
    char* labels[] = {"square", "cube", "sine", "sigmoid"};
    UnaryFn callbacks[] = {square, cube, sine, sigmoid};
    size_t count = sizeof(callbacks) / sizeof(UnaryFn);
    for (size_t i = 0; i < count; i++) {
        char* label = labels[i];
        UnaryFn cb = callbacks[i];

        // standard
        double dy = derivative(cb, a, h);
        printf("Standard (%s): dy = %.5f, dx = %.5f, step = %.5f\n", label, dy, a, h);

        // central
        dy = derivative_central(cb, a, h);
        printf("Central (%s): dy = %.5f, dx = %.5f, step = %.5f\n", label, dy, a, h);
    }

    return 0;
}
