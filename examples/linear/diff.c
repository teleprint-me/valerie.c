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

    double dx = derivative(square, 2.0, 0.01);
    printf("Derivative (dx): a = %.5f, h = %.5f, dy = %.5f\n", a, h, dx);

    double dy = derivative_central(square, 2.0, 0.01);
    printf("Derivative (dy): a = %.5f, h = %.5f, dy = %.5f\n", a, h, dy);

    return 0;
}
