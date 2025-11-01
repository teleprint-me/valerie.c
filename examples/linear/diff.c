/**
 * @file examples/linear/diff.c
 * @brief driver for experimenting with automated differentiation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Function pointer type for a function R -> R
typedef float (*UnaryFn)(float);

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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

void activate(float* a, float* x, size_t len) {
    for (size_t i = 0; i < len; i++) {
        a[i] = sigmoid(x[i]);
    }
}

// Apply row-major matrix multiplication (y = Wx + b)
void matmul(float* y, float* W, float* x, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;  // summed row
        float* w = W + i * cols;  // current row
        for (size_t j = 0; j < cols; j++) {
            sum += w[j] * x[j];  // dot product
        }
        y[i] = sum;  // update output column
    }
}

void dmatmul(
    float* dW, float* dx, const float* dy, const float* W, const float* x, size_t rows, size_t cols
) {
    // dW: ∂L/∂W[i,j] = dy[i] * x[j]
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dW[i * cols + j] += dy[i] * x[j];
        }
    }
    // dx: ∂L/∂x[j] = sum_i (dy[i] * W[i,j])
    for (size_t j = 0; j < cols; ++j) {
        dx[j] = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            dx[j] += dy[i] * W[i * cols + j];
        }
    }
}

void log_vector(float* x, size_t len) {
    (void) x;
    (void) len;
}

void log_matrix(float* W, size_t rows, size_t cols) {
    (void) W;
    (void) rows;
    (void) cols;
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
    for (size_t j = 0; j < rows; j++) {
        target[j] = prng();
    }

    // forward pass (activate the inputs; ignore weights for simplicity)
    for (size_t j = 0; j < rows; j++) {
        y[j] = 0.0f;  // initialize output
        for (size_t i = 0; i < cols; i++) {
            a[i] = sigmoid(x[i]);  // store activation
            y[j] += W[j * cols + i] * a[i];  // compute output
        }
    }

    // compute error
    float loss = mse(y, target, rows);

    // print for sanity
    printf("Loss: %.5f\n", (double) loss);

    /** backward passes are composed of 2 steps */

    // Derivative of activation (per input)
    for (size_t i = 0; i < cols; i++) {
        dx[i] = derivative(sigmoid, x[i], h);
    }

    // Compute error (dy = dL/dy = y - target for MSE, dL/dy_j = 2*(y_j - t_j))
    for (size_t j = 0; j < rows; j++) {
        dy[j] = 2.0f * (y[j] - target[j]);
    }

    // Compute dL/dW (gradient w.r.t weights): dL/dW[j * cols + i] = dy[j] * a[i]
    for (size_t j = 0; j < rows; j++) {
        for (size_t i = 0; i < cols; i++) {
            dW[j * cols + i] = dy[j] * a[i];
        }
    }

    // Gradient step: W -= lr * dL/dW
    for (size_t j = 0; j < rows; j++) {
        for (size_t i = 0; i < cols; i++) {
            W[j * cols + i] -= lr * dW[j * cols + i];
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
