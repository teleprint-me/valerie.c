/**
 * @file model/activation.c
 * @brief Canonical activation functions and their derivatives for neural nets.
 *
 * Forward and backward (derivative) functions.
 * - x: pre-activation
 * - a: post-activation
 *
 * @note Modifying this math will corrupt model behavior.
 *       Make sure activation and element-wise multiplication are preserved.
 */

#include <math.h>

#include "model/activation.h"

/**
 * @section Forward propagation
 * @{
 */

// σ(x) = 1 / 1 + exp(-x)
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Rectified Linear Unit (ReLU)
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// x∗σ(x), where σ(x) is the logistic sigmoid.
float silu(float x) {
    return x * sigmoid(x);  // β is a constant of 1
}

// Swish(β)(x) = x / (1 + e^(-βx))
float swish(float x, float beta) {
    return x / (1.0f + expf(-beta * x));
}

// SwiGLU(x) = Swish(β)(W * x + b) ⊙ (V * x + c)
float swiglu(float a, float g, float beta) {
    return swish(a, beta) * g;
}

/** @} */

/**
 * @section Backward propagation
 * @{
 */

float sigmoid_prime(float x) {
    float a = sigmoid(x);
    return a * (1.0f - a);
}

float relu_prime(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float silu_prime(float x) {
    float a = sigmoid(x);
    return a * (1 + (x * (1.0f - a)));
}

float swish_prime(float x, float beta) {
    float a = sigmoid(x * beta);
    return a + x * beta * a * (1.0f - a);
}

// SwiGLU: Derivative w.r.t a (first input)
float swiglu_prime_a(float a, float g, float beta) {
    return swish_prime(a, beta) * g;
}

// SwiGLU: Derivative w.r.t g (second input)
float swiglu_prime_g(float a, float beta) {
    return swish(a, beta);
}

/** @} */
