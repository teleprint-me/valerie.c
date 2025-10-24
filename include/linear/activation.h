/**
 * @file model/activation.h
 * @brief Canonical activation functions and their derivatives for neural nets.
 *
 * Forward and backward (derivative) functions.
 * - x: pre-activation
 * - a: post-activation
 *
 * @note Modifying this math will corrupt model behavior.
 *       Make sure activation and element-wise multiplication are preserved.
 *
 * SwiGLU(x) = Swish(β)(W * x + b) ⊙ (V * x + c)
 * A variant of the GLU activation function using the Swish activation function.
 *
 * @param x (batch × dim) - the input tensor
 * @param W (activation) - weight tensor for the "activation" branch
 * @param V (gate) - weight tensor for the "gate" branch
 * @param b (activation) - bias tensor for the "activation" branch
 * @param c (gate) - bias tensor for the "gate" branch
 * @param beta (scalar) - a scaling factor for the Swish activation function
 * @return SwiGLU(x)
 *
 * The SwiGLU activation function is a combination of two linear branches, one for the activation
 * and one for the gate, multiplied element-wise (⊙). The activation branch uses the Swish
 * activation function, which is defined as Swish(β)(x) = x / (1 + e^(-βx)). The gate branch uses
 * the standard sigmoid activation function (σ(x)). When β = 1, the Swish activation function is
 * equivalent to the Sigmoid Linear Unit (SiLU) activation function. When β = 0, the Swish
 * activation function turns into a scaled linear function. When β → ∞, the Swish activation
 * function approaches the ReLU activation function.
 *
 * # Mathematical Reference:
 * - SwiGLU(x) = Swish(β)(W * x + b) ⊙ (V * x + c)
 * - Swish(β)(x) = x / (1 + e^(-βx))
 * - Derivative: d/dx Swish(β)(x) = σ(βx) + xβσ(βx)(1 - σ(βx))
 *   where σ(z) = 1 / (1 + exp(-z))
 * - SwiGLU partial derivatives:
 *     - d/d(a): Swish'(a, β) * g
 *     - d/d(g): Swish(a, β)
 *
 * References:
 * - https://en.wikipedia.org/wiki/Activation_function
 * - https://en.wikipedia.org/wiki/Sigmoid_function
 * - https://en.wikipedia.org/wiki/Rectified_linear_unit
 * - https://en.wikipedia.org/wiki/Swish_function
 * - https://en.wikipedia.org/wiki/Gating_mechanism
 * - https://arxiv.org/pdf/2109.14545
 * - https://arxiv.org/abs/2002.05202
 * - https://arxiv.org/abs/2402.03804
 * - https://dublog.net/blog/all-the-activations/
 * - https://jcarlosroldan.com/post/348
 */

#ifndef MODEL_ACTIVATION_H
#define MODEL_ACTIVATION_H

/**
 * @name Forward Activations
 * @{
 */

float sigmoid(float x);  ///< Logistic sigmoid
float relu(float x);  ///< Rectified Linear Unit (ReLU)
float silu(float x);  ///< Sigmoid Linear Unit (SiLU), i.e., Swish(β=1)
float swish(float x, float beta);  ///< Swish activation (general β)
float swiglu(float a, float g, float beta);  ///< SwiGLU: Swish(a, β) * g

/** @} */

/**
 * @name Activation Derivatives (Backward)
 * @{
 */

float sigmoid_prime(float x);  ///< Derivative of sigmoid (wrt pre-activation x)
float relu_prime(float x);  ///< Derivative of relu
float silu_prime(float x);  ///< Derivative of silu (wrt pre-activation x)
float swish_prime(float x, float beta);  ///< Derivative of swish (wrt pre-activation x)
// SwiGLU derivatives: for each branch (β generalized!)
float swiglu_prime_a(float a, float g, float beta);  ///< d/d(a): Swish'(a, β) * g
float swiglu_prime_g(float a, float beta);  ///< d/d(g): Swish(a, β)

/** @} */

#endif  // MODEL_ACTIVATION_H
