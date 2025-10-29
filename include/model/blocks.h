/**
 * @file blocks.h
 * @brief Model blocks: forward-pass layer operations for Valerie transformer.
 * @copyright Copyright © 2025 Austin Berrio
 *
 * This header declares all core building blocks required for the forward pass
 * in transformer architectures: RMSNorm, matmul (quant-aware), rotary embedding,
 * softmax, residual connection, and high-level attention/feed-forward blocks.
 *
 * All operations are pure forward (inference-time) implementations.
 * All tensor arguments must be properly shaped and (unless otherwise noted)
 * of type TYPE_F32.
 */

#ifndef VALERIE_BLOCKS_H
#define VALERIE_BLOCKS_H

#include "linear/tensor.h"
#include "model/valerie.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Root-mean-square normalization (RMSNorm) for 1D vectors.
 * @ref https://arxiv.org/abs/1910.07467
 *
 * y = w * (x / sqrt(mean(x^2) + epsilon))
 *
 * @param y Output tensor (float vector)
 * @param w Weight tensor (float vector, per-feature)
 * @param x Input tensor (float vector)
 */
void rmsnorm(Tensor* y, Tensor* w, Tensor* x);

/**
 * @brief Matrix-vector multiply with quantization-aware dequantization.
 * @ref https://understandinglinearalgebra.org/sec-matrices-lin-combs.html
 *
 * y = W @ x
 *
 * @param y Output tensor (float vector, shape [rows])
 * @param W Weight matrix (any type, shape [rows, cols])
 * @param x Input vector (any type, shape [cols])
 */
void matmul(Tensor* y, Tensor* W, Tensor* x);

/**
 * @brief In-place rotary position embedding on a float buffer.
 * @ref https://arxiv.org/abs/2104.09864
 *
 * @param x    Buffer, shape [2*half_dim] (float*, real | imag parts)
 * @param rope Rotary embeddings (cos/sin tensors)
 * @param pos  Position (row) to use in rope tensors
 * @param len  Length of x (must be even)
 */
void rotary(float* x, Rotary* rope, size_t pos, size_t len);

/**
 * @brief In-place numerically stable softmax on a float buffer.
 * @ref https://deeplearningbook.org/contents/mlp.html#pf11
 *
 * @param x   Buffer to transform (float*)
 * @param len Length of buffer
 */
void softmax(float* x, size_t len);

/**
 * @brief In-place residual connection: dst += src (elementwise add).
 * @ref https://arxiv.org/abs/1512.03385
 *
 * @param dst Destination tensor (float vector)
 * @param src Source tensor (float vector)
 */
void residual(Tensor* dst, Tensor* src);

/**
 * @brief Single transformer attention block (forward pass, autoregressive).
 * @ref https://arxiv.org/abs/1706.03762
 *
 * All required shapes/dimensions are pulled from the Valerie and Layer structs.
 *
 * @param v   Model (Valerie*)
 * @param L   Layer (Layer*)
 * @param pos Sequence position (int)
 */
void forward_attn(Valerie* v, Layer* L, int pos);

/**
 * @brief Feed-forward network block (forward pass).
 * @ref https://deeplearningbook.org/contents/mlp.html#pf1
 *
 * @param v Model (Valerie*)
 * @param L Layer (Layer*)
 */
void forward_ffn(Valerie* v, Layer* L);

/**
 * @brief Full single-token forward pass (autoregressive).
 * Embedding lookup, layer stack, normalization, output projection.
 *
 * @param v   Model (Valerie*)
 * @param id  Token ID (int)
 * @param pos Position in sequence (int)
 * @return Pointer to output logits (float*)
 */
float* forward(Valerie* v, int id, int pos);

#ifdef __cplusplus
}
#endif

#endif  // VALERIE_BLOCKS_H
