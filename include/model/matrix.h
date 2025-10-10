/**
 * @file      model/matrix.h
 * @brief     Type-generic matrix operations (forward, backward, SGD) for ML.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * - Supports float, bfloat16, Q8, and custom types for all key ops.
 * - Minimal dependencies. Consistent, idiomatic, and easy to extend.
 */

#ifndef MODEL_MATRIX_H
#define MODEL_MATRIX_H

#include <stddef.h>  // size_t
#include <stdbool.h>
#include "core/type.h"  // TypeId, quant/dequant helpers
#include "core/lehmer.h"  // LehmerFn, LehmerArgs

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @section Matrix Allocation and Initialization
 * @{
 */

/**
 * @brief Allocate a row-major matrix (calloc) for a given type and shape.
 * @param rows Number of rows
 * @param cols Number of columns
 * @param id   Data type (see TypeId)
 * @return Pointer to buffer (must be freed by caller)
 */
void* mat_new(size_t rows, size_t cols, TypeId id);

/**
 * @brief Initialize matrix A (rows x cols) with values from a custom random distribution.
 * @param A         Matrix buffer (void*, length = rows * cols, type = id)
 * @param rows      Number of rows
 * @param cols      Number of columns
 * @param id        Data type (see TypeId)
 * @param lehmer_fn Random number callback (see core/lehmer.h)
 * @param lehmer_args Pointer to optional callback args (may be NULL)
 */
void mat_init(
    void* A, size_t rows, size_t cols, TypeId id, LehmerFn lehmer_fn, void* lehmer_args
);

/** @brief Uniform random initializer (Lehmer) */
void mat_lehmer(void* A, size_t rows, size_t cols, TypeId id);
/** @brief Xavier/Glorot uniform initializer */
void mat_xavier(void* A, size_t rows, size_t cols, TypeId id);
/** @brief Xavier normal (Box-Muller) initializer */
void mat_muller(void* A, size_t rows, size_t cols, TypeId id);

/** @} */

/**
 * @section Matrix Math
 */

/**
 * @brief Matrix-vector multiply: y = W x
 * @param[out] y   Output vector (float*, length = rows)
 * @param[in]  W   Row-major matrix (void*, rows x cols, type = id)
 * @param[in]  x   Input vector (void*, length = cols, type = id)
 * @param[in]  rows Number of rows (output length)
 * @param[in]  cols Number of columns (input length)
 * @param[in]  id   Data type for W and x
 */
void mat_mul(float* y, const void* W, const void* x, size_t rows, size_t cols, TypeId id);

/** @} */

/**
 * @section Backward/Gradient Ops
 */

/**
 * @brief Compute weight gradients: dW = d_next ⊗ x^T (outer product)
 * @param[out] dW      Output gradient matrix (void*, rows x cols, type = id)
 * @param[in]  d_next  Delta vector from next layer (void*, length = rows, type = id)
 * @param[in]  x       Activation vector from previous layer (float*, length = cols)
 * @param[in]  rows    Number of output neurons (rows)
 * @param[in]  cols    Number of input neurons (cols)
 * @param[in]  id      Data type for dW and d_next
 */
void mat_dW(void* dW, const void* d_next, const float* x, size_t rows, size_t cols, TypeId id);

/**
 * @brief Backprop chain rule: dy = (W_next^T * d_next) ⊙ f'(z)
 * @param[out] dy      Output delta (void*, length = rows, type = id)
 * @param[in]  W_next  Weight matrix of next layer (void*, rows_next x rows, type = id)
 * @param[in]  d_next  Delta from next layer (void*, length = rows_next, type = id)
 * @param[in]  z       Pre-activation values (float*, length = rows)
 * @param[in]  rows    Number of output neurons (current layer)
 * @param[in]  rows_next Number of output neurons (next layer)
 * @param[in]  id      Data type for all quantized buffers
 */
void mat_chain(
    void* dy,
    const void* W_next,
    const void* d_next,
    const float* z,
    size_t rows,
    size_t rows_next,
    TypeId id
);

/** @} */

/**
 * @section Optimizer Ops
 */

/**
 * @brief Apply SGD update to weights (with optional L2 and momentum)
 *
 * W:      weight matrix (void*, type = id_W)
 * dW:     gradient matrix (void*, type = id_dvW)
 * vW:     velocity buffer (void*, same type as dW, may be NULL if no momentum)
 *
 * - Supports all types (Q8, BF16, float).
 * - Call with vW = NULL, mu = 0.0f for vanilla SGD.
 * - For Nesterov, set nesterov = true.
 *
 * @param W        Weight matrix to update (void*, shape rows x cols, type id_W)
 * @param dW       Gradient buffer (void*, shape rows x cols, type id_dvW)
 * @param vW       Velocity buffer (void*, shape rows x cols, type id_dvW or float, may be NULL)
 * @param rows     Number of rows
 * @param cols     Number of columns
 * @param id_W     Data type of W
 * @param id_dvW   Data type of dW/vW
 * @param lr       Learning rate
 * @param lambda   L2 regularization factor
 * @param mu       Momentum (0.0f = none)
 * @param tau      Damping (usually 0.0f or 1.0f)
 * @param nesterov Enable Nesterov update (true/false)
 */
void mat_sgd(
    void* W,
    const void* dW,
    void* vW,
    size_t rows,
    size_t cols,
    TypeId id_W,
    TypeId id_dvW,
    float lr,
    float lambda,
    float mu,
    float tau,
    bool nesterov
);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // MODEL_MATRIX_H
