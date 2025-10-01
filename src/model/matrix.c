/**
 * @file      model/matrix.c
 * @brief     Type-generic matrix operations (forward, backward, SGD) for ML.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * - Supports float, bfloat16, Q8, and custom types for all key ops.
 * - Minimal dependencies. Consistent, idiomatic, and easy to extend.
 */

#include <stdlib.h>

#include "core/lehmer.h"
#include "core/type.h"

#include "model/activation.h"

/**
 * @section Matrix Allocation and Initialization
 * @{
 */

// Create a row-major matrix
void* mat_new(size_t rows, size_t cols, DataTypeId id) {
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    size_t stride = data_type_size(id);
    size_t n = rows * cols;
    return calloc(n, stride);
}

void mat_init(
    void* A, size_t rows, size_t cols, DataTypeId id, LehmerFn lehmer_fn, void* lehmer_args
) {
    assert(A);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);
    assert(lehmer_fn);

    // Calculate element size
    size_t stride = data_type_size(id);
    assert(stride > 0);

    // Calculate buffer length
    size_t n = rows * cols;

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        // xavier(fan_in, fan_out)
        float value = lehmer_fn(lehmer_args);  // must be thread_local
        void* dst = (uint8_t*) A + i * stride;
        quant_scalar(dst, value, id);
    }
}

void mat_lehmer(void* A, size_t rows, size_t cols, DataTypeId id) {
    assert(A);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    mat_init(A, rows, cols, id, lehmer_float_cb, NULL);
}

void mat_xavier(void* A, size_t rows, size_t cols, DataTypeId id) {
    assert(A);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    mat_init(A, rows, cols, id, lehmer_xavier_cb, &(LehmerArgs) {rows, cols});
}

void mat_muller(void* A, size_t rows, size_t cols, DataTypeId id) {
    assert(A);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    mat_init(A, rows, cols, id, lehmer_muller_cb, &(LehmerArgs) {rows, cols});
}

/** @} */

/**
 * @section Matrix Math
 * @{
 */

// Row-major matrix multiplication (y = Wx + b)
// bias is omitted because it's always 0
void mat_mul(float* y, const void* W, const void* x, size_t rows, size_t cols, DataTypeId id) {
    assert(y && W && x);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    size_t stride = data_type_size(id);
    assert(stride > 0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            // W[i][j]
            float W_dst;
            const void* W_src = (const uint8_t*) W + (i * cols + j) * stride;
            dequant_scalar(&W_dst, W_src, id);

            // x[j]
            float x_dst;
            const void* x_src = (const uint8_t*) x + j * stride;
            dequant_scalar(&x_dst, x_src, id);

            // dot product: W[i][j] * x[j]
            sum += W_dst * x_dst;
        }

        // y = W * x
        y[i] = sum;
    }
}

/** @} */

/**
 * @section Backward/Gradient Ops
 */

// dW = δ_next ⊗ x^T (outer product)
// dW, d_next must be of type id; x is always float*
void mat_dW(void* dW, const void* d_next, const float* x, size_t rows, size_t cols, DataTypeId id) {
    assert(dW && d_next && x);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    size_t stride = data_type_size(id);
    assert(stride > 0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        // Dequantize delta for this output (row)
        float d_next_i;
        const void* d_next_src = (const uint8_t*) d_next + i * stride;
        dequant_scalar(&d_next_i, d_next_src, id);

        for (size_t j = 0; j < cols; j++) {
            // Pre-compute outer product entry
            float temp = d_next_i * x[j];
            // Get the current derivative
            void* dW_ptr = (uint8_t*) dW + (i * cols + j) * stride;
            // Just update and set. Do not accumulate.
            quant_scalar(dW_ptr, temp, id);
        }
    }
}

// Backprop: dy = (W_next^T * d_next) ⊙ f'(z) (chain rule)
// dy, W_next, d_next must be of type id; z is float*
void mat_chain(
    void* dy,  // Output buffer (length = rows, type = id)
    const void* W_next,  // Weight matrix for next layer (rows_next x rows, type = id)
    const void* d_next,  // Delta vector from next layer (length = rows_next, type = id)
    const float* z,  // Pre-activation values (length = rows)
    size_t rows,  // Number of output neurons (current layer)
    size_t rows_next,  // Number of output neurons (next layer)
    DataTypeId id  // Data type for quantized buffers
) {
    assert(dy && W_next && d_next && z);
    assert(rows > 0 && rows_next > 0);
    assert(id < TYPE_COUNT);

    size_t stride = data_type_size(id);
    assert(stride > 0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < rows_next; j++) {
            // Dequantize d_next[j]
            float d_next_j;
            const void* d_next_src = (const uint8_t*) d_next + j * stride;
            dequant_scalar(&d_next_j, d_next_src, id);

            // Dequantize W_next[j, i] = W_next^T[i, j] (row-major)
            float W_next_T_ji;
            const void* W_next_src = (const uint8_t*) W_next + (j * rows + i) * stride;
            dequant_scalar(&W_next_T_ji, W_next_src, id);

            // Accumulate
            sum += W_next_T_ji * d_next_j;
        }

        // Apply pre-activation
        float temp = sum * silu_prime(z[i]);
        // Get the current ouput
        void* dy_ptr = (uint8_t*) dy + i * stride;
        // Update output
        quant_scalar(dy_ptr, temp, id);
    }
}

/** @} */

/**
 * @section Optimizer Ops
 */

// Apply SGD update to weights (type-agnostic, supports momentum)
void mat_sgd(
    void* W,  // [rows x cols], type id_W
    const void* dW,  // [rows x cols], type id_dvW
    void* vW,  // [rows x cols], type id_dvW or float, may be NULL
    size_t rows,
    size_t cols,
    DataTypeId id_W,
    DataTypeId id_dvW,
    float lr,
    float lambda,
    float mu,
    float tau,
    bool nesterov
) {
    assert(W && dW);
    assert(rows > 0 && cols > 0);
    assert(id_W < TYPE_COUNT && id_dvW < TYPE_COUNT);

    size_t stride_W = data_type_size(id_W);
    size_t stride_dvW = data_type_size(id_dvW);
    assert(stride_W > 0 && stride_dvW > 0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;

            // Get pointers
            void* W_ptr = (uint8_t*) W + idx * stride_W;
            const void* dW_ptr = (const uint8_t*) dW + idx * stride_dvW;

            // Dequantize to float for computation
            float w, g;
            dequant_scalar(&w, W_ptr, id_W);
            dequant_scalar(&g, dW_ptr, id_dvW);

            // L2 regularization (in float)
            if (lambda > 0.0f) {
                g += lambda * w;
            }

            // Momentum (optional)
            if (vW && mu > 0.0f) {
                float v;
                void* vW_ptr = (uint8_t*) vW + idx * stride_dvW;
                dequant_scalar(&v, vW_ptr, id_dvW);

                v = mu * v + (1.0f - tau) * g;

                if (nesterov) {
                    g += mu * v;  // Lookahead
                } else {
                    g = v;
                }

                // Store updated velocity
                quant_scalar(vW_ptr, v, id_dvW);
            }

            // Weight update
            w -= lr * g;
            quant_scalar(W_ptr, w, id_W);
        }
    }
}

/** @} */
