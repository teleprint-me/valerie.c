/**
 * @file      model/matrix.c
 * @brief     Type-generic matrix operations (forward, backward, SGD) for ML.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * - Supports float, bfloat16, Q8, and custom types for all key ops.
 * - Minimal dependencies. Consistent, idiomatic, and easy to extend.
 */

#include <stdlib.h>

#include <assert.h>

#include "linear/activation.h"
#include "linear/lehmer.h"
#include "linear/type.h"
#include "model/matrix.h"

/**
 * @section Matrix Allocation and Initialization
 * @{
 */

void* vec_new(size_t len, TypeId id) {
    assert(len > 0 && id < TYPE_COUNT);

    switch (id) {
        case TYPE_Q8: {
            quant8_t* q8 = malloc(sizeof(quant8_t));
            if (!q8) {
                return NULL;
            }

            size_t blocks = (len + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;

            q8->q = calloc(len, sizeof(int8_t));
            q8->w = calloc(blocks, sizeof(uint8_t));

            return q8;
        }
        default: {
            return calloc(len, type_size(id));
        }
    }
}

void vec_free(void* x, TypeId id) {
    if (x) {
        switch (id) {
            case TYPE_Q8: {
                quant8_t* q8 = x;
                free(q8->q);
                free(q8->w);
                free(q8);
                break;
            }
            default:
                free(x);
                break;
        }
    }
}

// Create a row-major matrix
void* mat_new(size_t rows, size_t cols, TypeId id) {
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    return vec_new(rows * cols, id);
}

void mat_free(void* W, TypeId id) {
    if (W) {
        vec_free(W, id);
    }
}

/// @note This requires per-thread seeding using thread ids.
/// omp_get_thread_num()
/// Other possible solutions are thread-locking or chunking per thread.
/// For now, it's best to just operate linearly to keep complexity low.
void mat_init(void* W, size_t rows, size_t cols, TypeId id, LehmerFn lehmer_fn, void* lehmer_args) {
    assert(W && rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);
    assert(lehmer_fn);

    // Calculate buffer length
    size_t len = rows * cols;

    switch (id) {
        // Init block-wise
        case TYPE_Q8: {
            quant8_t* q8 = (quant8_t*) W;
            float* src = malloc(len * sizeof(float));
            for (size_t i = 0; i < len; i++) {
                src[i] = lehmer_fn(lehmer_args);
            }
            q8_encode(q8, src, len, Q8_BLOCK_SIZE);
            free(src);
            break;
        }
        // Init element-wise
        default: {
            size_t stride = type_size(id);
            assert(stride > 0);
            for (size_t i = 0; i < len; i++) {
                float value = lehmer_fn(lehmer_args);
                void* dst = (uint8_t*) W + i * stride;
                quant(dst, value, id);
            }
            break;
        }
    }
}

void mat_lehmer(void* W, size_t rows, size_t cols, TypeId id) {
    assert(W && rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    mat_init(W, rows, cols, id, lehmer_float_cb, NULL);
}

void mat_xavier(void* W, size_t rows, size_t cols, TypeId id) {
    assert(W && rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    mat_init(W, rows, cols, id, lehmer_xavier_cb, &(LehmerArgs) {rows, cols});
}

void mat_muller(void* W, size_t rows, size_t cols, TypeId id) {
    assert(W && rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    mat_init(W, rows, cols, id, lehmer_muller_cb, &(LehmerArgs) {rows, cols});
}

/** @} */

/**
 * @section Matrix Math
 * @{
 */

// Row-major matrix multiplication (y = Wx + b)
// bias is omitted because it's always 0
void mat_mul(float* y, const void* W, const void* x, size_t rows, size_t cols, TypeId id) {
    assert(y && W && x);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    const size_t stride = type_size(id);
    assert(stride > 0);

    // Dequantize input vector once (shared across rows)
    float* xf = malloc(cols * sizeof(float));
    dequant_vec(xf, x, cols, id);

    // Alias this weights current row
    void* row_ptr = NULL;

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        // Allocate memory to this row for this thread
        float* Wf = malloc(cols * sizeof(float));

        // Decode this row of W into float buffer
        if (id == TYPE_Q8) {
            Q8* q8 = (Q8*) W;
            size_t blocks_per_row = (cols + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;

            Q8 row_view = {
                .q = q8->q + i * cols,
                .w = q8->w + i * blocks_per_row,
            };
            row_ptr = &row_view;
            dequant_vec(Wf, row_ptr, cols, id);
        } else {
            row_ptr = (uint8_t*) W + i * cols * stride;
            dequant_vec(Wf, row_ptr, cols, id);
        }

        // Compute dot(W[i, :], x)
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            sum += Wf[j] * xf[j];
        }

        y[i] = sum;
        free(Wf);
    }

    free(xf);
}

/** @} */

/**
 * @section Backward/Gradient Ops
 */

// dW = δ_next ⊗ x^T (outer product)
// dW, d_next must be of type id; x is always float*
void mat_dW(void* dW, const void* d_next, const float* x, size_t rows, size_t cols, TypeId id) {
    assert(dW && d_next && x);
    assert(rows > 0 && cols > 0);
    assert(id < TYPE_COUNT);

    size_t stride = type_size(id);
    assert(stride > 0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        // Dequantize delta for this output (row)
        float d_next_i;
        const void* d_next_src = (const uint8_t*) d_next + i * stride;
        dequant(&d_next_i, d_next_src, id);

        for (size_t j = 0; j < cols; j++) {
            // Pre-compute outer product entry
            float temp = d_next_i * x[j];
            // Get the current derivative
            void* dW_ptr = (uint8_t*) dW + (i * cols + j) * stride;
            // Just update and set. Do not accumulate.
            quant(dW_ptr, temp, id);
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
    TypeId id  // Data type for quantized buffers
) {
    assert(dy && W_next && d_next && z);
    assert(rows > 0 && rows_next > 0);
    assert(id < TYPE_COUNT);

    size_t stride = type_size(id);
    assert(stride > 0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < rows_next; j++) {
            // Dequantize d_next[j]
            float d_next_j;
            const void* d_next_src = (const uint8_t*) d_next + j * stride;
            dequant(&d_next_j, d_next_src, id);

            // Dequantize W_next[j, i] = W_next^T[i, j] (row-major)
            float W_next_T_ji;
            const void* W_next_src = (const uint8_t*) W_next + (j * rows + i) * stride;
            dequant(&W_next_T_ji, W_next_src, id);

            // Accumulate
            sum += W_next_T_ji * d_next_j;
        }

        // Apply pre-activation
        float temp = sum * silu_prime(z[i]);
        // Get the current ouput
        void* dy_ptr = (uint8_t*) dy + i * stride;
        // Update output
        quant(dy_ptr, temp, id);
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
    TypeId id_W,
    TypeId id_dvW,
    float lr,
    float lambda,
    float mu,
    float tau,
    bool nesterov
) {
    assert(W && dW);
    assert(rows > 0 && cols > 0);
    assert(id_W < TYPE_COUNT && id_dvW < TYPE_COUNT);

    size_t stride_W = type_size(id_W);
    size_t stride_dvW = type_size(id_dvW);
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
            dequant(&w, W_ptr, id_W);
            dequant(&g, dW_ptr, id_dvW);

            // L2 regularization (in float)
            if (lambda > 0.0f) {
                g += lambda * w;
            }

            // Momentum (optional)
            if (vW && mu > 0.0f) {
                float v;
                void* vW_ptr = (uint8_t*) vW + idx * stride_dvW;
                dequant(&v, vW_ptr, id_dvW);

                v = mu * v + (1.0f - tau) * g;

                if (nesterov) {
                    g += mu * v;  // Lookahead
                } else {
                    g = v;
                }

                // Store updated velocity
                quant(vW_ptr, v, id_dvW);
            }

            // Weight update
            w -= lr * g;
            quant(W_ptr, w, id_W);
        }
    }
}

/** @} */
