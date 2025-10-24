/**
 * @file q8.c
 * @brief Microscaling Floating Point Formats for Large Language Models.
 * @copyright Copyright © 2023 Austin Berrio
 * @ref https://arxiv.org/abs/2510.01863
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "linear/q8.h"

// internal use only? not sure yet. might be useful externally.
void q8_assert(size_t len) {
    assert(Q8_BLOCK_SIZE > 0 && "Requires at least 1 block");
    assert(len >= Q8_BLOCK_SIZE && "Length must be greater than or equal to block size");
    assert(len % Q8_BLOCK_SIZE == 0 && "Length must be evenly divisible by block size");
}

// useful for calculating total number of blocks and the current block index.
// a block is a segement from start to end of len n of a vector of len m where m > n.
// i still don't know what to name this.
// just block? block num? block len? block idx?
size_t q8_block(size_t n) {
    return n / Q8_BLOCK_SIZE;
}

quant8_t q8_vec_new(size_t len) {
    q8_assert(len);

    const size_t num_blocks = q8_block(len);
    return (quant8_t) {
        .q = calloc(len, sizeof(int8_t)),
        .w = calloc(num_blocks, sizeof(int8_t)),
    };
}

void q8_vec_free(quant8_t* q8) {
    if (q8) {
        free(q8->q);
        free(q8->w);
        q8->q = NULL;
        q8->w = NULL;
    }
}

// Allocate a Q8 matrix (array of quant8_t for each row)
quant8_t* q8_mat_new(size_t rows, size_t cols) {
    quant8_t* Wq = calloc(rows, sizeof(quant8_t));
    if (!Wq) {
        return NULL;
    }
    for (size_t r = 0; r < rows; r++) {
        Wq[r] = q8_vec_new(cols);
    }
    return Wq;
}

// Free a Q8 matrix (array of quant8_t for each row)
void q8_mat_free(quant8_t* Wq, size_t rows) {
    if (!Wq) {
        return;
    }
    for (size_t r = 0; r < rows; r++) {
        q8_vec_free(&Wq[r]);
    }
    free(Wq);
}

void q8_vec_encode(quant8_t* dst, const float* src, size_t len) {
    q8_assert(len);

    const int q8_max = 127;  // largest representable value for int8
    const int e4m3_exp_max = 7;  // largest representable exponent for e4m3
    const size_t block_size = Q8_BLOCK_SIZE;  // number of elements per block
    const size_t num_blocks = q8_block(len);  // number of blocks in this vector

    for (size_t b = 0; b < num_blocks; b++) {
        // Calculate offsets
        int8_t* q = dst->q + b * block_size;
        const float* x = src + b * block_size;

        // Find max normal |x|
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_size; i++) {
            float absval = fabsf(x[i]);
            if (absval > max_abs) {
                max_abs = absval;
            }
        }
        // handle all-zero/subnormal case
        int all_zero = (max_abs == 0.0f);

        // Compute shared exponent and scale
        int ilogb_p = all_zero ? 0 : ilogbf(max_abs);  // ilogb(0) is FP_ILOGB0, but we check above
        int w = all_zero ? 0 : ilogb_p - e4m3_exp_max;
        // For e4m3, exponent range is -7 to 8, so clamp w if needed
        if (!all_zero) {
            if (w < -7) {
                w = -7;
            }
            if (w > 8) {
                w = 8;
            }
        }
        // store scale for this block
        dst->w[b] = w;

        // compute scale for this block
        float scale = ldexpf(1.0f, w);  // 1.0 x 2^w = 2^w

        // Quantize
        for (size_t i = 0; i < block_size; i++) {
            // scale float to 8-bit
            float xw = all_zero ? 0.0f : x[i] / scale;

            // symmetric clamp to 8-bit range
            int r = (int) nearbyintf(xw);
            if (r > q8_max) {
                r = q8_max;
            }
            if (r < -q8_max) {
                r = -q8_max;
            }

            // store block scaled element
            q[i] = r;
        }
    }
}

void q8_vec_decode(float* dst, const quant8_t* src, size_t len) {
    q8_assert(len);

    // Dequantize
    for (size_t i = 0; i < len; i++) {
        size_t block = q8_block(i);
        float scale = ldexpf(1.0f, src->w[block]);
        dst[i] = src->q[i] * scale;
    }
}

// Encode a float matrix (row-major, flat) into a Q8 matrix
void q8_mat_encode(quant8_t* Wq, const float* W, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        const float* row = W + r * cols;
        q8_vec_encode(&Wq[r], row, cols);
    }
}

// Decode a Q8 matrix into a float matrix (row-major, flat)
void q8_mat_decode(float* W_out, const quant8_t* Wq, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float* row_out = W_out + r * cols;
        q8_vec_decode(row_out, &Wq[r], cols);
    }
}
