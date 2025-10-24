/**
 * @file q8.h
 * @brief Microscaling Floating Point Formats for Large Language Models.
 * @copyright Copyright © 2023 Austin Berrio
 * @ref https://arxiv.org/abs/2510.01863
 *
 * Low-level API for blockwise quantization using 8-bit int + block exponents.
 * Provides both vector and matrix (row-wise) utilities.
 */

#ifndef Q8_H
#define Q8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef Q8_BLOCK_SIZE
    #define Q8_BLOCK_SIZE 32
#endif

/**
 * @struct quant8_t
 * @brief Q8-quantized vector: signed int8 data + per-block int8 exponents.
 *        Matrix storage is always an array of these (one per row).
 */
typedef struct quant8_t {
    int8_t* q;  ///< Quantized values [len]
    int8_t* w;  ///< Block exponents/scales [len / Q8_BLOCK_SIZE]
} quant8_t;

/**
 * Vector Q8 API
 **/

/**
 * @brief Check Q8 vector length for block size invariants.
 *        Aborts on invalid input.
 */
void q8_assert(size_t len);

/**
 * @brief Utility: Returns the number of Q8 blocks in a vector of length `n`.
 *        Example: q8_block(64) == 2 when Q8_BLOCK_SIZE == 32.
 */
size_t q8_block(size_t n);

/**
 * @brief Allocate a Q8-quantized vector of `len` elements.
 *        Returns zero-initialized struct. Caller must free with q8_vec_free.
 *        Aborts on invalid length (not multiple of Q8_BLOCK_SIZE).
 */
quant8_t q8_vec_new(size_t len);

/**
 * @brief Free the storage for a Q8 vector (no-op on NULL input).
 *        Sets pointers to NULL.
 */
void q8_vec_free(quant8_t* q8);

/**
 * @brief Quantize a float vector into Q8 format (blockwise).
 *        `dst` must be preallocated (see q8_vec_new).
 *        No-op if `len` is zero.
 */
void q8_vec_encode(quant8_t* dst, const float* src, size_t len);

/**
 * @brief Dequantize a Q8 vector back to float.
 *        Output array must have length `len`.
 */
void q8_vec_decode(float* dst, const quant8_t* src, size_t len);

/**
 * Matrix (Rowwise Q8) API
 **/

/**
 * @brief Allocate a matrix of Q8 vectors (one per row).
 *        Returns array of length [rows] (each q8_vec_new(cols)).
 *        Returns NULL on allocation failure.
 */
quant8_t* q8_mat_new(size_t rows, size_t cols);

/**
 * @brief Free an array of Q8 vectors (matrix, length [rows]).
 *        Each vector is freed with q8_vec_free.
 *        No-op on NULL pointer.
 */
void q8_mat_free(quant8_t* Wq, size_t rows);

/**
 * @brief Quantize a float matrix (row-major, [rows*cols]) into a Q8 matrix.
 *        Each row is quantized independently. Wq must be preallocated by q8_mat_new.
 */
void q8_mat_encode(quant8_t* Wq, const float* W, size_t rows, size_t cols);

/**
 * @brief Dequantize a Q8 matrix to a float matrix (row-major, [rows*cols]).
 *        Each row is dequantized independently.
 */
void q8_mat_decode(float* W_out, const quant8_t* Wq, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif  // Q8_H
