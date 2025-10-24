/**
 * @file quant.h
 * @brief Unified quantization interface for scalar, vector, and matrix types.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This header provides high-level, type-driven quantization routines for:
 *   - Scalar encode/decode (float <-> compact type)
 *   - Vector encode/decode (float[] <-> quantized[] or quant8_t)
 *   - Matrix encode/decode (float[] <-> quantized[] or quant8_t[])
 *
 * Special handling is provided for Q8 (microscaling) types:
 *   - Vectors: quant8_t (use q8_vec_* routines)
 *   - Matrices: array of quant8_t, one per row (use q8_mat_* routines)
 *
 * Caller is responsible for correct output buffer type/allocation.
 */

#ifndef QUANT_H
#define QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include "linear/type.h"

/**
 * @name Scalar Quantization
 * @{
 */

/**
 * @brief Quantize a single scalar float into a target type.
 *        dst must point to the appropriate type for dst_id.
 * @param[out] dst   Pointer to output value (float, uint16_t, quant8_t, etc.).
 * @param[in]  src   Input float value.
 * @param[in]  dst_id Destination type identifier (TypeId).
 * @return true if successful, false if unsupported type.
 */
bool quant(void* dst, float src, TypeId dst_id);

/**
 * @brief Dequantize a single scalar value to float.
 *        src must point to the correct type for src_id.
 * @param[out] dst    Pointer to output float.
 * @param[in]  src    Input value (float, uint16_t, quant8_t, etc.).
 * @param[in]  src_id Source type identifier (TypeId).
 * @return true if successful, false if unsupported type.
 */
bool dequant(float* dst, const void* src, TypeId src_id);

/** @} */

/**
 * @name Vector Quantization (1D)
 * @{
 */

/**
 * @brief Quantize a float array to a given type.
 *        For TYPE_Q8, dst must be quant8_t* (single Q8 vector).
 *        For others, dst is an array of [len] elements of the target type.
 * @param[out] dst     Output buffer (see above).
 * @param[in]  src     Input float array [len].
 * @param[in]  len     Number of elements.
 * @param[in]  dst_id  Destination type identifier (TypeId).
 * @return true if successful, false if error/unsupported.
 */
bool quant_vec(void* dst, const float* src, size_t len, TypeId dst_id);

/**
 * @brief Dequantize a quantized array to float array.
 *        For TYPE_Q8, src must be quant8_t* (single Q8 vector).
 *        For others, src is an array of [len] elements of the quantized type.
 * @param[out] dst     Output float array [len].
 * @param[in]  src     Input quantized buffer.
 * @param[in]  len     Number of elements.
 * @param[in]  src_id  Source type identifier (TypeId).
 * @return true if successful, false if error/unsupported.
 */
bool dequant_vec(float* dst, const void* src, size_t len, TypeId src_id);

/** @} */

/**
 * @name Matrix Quantization (2D, flat row-major)
 * @{
 */

/**
 * @brief Quantize a float matrix (flat row-major [rows*cols]) into target type.
 *        For TYPE_Q8, dst must be quant8_t* array of [rows] (each row cols long).
 *        For others, dst is a flat array of [rows*cols] elements of the target type.
 * @param[out] dst     Output buffer (see above).
 * @param[in]  src     Input float matrix [rows*cols].
 * @param[in]  rows    Number of rows.
 * @param[in]  cols    Number of columns.
 * @param[in]  dst_id  Destination type identifier (TypeId).
 * @return true if successful, false if error/unsupported.
 */
bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, TypeId dst_id);

/**
 * @brief Dequantize a quantized matrix to float (flat row-major [rows*cols]).
 *        For TYPE_Q8, src must be quant8_t* array of [rows] (each row cols long).
 *        For others, src is a flat array of [rows*cols] quantized elements.
 * @param[out] dst     Output float array [rows*cols].
 * @param[in]  src     Input quantized buffer.
 * @param[in]  rows    Number of rows.
 * @param[in]  cols    Number of columns.
 * @param[in]  src_id  Source type identifier (TypeId).
 * @return true if successful, false if error/unsupported.
 */
bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, TypeId src_id);

/** @} */

#endif  // QUANT_H
