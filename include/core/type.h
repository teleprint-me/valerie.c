/**
 * @file type.h
 * @brief Core numeric type definitions and quantization interface.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This header defines the canonical numeric types used throughout the
 * library, including standard IEEE-754 formats (e8m23, e5m10, e8m7, e4m3)
 * and custom quantized formats such as Q8 (Microscaling 8-bit).
 *
 * ## Overview
 *
 * The type system provides a unified interface for:
 *  - Encoding and decoding between floating-point and reduced-precision formats.
 *  - Scalar, vector, and matrix quantization utilities.
 *  - Compact microscaling-style quantization using block-shared exponents.
 *
 * Each format has explicit encode/decode functions that perform deterministic
 * bit-level conversions according to its respective specification.
 * Higher-level vector functions call these scalar routines internally.
 *
 * The `Q8` type implements the Microscaling data format used in transformer
 * models, where a single 8-bit E4M3 scale is shared across fixed-size blocks
 * of signed 8-bit quantized values.
 *
 * @see https://standards.ieee.org/ieee/754/6210/
 * @see https://dl.acm.org/doi/10.1145/103162.103163
 * @see https://arxiv.org/abs/1710.03740
 * @see https://arxiv.org/abs/2209.05433
 * @see https://arxiv.org/abs/2510.01863
 * @see https://en.wikipedia.org/wiki/IEEE_754
 */

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum TypeId
 * @brief Enumeration of supported numeric data types.
 *
 * These identifiers describe the precision and bit layout of each numeric type.
 * They are used for runtime dispatch, metadata lookup, and quantization.
 */
typedef enum TypeId {
    TYPE_F32,  ///< 32-bit IEEE-754 float (binary32)
    TYPE_E8M23,  ///< 32-bit IEEE-754 float alias (e8m23)
    TYPE_E5M10,  ///< 16-bit float (half precision)
    TYPE_E8M7,  ///< 16-bit extended precision float (used in FMA ops)
    TYPE_E4M3,  ///< 8-bit float (microscaling base format)
    TYPE_Q8,  ///< 8-bit quantized block format (Microscaling)
    TYPE_COUNT  ///< Sentinel: number of supported types
} TypeId;

/**
 * @struct Type
 * @brief Metadata descriptor for a numeric type.
 *
 * Each entry in the static `TYPE_DATA` table provides alignment, size,
 * and type identifier for the corresponding data format.
 */
typedef struct Type {
    const char* name;  ///< Human-readable name (e.g. "f32", "e4m3").
    uint32_t alignment;  ///< Required memory alignment in bytes.
    uint32_t size;  ///< Size in bytes per element.
    TypeId id;  ///< Type identifier.
} Type;

/**
 * @union Float32Union
 * @brief Utility union for bit-level reinterpretation between float and uint32.
 *
 * Used internally for encoding and decoding IEEE-754 32-bit values.
 */
typedef union Float32Union {
    float v;  ///< 32-bit float value.
    uint32_t b;  ///< 32-bit unsigned integer representation.
} Float32Union;

/**
 * @def Q8_BLOCK_SIZE
 * @brief Default number of elements per Q8 quantization block.
 *
 * Each block shares a single E4M3-encoded scaling factor. Override this macro
 * before including the header to change the default block granularity.
 */
#ifndef Q8_BLOCK_SIZE
    #define Q8_BLOCK_SIZE 32
#endif

/**
 * @struct Q8
 * @brief Microscaling 8-bit quantized block format.
 *
 * A lightweight container storing quantized elements and their per-block scales.
 * The data structure intentionally omits shape or dimension metadata, which
 * is managed externally by tensor or array descriptors.
 *
 * Memory layout:
 * ```
 *  Q8 = {
 *      uint8_t* s;  // num_blocks elements, each E4M3-encoded scale
 *      int8_t*  q;  // n elements, signed quantized values
 *  }
 * ```
 *
 * This representation minimizes memory footprint and bandwidth by
 * using 1 byte per quantized value and 1 byte per block scale.
 */
typedef struct Q8 {
    uint8_t* s;  ///< Shared scaling factors (E4M3-encoded per block).
    int8_t* q;  ///< Signed quantized values (per element).
} Q8;

/**
 * @brief Static table of data type metadata.
 *
 * The array index corresponds to the `TypeId` enumeration.
 * Use `type_data()` for safe lookup with bounds checking.
 */
static const Type TYPE_DATA[TYPE_COUNT] = {
    [TYPE_F32] = {"f32", alignof(float), sizeof(float), TYPE_F32},
    [TYPE_E8M23] = {"e8m23", alignof(uint32_t), sizeof(uint32_t), TYPE_E8M23},
    [TYPE_E5M10] = {"e5m10", alignof(uint16_t), sizeof(uint16_t), TYPE_E5M10},
    [TYPE_E8M7] = {"e8m7", alignof(uint16_t), sizeof(uint16_t), TYPE_E8M7},
    [TYPE_E4M3] = {"e4m3", alignof(uint8_t), sizeof(uint8_t), TYPE_E4M3},
    [TYPE_Q8] = {"q8", alignof(Q8), sizeof(Q8), TYPE_Q8},
};

/**
 * Metadata Accessors
 * @{
 */

/**
 * @brief Retrieve metadata for a given type identifier.
 * @param id Type identifier.
 * @return Pointer to the Type metadata, or NULL if invalid.
 */
const Type* type_data(TypeId id);

/**
 * @brief Get the string name of a type identifier.
 * @param id Type identifier.
 * @return Constant string (e.g. "e4m3") or "unknown" if invalid.
 */
const char* type_name(TypeId id);

/**
 * @brief Get the size in bytes of a type identifier.
 * @param id Type identifier.
 * @return Size in bytes, or 0 if invalid.
 */
uint32_t type_size(TypeId id);

/** @} */

/**
 * Floating-Point Conversions
 * @{
 */

/* IEEE-754 float32 */
uint32_t e8m23_encode(float v);
float e8m23_decode(uint32_t b);

/* Half precision (e5m10) */
uint16_t e5m10_encode(float v);
float e5m10_decode(uint16_t b);

/* Extended 16-bit precision (e8m7) */
uint16_t e8m7_encode(float v);
float e8m7_decode(uint16_t b);

/* 8-bit float (e4m3) */
uint8_t e4m3_encode(float v);
float e4m3_decode(uint8_t b);

/** @} */

/**
 * Block Quantization
 * @{
 */

/**
 * @brief Quantize a float array into Q8 format using shared E4M3 block scales.
 * @param[out] dst Output Q8 structure containing quantized values and scales.
 * @param[in] src Input float array.
 * @param[in] n Number of elements.
 * @param[in] block_size Elements per block (typically Q8_BLOCK_SIZE).
 */
void q8_encode(Q8* dst, const float* src, size_t n, size_t block_size);

/**
 * @brief Dequantize Q8 blocks back into float values.
 * @param[out] dst Output float array.
 * @param[in] src Input Q8 structure.
 * @param[in] n Number of elements.
 * @param[in] block_size Elements per block (typically Q8_BLOCK_SIZE).
 */
void q8_decode(float* dst, const Q8* src, size_t n, size_t block_size);

/** @} */

/**
 * Quantization Interface
 * @{
 */

/**
 * @brief Quantize a single scalar into a target type.
 * @param[out] dst Destination buffer for encoded value.
 * @param[in] src Input float value.
 * @param[in] dst_id Destination type identifier.
 * @return true if successful, false if type unsupported.
 */
bool quant(void* dst, float src, TypeId dst_id);

/**
 * @brief Decode a single scalar from a quantized value.
 * @param[out] dst Output float value.
 * @param[in] src Encoded value buffer.
 * @param[in] src_id Source type identifier.
 * @return true if successful, false if type unsupported.
 */
bool dequant(float* dst, const void* src, TypeId src_id);

/**
 * @brief Quantize a 1D array of float values into a target format.
 *
 * This is the general-purpose vector quantization routine.
 * For TYPE_Q8, this will internally call `q8_encode`.
 */
bool quant_vec(void* dst, const float* src, size_t len, TypeId dst_id);

/**
 * @brief Dequantize a 1D array from a quantized format to float32.
 *
 * For TYPE_Q8, this will internally call `q8_decode`.
 */
bool dequant_vec(float* dst, const void* src, size_t len, TypeId src_id);

/**
 * @brief Quantize a 2D matrix (flattened) of float values into a target format.
 * @param[out] dst Destination buffer for encoded values.
 * @param[in] src Input float matrix.
 * @param[in] rows Number of rows.
 * @param[in] cols Number of columns.
 * @param[in] dst_id Destination type identifier.
 * @return true if successful, false otherwise.
 */
bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, TypeId dst_id);

/**
 * @brief Dequantize a 2D matrix (flattened) from a quantized format to float32.
 * @param[out] dst Output float matrix.
 * @param[in] src Input encoded matrix.
 * @param[in] rows Number of rows.
 * @param[in] cols Number of columns.
 * @param[in] src_id Source type identifier.
 * @return true if successful, false otherwise.
 */
bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, TypeId src_id);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DATA_TYPE_H */
