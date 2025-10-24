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
#include <stdint.h>

#include "linear/scalar.h"
#include "linear/q8.h"

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
 * @brief Static table of data type metadata.
 *
 * The array index corresponds to the `TypeId` enumeration.
 * Use `type_data()` for safe lookup with bounds checking.
 */
static const Type TYPE_DATA[TYPE_COUNT] = {
    [TYPE_F32] = {"f32", alignof(float), sizeof(float), TYPE_F32},
    [TYPE_E8M23] = {"e8m23", alignof(float32_t), sizeof(float32_t), TYPE_E8M23},
    [TYPE_E5M10] = {"e5m10", alignof(float16_t), sizeof(float16_t), TYPE_E5M10},
    [TYPE_E8M7] = {"e8m7", alignof(bfloat16_t), sizeof(float16_t), TYPE_E8M7},
    [TYPE_E4M3] = {"e4m3", alignof(float8_t), sizeof(float8_t), TYPE_E4M3},
    [TYPE_Q8] = {"q8", alignof(quant8_t), sizeof(quant8_t), TYPE_Q8},
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

#ifdef __cplusplus
}
#endif

#endif /* DATA_TYPE_H */
