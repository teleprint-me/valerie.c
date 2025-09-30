/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/type.h
 *
 * @brief API for numeric data types and conversions.
 *
 * Features:
 * - Single and half-precision floating-point support.
 * - 8-bit and 4-bit quantized integer support.
 * - Minimal dependencies with a consistent, extensible design.
 */

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>

#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Union for floating-point bit manipulation
typedef union FloatBits {
    float value; /**< Floating-point value */
    uint32_t bits; /**< Bit-level representation */
} FloatBits;

// Quantization structure
typedef struct QuantBits {
    uint8_t bits; /**< Quantized value with baked residual */
    uint16_t scalar; /**< Scaling factor */
} QuantBits;

// Type aliases for encodings
typedef QuantBits quant8_t; /**< 8-bit quantization */
typedef uint16_t float16_t;
typedef uint16_t bfloat16_t;

// Supported data types
typedef enum DataTypeId {
    TYPE_FLOAT32, /**< 32-bit floating-point (IEEE-754) */
    TYPE_FLOAT16, /**< 16-bit floating-point (IEEE-754) */
    TYPE_BFLOAT16, /**< 16-bit floating-point (bfloat16) */
    TYPE_QUANT8, /**< 8-bit quantized integer */
    TYPE_INT32, /**< 32-bit signed integer */
    TYPE_INT16, /**< 16-bit signed integer */
    TYPE_INT8, /**< 8-bit signed integer */
    TYPE_UINT32, /**< 32-bit unsigned integer */
    TYPE_UINT16, /**< 16-bit unsigned integer */
    TYPE_UINT8, /**< 8-bit unsigned integer */
    TYPE_BOOL, /**< Boolean */
    TYPE_CHAR, /**< 1-byte character */
    TYPE_COUNT /**< Total number of types */
} DataTypeId;

// Data type sign
typedef enum DataTypeSign {
    TYPE_NOT_APPLICABLE, /**< Not applicable (e.g., for packed types) */
    TYPE_IS_SIGNED, /**< Signed types */
    TYPE_IS_UNSIGNED /**< Unsigned types */
} DataTypeSign;

// Metadata for data types
typedef struct DataType {
    const char* name; /**< Human-readable name */
    uint32_t alignment; /**< Memory alignment in bytes */
    uint32_t size; /**< Size in bytes */
    DataTypeSign sign; /**< Signed/unsigned status */
    DataTypeId id; /**< Unique identifier */
} DataType;

// Static array of supported types
static const DataType TYPES[TYPE_COUNT] = {
    [TYPE_FLOAT32] = {"float32", alignof(float), sizeof(float), TYPE_IS_SIGNED, TYPE_FLOAT32},
    [TYPE_FLOAT16]
    = {"float16", alignof(float16_t), sizeof(float16_t), TYPE_IS_UNSIGNED, TYPE_FLOAT16},
    [TYPE_BFLOAT16]
    = {"bfloat16", alignof(bfloat16_t), sizeof(bfloat16_t), TYPE_IS_UNSIGNED, TYPE_BFLOAT16},
    [TYPE_QUANT8]
    = {"qint8", alignof(quant8_t), sizeof(quant8_t), TYPE_NOT_APPLICABLE, TYPE_QUANT8},
    [TYPE_INT32] = {"int32", alignof(int32_t), sizeof(int32_t), TYPE_IS_SIGNED, TYPE_INT32},
    [TYPE_INT16] = {"int16", alignof(int16_t), sizeof(int16_t), TYPE_IS_SIGNED, TYPE_INT16},
    [TYPE_INT8] = {"int8", alignof(int8_t), sizeof(int8_t), TYPE_IS_SIGNED, TYPE_INT8},
    [TYPE_UINT32] = {"uint32", alignof(uint32_t), sizeof(uint32_t), TYPE_IS_UNSIGNED, TYPE_UINT32},
    [TYPE_UINT16] = {"uint16", alignof(uint16_t), sizeof(uint16_t), TYPE_IS_UNSIGNED, TYPE_UINT16},
    [TYPE_UINT8] = {"uint8", alignof(uint8_t), sizeof(uint8_t), TYPE_IS_UNSIGNED, TYPE_UINT8},
    [TYPE_BOOL] = {"bool", alignof(bool), sizeof(bool), TYPE_NOT_APPLICABLE, TYPE_BOOL},
    [TYPE_CHAR] = {"char", alignof(char), sizeof(char), TYPE_IS_UNSIGNED, TYPE_CHAR},
};

// Data type management
const DataType* data_type_get(DataTypeId id); /**< Retrieve metadata by type ID */
uint32_t data_type_size(DataTypeId id); /**< Get size of type by ID */
const char* data_type_name(DataTypeId id); /**< Get name of type by ID */

// Scalar conversions

// Floating-point encoding/decoding
uint32_t encode_fp32(float value); /**< Encode 32-bit float to bits */
float decode_fp32(uint32_t bits); /**< Decode bits to 32-bit float */

// Half-precision floating-point
float16_t encode_fp16(float value); /**< Quantize 32-bit float to 16-bit */
float decode_fp16(float16_t bits); /**< Dequantize 16-bit to 32-bit float */

// Google brain floating-point format
bfloat16_t encode_bf16(float value); /**< Quantize 32-bit float to 16-bit */
float decode_bf16(bfloat16_t bits); /**< Dequantize 16-bit to 32-bit float */

// 8-bit integer quantization
quant8_t encode_q8(float value); /**< Quantize 32-bit float to 8-bit */
float decode_q8(quant8_t q8); /**< Dequantize 8-bit to 32-bit float */

// Supports 32, 16, and 8-bit formats. Q4 is excluded.
bool quant_scalar(void* dst, float src, DataTypeId dst_id);
bool dequant_scalar(float* dst, const void* src, DataTypeId src_id);

// Vector conversions (1D arrays)

// Supports 32, 16, and 8-bit formats.
bool quant_vec(void* out, const float* in, size_t len, DataTypeId dst_id);
bool dequant_vec(float* out, const void* in, size_t len, DataTypeId src_id);

// Matrix conversions (2D flat arrays)
bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, DataTypeId dst_id);
bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, DataTypeId src_id);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DATA_TYPE_H
