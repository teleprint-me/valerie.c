/**
 * @file core/type.h
 * @brief Numeric data types and conversions API.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * Features:
 * - Single and half-precision floating-point support.
 * - 8-bit and 4-bit quantized integer support (Q4 planned).
 * - Consistent, extensible type metadata and conversion utilities.
 */

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Union for floating-point bit manipulation */
typedef union FloatBits {
    float value;
    uint32_t bits;
} FloatBits;

/** @brief Quantized 8-bit value and scaling factor */
typedef struct QuantBits {
    uint8_t bits;
    uint16_t scalar;
} QuantBits;

// Type aliases for encodings
typedef QuantBits quant8_t; /**< 8-bit quantization */
typedef uint16_t float16_t; /**< IEEE-754 half precision */
typedef uint16_t bfloat16_t; /**< Google brain float16 */

// Supported data types
typedef enum DataTypeId {
    TYPE_FLOAT32,
    TYPE_FLOAT16,
    TYPE_BFLOAT16,
    TYPE_QUANT8,
    TYPE_INT32,
    TYPE_INT16,
    TYPE_INT8,
    TYPE_UINT32,
    TYPE_UINT16,
    TYPE_UINT8,
    TYPE_BOOL,
    TYPE_CHAR,
    TYPE_COUNT
} DataTypeId;

// Data type sign
typedef enum DataTypeSign {
    TYPE_NOT_APPLICABLE,
    TYPE_IS_SIGNED,
    TYPE_IS_UNSIGNED,
} DataTypeSign;

// Metadata for data types
typedef struct DataType {
    const char* name;
    uint32_t alignment;
    uint32_t size;
    DataTypeSign sign;
    DataTypeId id;
} DataType;

// Static table of type metadata
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

/**
 * @name Data Type Metadata Utilities
 * @{
 */

/**
 * @brief Retrieve metadata by type ID
 */
const DataType* data_type_get(DataTypeId id);

/**
 * @brief Get size in bytes of a data type by ID
 */
uint32_t data_type_size(DataTypeId id);

/**
 * @brief Get human-readable name for a data type by ID
 */
const char* data_type_name(DataTypeId id);

/** @} */

/**
 * @name Scalar Encoding/Decoding
 * @{
 */

/**
 * @brief Encode 32-bit float as raw bits
 */
uint32_t encode_fp32(float value);

/**
 * @brief Decode 32-bit float from raw bits
 */
float decode_fp32(uint32_t bits);

/**
 * @brief Quantize 32-bit float to IEEE half-precision (16-bit)
 */
float16_t encode_fp16(float value);

/**
 * @brief Dequantize IEEE half-precision (16-bit) to 32-bit float
 */
float decode_fp16(float16_t bits);

/**
 * @brief Quantize 32-bit float to bfloat16 (16-bit, Google)
 */
bfloat16_t encode_bf16(float value);

/**
 * @brief Dequantize bfloat16 (16-bit, Google) to 32-bit float
 */
float decode_bf16(bfloat16_t bits);

/**
 * @brief Quantize 32-bit float to 8-bit quantized (custom)
 */
quant8_t encode_q8(float value);

/**
 * @brief Dequantize 8-bit quantized value to 32-bit float
 */
float decode_q8(quant8_t q8);

/** @} */

/**
 * @name Scalar Quantization/Dequantization (by type)
 * @{
 */

/**
 * @brief Quantize a scalar float to another type.
 * @param[out] dst Pointer to destination buffer (scalar of type dst_id)
 * @param[in]  src Source scalar float value
 * @param[in]  dst_id Target data type ID
 * @return true if successful, false otherwise
 */
bool quant_scalar(void* dst, float src, DataTypeId dst_id);

/**
 * @brief Dequantize a scalar of given type to float.
 * @param[out] dst Pointer to destination float
 * @param[in]  src Pointer to source scalar (of type src_id)
 * @param[in]  src_id Source data type ID
 * @return true if successful, false otherwise
 */
bool dequant_scalar(float* dst, const void* src, DataTypeId src_id);

/** @} */

/**
 * @name Vector Quantization/Dequantization
 * @{
 */

/**
 * @brief Quantize a vector of floats to a vector of another type.
 * @param[out] dst Destination buffer (length elements of type dst_id)
 * @param[in]  src Source buffer (float*, length elements)
 * @param[in]  len Number of elements
 * @param[in]  dst_id Target data type ID
 * @return true if successful, false otherwise
 */
bool quant_vec(void* dst, const float* src, size_t len, DataTypeId dst_id);

/**
 * @brief Dequantize a vector of src_id to floats.
 * @param[out] dst Destination buffer (float*, length elements)
 * @param[in]  src Source buffer (length elements of type src_id)
 * @param[in]  len Number of elements
 * @param[in]  src_id Source data type ID
 * @return true if successful, false otherwise
 */
bool dequant_vec(float* dst, const void* src, size_t len, DataTypeId src_id);

/** @} */

/**
 * @name Matrix Quantization/Dequantization
 * @{
 */

/**
 * @brief Quantize a flat matrix (row-major) of floats to another type.
 * @param[out] dst Destination buffer ((rows * cols) of type dst_id)
 * @param[in]  src Source buffer (float*, rows * cols)
 * @param[in]  rows Number of rows
 * @param[in]  cols Number of columns
 * @param[in]  dst_id Target data type ID
 * @return true if successful, false otherwise
 */
bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, DataTypeId dst_id);

/**
 * @brief Dequantize a flat matrix (row-major) of src_id to floats.
 * @param[out] dst Destination buffer (float*, rows * cols)
 * @param[in]  src Source buffer ((rows * cols) of type src_id)
 * @param[in]  rows Number of rows
 * @param[in]  cols Number of columns
 * @param[in]  src_id Source data type ID
 * @return true if successful, false otherwise
 */
bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, DataTypeId src_id);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // DATA_TYPE_H
