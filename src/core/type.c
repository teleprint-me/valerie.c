/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/type.c
 *
 * @brief API for numeric data types and conversions.
 *
 * Features:
 * - Single and half-precision floating-point support.
 * - 8-bit quantized integer support.
 * - Minimal dependencies with a consistent, extensible design.
 */

#include <math.h>

#include "core/type.h"

/**
 * @section Data type management
 */

const DataType* data_type_get(DataTypeId id) {
    // Bounds checking to avoid invalid access
    if (id >= TYPE_COUNT) {
        return NULL;  // Invalid type
    }
    return &TYPES[id];
}

uint32_t data_type_size(DataTypeId id) {
    const DataType* type = data_type_get(id);
    return type ? type->size : 0;
}

const char* data_type_name(DataTypeId id) {
    const DataType* type = data_type_get(id);
    return type ? type->name : "Unknown";
}

/** @} */

/**
 * @section Unit conversions
 * Supports 32, 16, and 8-bit formats.
 */

// 32-bit encoding and decoding
uint32_t encode_fp32(float value) {
    FloatBits raw;
    raw.value = value;
    return raw.bits;
}

float decode_fp32(uint32_t bits) {
    FloatBits raw;
    raw.bits = bits;
    return raw.value;
}

// Half-precision floating-point quantization
float16_t encode_fp16(float value) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(value) * scale_to_inf) * scale_to_zero;

    const uint32_t w = encode_fp32(value);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000;
    uint32_t bias = shl1_w & 0xFF000000;

    if (bias < 0x71000000) {
        bias = 0x71000000;
    }

    base = decode_fp32((bias >> 1) + 0x07800000) + base;
    const uint32_t bits = encode_fp32(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00;
    const uint32_t mantissa_bits = bits & 0x00000FFF;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > 0xFF000000 ? 0x7E00 : nonsign);
}

float decode_fp16(float16_t bits) {
    const uint32_t w = (uint32_t) bits << 16;
    const uint32_t sign = w & 0x80000000;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0 << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = decode_fp32((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = 0x7E000000;
    const float magic_bias = 0.5f;
    const float denormalized_value = decode_fp32((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = 1 << 27;
    const uint32_t result = sign
                            | (two_w < denormalized_cutoff ? encode_fp32(denormalized_value)
                                                           : encode_fp32(normalized_value));
    return decode_fp32(result);
}

// Google brain floating-point format
bfloat16_t encode_bf16(float value) {
    const uint32_t exp = 0x7f800000u;
    const uint32_t sign = 0x80000000u;
    const uint32_t abs = 0x7fffffffu;
    const uint32_t qnan = 0x40u;

    FloatBits raw = {.value = value};

    if ((raw.bits & abs) > exp) { /* nan */
        return (uint16_t) ((raw.bits >> 16) | qnan);
    }

    if (!(raw.bits & exp)) { /* subnormal */
        return (uint16_t) ((raw.bits & sign) >> 16); /* flush */
    }

    return (uint16_t) ((raw.bits + (0x7fff + ((raw.bits >> 16) & 1))) >> 16);
}

float decode_bf16(bfloat16_t bits) {
    FloatBits raw;
    raw.bits = (uint32_t) bits << 16;
    return raw.value;
}

// 8-bit quantization with residual baking
quant8_t encode_q8(float value) {
    quant8_t q8;

    // Define integer domain
    uint32_t z_domain = 255;
    // Reflect and compute effective real domain
    float r_domain = fabsf(value);

    // Special case for zero
    if (r_domain == 0.0f) {
        q8.scalar = encode_fp16(1.0f);
        q8.bits = 0;
        return q8;
    }

    // Calculate squeezing ratio
    float alpha = (r_domain > z_domain) ? z_domain / r_domain : 1.0f;
    // Calculate the base step size
    float step_size = r_domain / z_domain;  // Decoupled from scalar
    // Quantize the value using the base step size (exponent/mantissa?)
    uint8_t bits = roundf(value / step_size);
    // Calculate the residual precision (bias?)
    float residual = (value - (bits * step_size));

    // Calculate the scalar based on the squeezed range
    float scalar = step_size * alpha + residual;

    // Quantize to scalar to half-precision
    q8.scalar = encode_fp16(scalar);
    // Quantize the value
    q8.bits = bits;

    return q8;
}

// Dequantize the value
float decode_q8(quant8_t q8) {
    // Dequantize the scalar value
    float scalar = decode_fp16(q8.scalar);
    // Scale the bits proportionally back to original float
    return (float) (q8.bits * scalar);
}

/** @} */

/**
 * @section Scalar conversions
 * Supports 32, 16, and 8-bit formats.
 */

bool quant_scalar(void* dst, float src, DataTypeId dst_id) {
    switch (dst_id) {
        case TYPE_FLOAT32: {
            *(float*) dst = src;
            break;
        }
        case TYPE_UINT32: {
            *(uint32_t*) dst = encode_fp32(src);
            break;
        }
        case TYPE_FLOAT16: {
            *(uint16_t*) dst = encode_fp16(src);
            break;
        }
        case TYPE_BFLOAT16: {
            *(uint16_t*) dst = encode_bf16(src);
            break;
        }
        case TYPE_QUANT8: {
            *(quant8_t*) dst = encode_q8(src);
            break;
        }
        default:
            return false;
    }
    return true;
}

bool dequant_scalar(float* dst, const void* src, DataTypeId src_id) {
    switch (src_id) {
        case TYPE_FLOAT32: {
            *dst = *(const float*) src;
            break;
        }
        case TYPE_UINT32: {
            *dst = decode_fp32(*(uint32_t*) src);
            break;
        }
        case TYPE_FLOAT16: {
            *dst = decode_fp16(*(uint16_t*) src);
            break;
        }
        case TYPE_BFLOAT16: {
            *dst = decode_bf16(*(uint16_t*) src);
            break;
        }
        case TYPE_QUANT8: {
            *dst = decode_q8(*(quant8_t*) src);
            break;
        }
        default:
            return false;
    }
    return true;
}

/** @} */

/**
 * Vector conversions (1D arrays)
 * Supports 32, 16, and 8-bit formats.
 */

bool quant_vec(void* out, const float* in, size_t len, DataTypeId dst_id) {
    assert(in != NULL);
    assert(out != NULL);
    assert(len > 0);
    assert(dst_id < TYPE_COUNT);

    size_t stride = data_type_size(dst_id);
    assert(stride > 0);

    for (size_t i = 0; i < len; ++i) {
        void* dst = (uint8_t*) out + i * stride;
        if (!quant_scalar(dst, in[i], dst_id)) {
            return false;
        }
    }

    return true;
}

bool dequant_vec(float* out, const void* in, size_t len, DataTypeId src_id) {
    assert(in != NULL);
    assert(out != NULL);
    assert(len > 0);
    assert(src_id < TYPE_COUNT);

    size_t stride = data_type_size(src_id);
    assert(stride > 0);

    for (size_t i = 0; i < len; ++i) {
        const void* src = (const uint8_t*) in + i * stride;
        if (!dequant_scalar(&out[i], src, src_id)) {
            return false;
        }
    }

    return true;
}

/** @} */

/**
 * Matrix conversions (2D flat arrays)
 * Supports 32, 16, and 8-bit formats.
 */

bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, DataTypeId dst_id) {
    assert(src != NULL);
    assert(dst != NULL);
    assert(rows > 0);
    assert(cols > 0);
    assert(dst_id < TYPE_COUNT);

    return quant_vec(dst, src, rows * cols, dst_id);
}

bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, DataTypeId src_id) {
    assert(src != NULL);
    assert(dst != NULL);
    assert(rows > 0);
    assert(cols > 0);
    assert(src_id < TYPE_COUNT);

    return dequant_vec(dst, src, rows * cols, src_id);
}

/** @} */
