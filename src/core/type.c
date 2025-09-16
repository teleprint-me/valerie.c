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

#include "core/type.h"

// Data type management

const DataType* data_type_get(DataTypeId id) {
    // Bounds checking to avoid invalid access
    if (id >= TYPE_COUNT) {
        return NULL; // Invalid type
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

// Scalar Conversions

// 32-bit encoding and decoding
uint32_t encode_scalar_fp32(float value) {
    FloatBits raw;
    raw.value = value;
    return raw.bits;
}

float decode_scalar_fp32(uint32_t bits) {
    FloatBits raw;
    raw.bits = bits;
    return raw.value;
}

// Half-precision floating-point quantization
uint16_t quantize_scalar_fp16(float value) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(value) * scale_to_inf) * scale_to_zero;

    const uint32_t w = encode_scalar_fp32(value);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000;
    uint32_t bias = shl1_w & 0xFF000000;

    if (bias < 0x71000000) {
        bias = 0x71000000;
    }

    base = decode_scalar_fp32((bias >> 1) + 0x07800000) + base;
    const uint32_t bits = encode_scalar_fp32(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00;
    const uint32_t mantissa_bits = bits & 0x00000FFF;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > 0xFF000000 ? 0x7E00 : nonsign);
}

float dequantize_scalar_fp16(uint16_t bits) {
    const uint32_t w = (uint32_t) bits << 16;
    const uint32_t sign = w & 0x80000000;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0 << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = decode_scalar_fp32((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = 0x7E000000;
    const float magic_bias = 0.5f;
    const float denormalized_value = decode_scalar_fp32((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = 1 << 27;
    const uint32_t result = sign
                            | (two_w < denormalized_cutoff ? encode_scalar_fp32(denormalized_value)
                                                           : encode_scalar_fp32(normalized_value));
    return decode_scalar_fp32(result);
}

// Google brain floating-point format
uint16_t quantize_scalar_bf16(float value) {
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

float dequantize_scalar_bf16(uint16_t bits) {
    FloatBits raw;
    raw.bits = (uint32_t) bits << 16;
    return raw.value;
}

// 8-bit quantization with residual baking
Q8 quantize_scalar_q8(float value) {
    Q8 q8;

    // Define integer domain
    uint32_t z_domain = 255;
    // Reflect and compute effective real domain
    float r_domain = fabsf(value);

    // Special case for zero
    if (r_domain == 0.0f) {
        q8.scalar = quantize_scalar_fp16(1.0f);
        q8.bits = 0;
        return q8;
    }

    // Calculate squeezing ratio
    float alpha = (r_domain > z_domain) ? z_domain / r_domain : 1.0f;
    // Calculate the base step size
    float step_size = r_domain / z_domain; // Decoupled from scalar
    // Quantize the value using the base step size (exponent/mantissa?)
    uint8_t bits = roundf(value / step_size);
    // Calculate the residual precision (bias?)
    float residual = (value - (bits * step_size));

    // Calculate the scalar based on the squeezed range
    float scalar = step_size * alpha + residual;

    // Quantize to scalar to half-precision
    q8.scalar = quantize_scalar_fp16(scalar);
    // Quantize the value
    q8.bits = bits;

    return q8;
}

// Dequantize the value
float dequantize_scalar_q8(Q8 q8) {
    // Dequantize the scalar value
    float scalar = dequantize_scalar_fp16(q8.scalar);
    // Scale the bits proportionally back to original float
    return (float) (q8.bits * scalar);
}

// Generic interface

// Supports 32, 16, and 8-bit formats. Q4 is excluded.
bool quantize_scalar(float input, void* output, DataTypeId id) {
    switch (id) {
        case TYPE_FLOAT32:
            {
                uint32_t* out = TYPE_CAST_SAFE(output, uint32_t, sizeof(uint32_t));
                *out = encode_scalar_fp32(input);
                break;
            }
        case TYPE_FLOAT16:
            {
                uint16_t* out = TYPE_CAST_SAFE(output, uint16_t, sizeof(uint16_t));
                *out = quantize_scalar_fp16(input);
                break;
            }
        case TYPE_BFLOAT16:
            {
                uint16_t* out = TYPE_CAST_SAFE(output, uint16_t, sizeof(uint16_t));
                *out = quantize_scalar_bf16(input);
                break;
            }
        case TYPE_QUANT8:
            {
                Q8* out = TYPE_CAST_SAFE(output, Q8, sizeof(Q8));
                *out = quantize_scalar_q8(input);
                break;
            }
        default:
            return false;
    }
    return true;
}

bool dequantize_scalar(const void* input, float* output, DataTypeId id) {
    float* out = (float*) output;
    switch (id) {
        case TYPE_FLOAT32:
            *out = decode_scalar_fp32(*(uint32_t*) input);
            break;
        case TYPE_FLOAT16:
            *out = dequantize_scalar_fp16(*(uint16_t*) input);
            break;
        case TYPE_BFLOAT16:
            *out = dequantize_scalar_bf16(*(uint16_t*) input);
            break;
        case TYPE_QUANT8:
            *out = dequantize_scalar_q8(*(Q8*) input);
            break;
        default:
            return false;
    }
    return true;
}

// Vector Conversions (1D arrays)

// Half-precision floating-point quantization
void quantize_row_fp16(const float* input, uint16_t* output, size_t length) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);

    for (size_t i = 0; i < length; i++) {
        output[i] = quantize_scalar_fp16(input[i]);
    }
}

void dequantize_row_fp16(const uint16_t* input, float* output, size_t length) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);

    for (size_t i = 0; i < length; i++) {
        output[i] = dequantize_scalar_fp16(input[i]);
    }
}

// Google brain floating-point quantization
void quantize_row_bf16(const float* input, uint16_t* output, size_t length) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);

    for (size_t i = 0; i < length; i++) {
        output[i] = quantize_scalar_bf16(input[i]);
    }
}

void dequantize_row_bf16(const uint16_t* input, float* output, size_t length) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);

    for (size_t i = 0; i < length; i++) {
        output[i] = dequantize_scalar_bf16(input[i]);
    }
}

// 8-bit integer quantization
void quantize_row_q8(const float* input, Q8Row output, size_t length) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);

    for (size_t i = 0; i < length; i++) {
        output[i] = quantize_scalar_q8(input[i]);
    }
}

void dequantize_row_q8(const Q8Row input, float* output, size_t length) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);

    for (size_t i = 0; i < length; i++) {
        output[i] = dequantize_scalar_q8(input[i]);
    }
}

// Generic interface

// Supports 32, 16, and 8-bit formats. Q4 is excluded.
bool quantize_row(const float* input, void* output, size_t length, DataTypeId id) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);
    assert(id < TYPE_COUNT);

    size_t stride = data_type_size(id);
    assert(stride > 0);

    for (size_t i = 0; i < length; ++i) {
        void* dst = (uint8_t*) output + i * stride;
        if (!quantize_scalar(input[i], dst, id)) {
            return false;
        }
    }
    return true;
}

bool dequantize_row(const void* input, float* output, size_t length, DataTypeId id) {
    assert(input != NULL);
    assert(output != NULL);
    assert(length > 0);
    assert(id < TYPE_COUNT);

    size_t stride = data_type_size(id);
    assert(stride > 0);

    for (size_t i = 0; i < length; ++i) {
        const void* src = (const uint8_t*) input + i * stride;
        if (!dequantize_scalar(src, &output[i], id)) {
            return false;
        }
    }
    return true;
}
