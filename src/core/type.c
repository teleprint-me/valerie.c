/**
 * @file type.c
 * @brief Core numeric type definitions and quantization interface.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "core/type.h"

/**
 * Metadata Accessors
 * @{
 */

const Type* type_data(TypeId id) {
    return id < TYPE_COUNT ? &TYPE_DATA[id] : NULL;
}

const char* type_name(TypeId id) {
    const Type* type = type_data(id);
    return type ? type->name : "unknown";
}

uint32_t type_size(TypeId id) {
    const Type* type = type_data(id);
    return type ? type->size : 0;
}

/** @} */

/**
 * Floating-Point Conversions
 * @{
 */

// e8m23
uint32_t e8m23_encode(float v) {
    Float32Union u = {.v = v};  // map and align float to unsigned int
    return u.b;  // type pun from float to unsigned int
}

float e8m23_decode(uint32_t b) {
    Float32Union u = {.b = b};  // map and align unsigned int to float
    return u.v;  // type pun from unsigned int to float
}

// e5m10 (23 - 13 = 10)
uint16_t e5m10_encode(float v) {
    uint32_t b = e8m23_encode(v);
    uint32_t sign = (b >> 31) & 0x1;
    uint32_t exponent = (b >> 23) & 0xFF;
    uint32_t mantissa = b & 0x7FFFFF;

    // Zero
    if (!exponent && !mantissa) {
        return (sign << 15);  // ±0
    }

    // Subnormal
    if (!exponent && mantissa) {
        return (sign << 15) | (mantissa >> 13);  // flush
    }

    // Inf
    if (exponent == 0xFF && !mantissa) {
        // exponent is filled with ones
        return (sign << 15) | 0x7C00;  // ±inf
    }

    // NaN
    if (exponent == 0xFF && mantissa) {
        // exponent is filled with ones, mantissa has a leading 1.
        return (sign << 15) | 0x7E00;  // nan
    }

    // Normal
    int32_t rebias = (int32_t) exponent - 127 + 15;

    // Clamp to min
    if (rebias < 0) {
        rebias = 0;  // underflow
    }

    // Clamp to max
    if ((uint32_t) rebias > 0x1F) {
        rebias = 0x1F;  // overflow
    }

    // Rebase
    return (sign << 15) | ((uint32_t) rebias << 10) | (mantissa >> 13);
}

float e5m10_decode(uint16_t b) {
    uint32_t sign = (b >> 15) & 0x1;
    uint32_t exponent = (b >> 10) & 0x1F;  // 5-bit exponent
    uint32_t mantissa = b & 0x3FF;  // 10-bit mantissa

    // Zero
    if (!exponent && !mantissa) {
        return e8m23_decode((sign << 31));  // ±0
    }

    // Subnormal
    if (!exponent && mantissa) {
        return e8m23_decode((sign << 31) | (mantissa << 13));  // flush
    }

    // Inf
    if (exponent == 0x1F && !mantissa) {
        // exponent is filled with ones
        return e8m23_decode((sign << 31) | 0x7F800000);  // ±inf
    }

    // NaN
    if (exponent == 0x1F && mantissa) {
        // exponent is filled with ones, mantissa has a leading 1.
        return e8m23_decode((sign << 31) | 0x7FC00000);  // nan
    }

    // Normal
    int32_t rebias = (int32_t) exponent - 15 + 127;

    // Clamp to min
    if (rebias < 0) {
        rebias = 0;  // underflow
    }

    // Clamp to max
    if (rebias > 0xFF) {
        rebias = 0xFF;  // overflow
    }

    // Rebase
    return e8m23_decode((sign << 31) | ((uint32_t) rebias << 23) | (mantissa << 13));
}

// e8m7 (fused multiply-add)
uint16_t e8m7_encode(float v) {
    uint32_t b = e8m23_encode(v);
    uint32_t sign = (b >> 31) & 0x1;
    uint32_t exponent = (b >> 23) & 0xFF;
    uint32_t mantissa = b & 0x7FFFFF;

    // Zero
    if (!exponent && !mantissa) {
        return (sign << 15);  // ±0
    }

    // Subnormal
    if (!exponent && mantissa) {
        return (sign << 15) | (mantissa >> 16);  // flush
    }

    // Inf
    if (exponent == 0xFF && !mantissa) {
        // exponent is filled with ones
        return (sign << 15) | 0x7F80;  // ±inf
    }

    // NaN
    if (exponent == 0xFF && mantissa) {
        // exponent is filled with ones, mantissa has a leading 1.
        return (sign << 15) | 0x7FC0;  // nan
    }

    return b >> 16;
}

float e8m7_decode(uint16_t b) {
    uint32_t sign = (b >> 15) & 0x1;
    uint32_t exponent = (b >> 7) & 0xFF;  // 8-bit exponent
    uint32_t mantissa = b & 0x7F;  // 7-bit mantissa

    // Zero
    if (!exponent && !mantissa) {
        return e8m23_decode((sign << 31));  // ±0
    }

    // Subnormal
    if (!exponent && mantissa) {
        return e8m23_decode((sign << 31) | (mantissa << 16));  // flush
    }

    // Inf
    if (exponent == 0xFF && !mantissa) {
        // exponent is filled with ones
        return e8m23_decode((sign << 31) | 0x7F800000);  // ±inf
    }

    // NaN
    if (exponent == 0xFF && mantissa) {
        // exponent is filled with ones, mantissa has a leading 1.
        return e8m23_decode((sign << 31) | 0x7FC00000);  // nan
    }

    return e8m23_decode(((uint32_t) b) << 16);
}

// e4m3
uint8_t e4m3_encode(float v) {
    uint32_t b = e8m23_encode(v);
    uint32_t sign = (b >> 31) & 0x1;
    uint32_t exponent = (b >> 23) & 0xFF;
    uint32_t mantissa = b & 0x7FFFFF;

    // Zero
    if (!exponent && !mantissa) {
        return (sign << 7);
    }

    // Subnormal
    if (!exponent && mantissa) {
        return (sign << 7) | (mantissa >> 20);
    }

    // Inf
    if (exponent == 0xFF && !mantissa) {
        return (sign << 7) | 0x78;  // exp=1111, mant=000
    }

    // NaN
    if (exponent == 0xFF && mantissa) {
        return (sign << 7) | 0x7F;  // exp=1111, mant=111
    }

    // Normalized
    int32_t rebias = (int32_t) exponent - 127 + 7;

    // Clamp to min
    if (rebias < 0) {
        rebias = 0;  // Underflow
    }

    // Overflow
    if (rebias > 0xF) {
        return (sign << 7) | 0x78;  // inf
    }

    return (sign << 7) | ((uint32_t) rebias << 3) | (mantissa >> 20);
}

float e4m3_decode(uint8_t b) {
    uint32_t sign = (b >> 7) & 0x1;
    uint32_t exponent = (b >> 3) & 0xF;
    uint32_t mantissa = b & 0x7;

    // Zero
    if (!exponent && !mantissa) {
        return e8m23_decode(sign << 31);
    }

    // Subnormal
    if (!exponent && mantissa) {
        return e8m23_decode((sign << 31) | (mantissa << 20));
    }

    // Inf
    if (exponent == 0xF && mantissa == 0) {
        return e8m23_decode((sign << 31) | 0x7F800000);
    }

    // NaN
    if (exponent == 0xF && mantissa != 0) {
        return e8m23_decode((sign << 31) | 0x7FC00000);
    }

    // Normal
    int32_t rebias = (int32_t) exponent - 7 + 127;

    if (rebias < 0) {
        rebias = 0;
    }

    if (rebias > 0xFF) {
        rebias = 0xFF;
    }

    return e8m23_decode((sign << 31) | ((uint32_t) rebias << 23) | (mantissa << 20));
}

/** @} */

/**
 * Block Quantization
 * @{
 */

void q8_encode(Q8* dst, const float* src, size_t n, size_t block_size) {
    const int8_t limit = 127;
    const float eps = 1e-6f;

    const size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        const size_t offset = b * block_size;
        const size_t len = (offset + block_size <= n) ? block_size : (n - offset);

        const float* xb = src + offset;
        int8_t* qb = dst->q + offset;

        // Find max magnitude
        float max_abs = 0.0f;
        for (size_t i = 0; i < len; i++) {
            if (fabsf(xb[i]) > max_abs) {
                max_abs = fabsf(xb[i]);
            }
        }

        // Shared scale (linear variant)
        float scale = (max_abs < eps) ? 1.0f : (max_abs / (float) limit);

        // Encode scale as E4M3 (shared per block)
        dst->s[b] = e4m3_encode(scale);

        // Quantize
        for (size_t i = 0; i < len; i++) {
            float q = xb[i] / scale;
            qb[i] = (int8_t) fminf(fmaxf(roundf(q), -limit), limit);
        }
    }
}

void q8_decode(float* dst, const Q8* src, size_t n, size_t block_size) {
    const size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        const size_t offset = b * block_size;
        const size_t len = (offset + block_size <= n) ? block_size : (n - offset);

        const float scale = e4m3_decode(src->s[b]);
        const int8_t* qb = src->q + offset;
        float* xb = dst + offset;

        for (size_t i = 0; i < len; i++) {
            xb[i] = qb[i] * scale;
        }
    }
}

/** @} */

/**
 * Quantization Interface
 * @{
 */

/**
 * Scalar conversions
 */

bool quant(void* dst, float src, TypeId dst_id) {
    switch (dst_id) {
        case TYPE_F32:
            *(float*) dst = src;
            break;
        case TYPE_E8M23:
            *(uint32_t*) dst = e8m23_encode(src);
            break;
        case TYPE_E5M10:
            *(uint16_t*) dst = e5m10_encode(src);
            break;
        case TYPE_E8M7:
            *(uint16_t*) dst = e8m7_encode(src);
            break;
        case TYPE_E4M3:
            *(uint8_t*) dst = e4m3_encode(src);
            break;
        default:
            return false;
    }
    return true;
}

bool dequant(float* dst, const void* src, TypeId src_id) {
    switch (src_id) {
        case TYPE_F32:
            *dst = *(const float*) src;
            break;
        case TYPE_E8M23:
            *dst = e8m23_decode(*(const uint32_t*) src);
            break;
        case TYPE_E5M10:
            *dst = e5m10_decode(*(const uint16_t*) src);
            break;
        case TYPE_E8M7:
            *dst = e8m7_decode(*(const uint16_t*) src);
            break;
        case TYPE_E4M3:
            *dst = e4m3_decode(*(const uint8_t*) src);
            break;
        default:
            return false;
    }
    return true;
}

/**
 * Vector conversions (1D arrays)
 */

bool quant_vec(void* dst, const float* src, size_t len, TypeId dst_id) {
    assert(dst && src && len > 0);

    switch (dst_id) {
        case TYPE_Q8: {
            Q8* q8 = (Q8*) dst;
            q8_encode(q8, src, len, Q8_BLOCK_SIZE);
            return true;
        }
        default: {
            size_t stride = type_size(dst_id);
            assert(stride > 0);
            for (size_t i = 0; i < len; ++i) {
                void* dst_elem = (uint8_t*) dst + i * stride;
                if (!quant(dst_elem, src[i], dst_id)) {
                    return false;
                }
            }
            return true;
        }
    }
}

bool dequant_vec(float* dst, const void* src, size_t len, TypeId src_id) {
    assert(dst && src && len > 0);

    switch (src_id) {
        case TYPE_Q8: {
            const Q8* q8 = (const Q8*) src;
            q8_decode(dst, q8, len, Q8_BLOCK_SIZE);
            return true;
        }
        default: {
            size_t stride = type_size(src_id);
            assert(stride > 0);
            for (size_t i = 0; i < len; ++i) {
                const void* src_elem = (const uint8_t*) src + i * stride;
                if (!dequant(&dst[i], src_elem, src_id)) {
                    return false;
                }
            }
            return true;
        }
    }
}

/**
 * Matrix conversions (2D flat arrays)
 */

bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, TypeId dst_id) {
    assert(dst != NULL && src != NULL && rows * cols > 0);
    assert(dst_id < TYPE_COUNT);

    return quant_vec(dst, src, rows * cols, dst_id);
}

bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, TypeId src_id) {
    assert(dst != NULL && src != NULL && rows * cols > 0);
    assert(src_id < TYPE_COUNT);

    return dequant_vec(dst, src, rows * cols, src_id);
}

/** @} */
