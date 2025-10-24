/**
 * @file quant.c
 * @brief Unified quantization interface for scalar, vector, and matrix types.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <assert.h>
#include "linear/scalar.h"
#include "linear/q8.h"
#include "linear/type.h"
#include "linear/quant.h"

// --- Scalar conversions ---
bool quant(void* dst, float src, TypeId dst_id) {
    assert(dst_id < TYPE_COUNT);
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
    assert(src_id < TYPE_COUNT);
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

// --- Vector conversions ---
// dst: output buffer (Q8: quant8_t*, others: array of scalars)
bool quant_vec(void* dst, const float* src, size_t len, TypeId dst_id) {
    assert(dst && src && len > 0);
    assert(dst_id < TYPE_COUNT);
    switch (dst_id) {
        case TYPE_Q8:
            q8_vec_encode((quant8_t*) dst, src, len);
            return true;
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

// dst: output array of floats (length = len)
bool dequant_vec(float* dst, const void* src, size_t len, TypeId src_id) {
    assert(dst && src && len > 0);
    assert(src_id < TYPE_COUNT);
    switch (src_id) {
        case TYPE_Q8:
            q8_vec_decode(dst, (const quant8_t*) src, len);
            return true;
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

// --- Matrix conversions ---
// dst: output buffer (Q8: quant8_t*, others: array of scalars, flat row-major)
bool quant_mat(void* dst, const float* src, size_t rows, size_t cols, TypeId dst_id) {
    assert(dst && src && rows > 0 && cols > 0);
    assert(dst_id < TYPE_COUNT);
    switch (dst_id) {
        case TYPE_Q8:
            q8_mat_encode((quant8_t*) dst, src, rows, cols);
            return true;
        default:
            // flat row-major matrix of non-Q8 type
            return quant_vec(dst, src, rows * cols, dst_id);
    }
}

// dst: output array of floats (flat row-major: rows * cols)
bool dequant_mat(float* dst, const void* src, size_t rows, size_t cols, TypeId src_id) {
    assert(dst && src && rows > 0 && cols > 0);
    assert(src_id < TYPE_COUNT);
    switch (src_id) {
        case TYPE_Q8:
            q8_mat_decode(dst, (const quant8_t*) src, rows, cols);
            return true;
        default:
            return dequant_vec(dst, src, rows * cols, src_id);
    }
}
