/**
 * @file scalar.h
 * @brief Bit-level conversion and storage of IEEE-754 and reduced-precision floating-point types.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This module provides type definitions and encode/decode functions for:
 *   - Standard 32-bit float (e8m23)
 *   - Half precision (e5m10)
 *   - Brain float (e8m7)
 *   - 8-bit float (e4m3)
 * All conversions are lossy except for e8m23, which is a direct reinterpretation.
 *
 * Special values (NaN, Inf) are preserved where possible, but may collapse to reserved patterns
 * in low-precision formats. Subnormals may be flushed to zero.
 *
 * Example:
 *   float32_t bits = e5m10_encode(0.42f);
 *   float val = e5m10_decode(bits);
 *
 * References:
 *   - https://en.wikipedia.org/wiki/IEEE_754
 *   - https://arxiv.org/abs/1710.03740
 *   - https://arxiv.org/abs/2209.05433
 *   - https://standards.ieee.org/ieee/754/6210/
 */

#ifndef SCALAR_H
#define SCALAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @union FloatUnion
 * @brief Utility for bitwise reinterpretation between float and uint32_t.
 */
typedef union FloatUnion {
    float v;  ///< IEEE-754 32-bit float value.
    uint32_t b;  ///< Bitwise representation as uint32_t.
} FloatUnion;

/* Semantic typedefs for compact float formats */
typedef uint32_t float32_t;  ///< Bit pattern of IEEE-754 32-bit float
typedef uint16_t float16_t;  ///< IEEE-754 half precision (e5m10)
typedef uint16_t bfloat16_t;  ///< Brain floating-point (e8m7)
typedef uint8_t float8_t;  ///< 8-bit float (e4m3)

/**
 * @name Floating-Point Conversions
 * @{
 */

/**
 * @brief Encode a float (32-bit) as IEEE-754 bit pattern (e8m23).
 * @param v Input value (float).
 * @return Bitwise representation (uint32_t).
 *
 * This is a lossless encode (just bit casting).
 */
float32_t e8m23_encode(float v);

/**
 * @brief Decode IEEE-754 bit pattern to float (e8m23).
 * @param b Input bits (uint32_t).
 * @return Decoded float.
 */
float e8m23_decode(float32_t b);

/**
 * @brief Encode float as half-precision (e5m10).
 * @param v Input value (float).
 * @return Encoded half-precision (uint16_t).
 *
 * Lossy: Rounds and flushes subnormals/overflows as needed.
 */
float16_t e5m10_encode(float v);

/**
 * @brief Decode half-precision (e5m10) to float.
 * @param b Encoded bits (uint16_t).
 * @return Decoded float (approximate).
 */
float e5m10_decode(float16_t b);

/**
 * @brief Encode float as bfloat16 (e8m7).
 * @param v Input value (float).
 * @return Encoded bfloat16 (uint16_t).
 *
 * Lossy: Only 7 bits of mantissa retained.
 */
bfloat16_t e8m7_encode(float v);

/**
 * @brief Decode bfloat16 (e8m7) to float.
 * @param b Encoded bits (uint16_t).
 * @return Decoded float (approximate).
 */
float e8m7_decode(bfloat16_t b);

/**
 * @brief Encode float as 8-bit float (e4m3).
 * @param v Input value (float).
 * @return Encoded float8 (uint8_t).
 *
 * Lossy: 4-bit exponent, 3-bit mantissa. May overflow/underflow small/large values.
 */
float8_t e4m3_encode(float v);

/**
 * @brief Decode 8-bit float (e4m3) to float.
 * @param b Encoded bits (uint8_t).
 * @return Decoded float (approximate).
 */
float e4m3_decode(float8_t b);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // SCALAR_H
