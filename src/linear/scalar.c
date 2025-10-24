/**
 * @file scalar.h
 * @brief Bit-level conversion and storage of IEEE-754 and reduced-precision floating-point types.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdint.h>

#include "linear/scalar.h"

// e8m23
uint32_t e8m23_encode(float v) {
    FloatUnion u = {.v = v};  // map and align float to unsigned int
    return u.b;  // type pun from float to unsigned int
}

float e8m23_decode(uint32_t b) {
    FloatUnion u = {.b = b};  // map and align unsigned int to float
    return u.v;  // type pun from unsigned int to float
}

// e5m10
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
