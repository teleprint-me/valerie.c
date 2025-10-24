/**
 * @section e4m3 (8-bit float)
 * @brief IEEE-like FP8 variant with 4 exponent bits and 3 mantissa bits.
 *
 *   Sign: 1 bit
 *   Exponent: 4 bits (bias = 7)
 *   Mantissa: 3 bits
 *
 * This format matches the NVIDIA/ARM FP8 E4M3 representation used for
 * activations and forward-pass operations. It prioritizes numerical density
 * over dynamic range.
 *
 * Effective numeric range: ±[2^-6, 2^8) ≈ ±[0.015625, 448.0)
 * Relative precision ≈ 12.5% per step.
 *
 * NaN: exp = 0b1111, mant != 0
 * Inf: exp = 0b1111, mant = 0
 *
 * Reference:
 *   - NVIDIA Hopper Tensor Core FP8
 *   - "8-bit Floating Point Formats for Deep Learning" (Micikevicius et al.)
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "linear/scalar.h"

int main(void) {
    srand(42);

    const size_t n = 8;
    float x[n];
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float) rand() / (float) RAND_MAX) * sinf((float) i * 0.25f);
    }

    printf("E4M3 Encoding Demonstration\n");
    printf("----------------------------\n");

    for (size_t i = 0; i < n; i++) {
        uint8_t e = e4m3_encode(x[i]);
        float d = e4m3_decode(e);
        printf("%5.5f -> 0x%02X -> %5.3f\n", (double) x[i], e, (double) d);
    }
    printf(
        "e4m3_encode(0.0076) = 0x%02X, decode = %f\n",
        e4m3_encode(0.0076f),
        (double) e4m3_decode(e4m3_encode(0.0076f))
    );

    printf("----------------------------\n");
    printf("Note: Values > ~448.0 saturate to INF.\n");
    return 0;
}
