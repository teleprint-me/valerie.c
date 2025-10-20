/**
 * @file q8.c
 * @brief Microscaling-style Q8 quantization using shared E4M3 block scales.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "core/type.h"

int main(void) {
    const size_t N = 32;
    const size_t B = 8;

    float x[N];
    for (size_t i = 0; i < N; i++) {
        x[i] = sinf((float) i * 0.25f) * 5.0f;
    }

    Q8 q8 = {
        .q = calloc(N, sizeof(int8_t)),
        .w = calloc(N / B, sizeof(uint8_t)),
    };

    q8_encode(&q8, x, N, B);

    float recon[N];
    q8_decode(recon, &q8, N, B);

    printf(" idx | original    quant  recon\n");
    printf("-----+----------------------------\n");
    for (size_t i = 0; i < N; i++) {
        printf("%4zu | %+10.5f  %4d  %+10.5f\n", i, (double) x[i], q8.q[i], (double) recon[i]);
    }

    free(q8.w);
    free(q8.q);
}
