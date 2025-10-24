/**
 * @file bin/micro.c
 * @brief Microscaling Floating Point Formats for Large Language Models.
 * @ref https://arxiv.org/abs/2510.01863
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <math.h>

#include "linear/lehmer.h"
#include "linear/q8.h"

int main(void) {
    lehmer_init(42);

    // Emulate a quant sequence
    const size_t length = 16;  // input sequence vector

    // Create and init input vector
    float x[length];
    for (size_t i = 0; i < length; i++) {
        x[i] = lehmer_float() * sinf(i + 1 * 0.25f) * 5.0f;
    }

    // Encode input vector
    quant8_t q8 = q8_vec_new(length);
    q8_vec_encode(&q8, x, length);

    // Decode input vector
    float y[length];
    q8_vec_decode(y, &q8, length);

    printf(" idx |    x    q    w    y    e\n");
    printf("-----+----------------------------\n");
    for (size_t i = 0; i < length; i++) {
        size_t b = q8_block(i);
        float err = fabsf(x[i] - y[i]);
        printf(
            "%4zu | %+10.5f  %4d  %4d  %+10.5f  %+10.5f\n",
            i,
            (double) x[i],
            q8.q[i],
            q8.w[b],
            (double) y[i],
            (double) err
        );
    }

    float max_err = 0.0f, mae = 0.0f;
    for (size_t i = 0; i < length; i++) {
        float err = fabsf(x[i] - y[i]);
        if (err > max_err) {
            max_err = err;
        }
        mae += err;
    }
    mae /= length;
    printf("Max error: %g, Mean absolute error: %g\n", (double) max_err, (double) mae);

    // Clean up
    q8_vec_free(&q8);
    return 0;
}
