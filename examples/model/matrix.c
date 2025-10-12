/**
 * @file examples/model/matrix.c
 */

#include <stdlib.h>
#include <stdio.h>
#include "core/lehmer.h"
#include "core/type.h"
#include "model/matrix.h"

int main(void) {
    lehmer_init(1337);

    const TypeId dtype = TYPE_F32;
    const size_t rows = 3, cols = 4;
    float* W = mat_new(rows, cols, dtype);
    mat_xavier(W, rows, cols, dtype);

    // input vector
    float* x = calloc(cols, sizeof(float));
    for (size_t j = 0; j < cols; ++j) {
        x[j] = lehmer_float();
    }

    // output
    float* y = calloc(rows, sizeof(float));
    mat_mul(y, W, x, rows, cols, dtype);

    printf("Matrix (W):\n");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("% .5f ", (double) W[i * cols + j]);
        }
        printf("\n");
    }

    printf("\nVector (x):\n");
    for (size_t j = 0; j < cols; ++j) {
        printf("% .5f ", (double) x[j]);
    }
    printf("\n\nResult (y = W * x):\n");
    for (size_t i = 0; i < rows; ++i) {
        printf("% .5f ", (double) y[i]);
    }
    printf("\n");

    free(W);
    free(x);
    free(y);
    return 0;
}
