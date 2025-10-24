/**
 * @file examples/model/rotray.c
 * @brief Driver for precomputing RoPE frequencies.
 */

#include <stdio.h>
#include <math.h>
#include "linear/type.h"
#include "model/matrix.h"

int main(void) {
    int d_model = 16;
    int n_heads = 2;
    int seq_len = 5;
    float theta = 10000.0f;

    int dim = d_model / n_heads;  // per-head dimension
    int rows = seq_len;
    int cols = dim / 2;

    float* cos = mat_new(rows, cols, TYPE_F32);
    float* sin = mat_new(rows, cols, TYPE_F32);
    float* freqs = malloc(cols * sizeof(float));

    // base frequencies
    for (int j = 0; j < cols; j++) {
        // 1 / (theta ** (j / dim))
        freqs[j] = 1.0f / powf(theta, (float) j / (float) dim);
    }

    // outer product
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float angle = (float) i * freqs[j];
            cos[i * cols + j] = cosf(angle);
            sin[i * cols + j] = sinf(angle);
        }
    }

    printf("cosine:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("% .5f ", (double) cos[i * cols + j]);
        }
        printf("\n");
    }

    printf("sine:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("% .5f ", (double) sin[i * cols + j]);
        }
        printf("\n");
    }

    free(freqs);
    mat_free(cos, TYPE_F32);
    mat_free(sin, TYPE_F32);
    return 0;
}
