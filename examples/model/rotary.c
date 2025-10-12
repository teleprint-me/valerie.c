/**
 * @file examples/model/rotray.c
 * @brief Driver for precomputing RoPE frequencies.
 */

#include <stdio.h>
#include <math.h>
#include "core/type.h"
#include "model/matrix.h"

int main(void) {
    int d_model = 8;
    int n_heads = 2;
    int seq_len = 5;
    float theta = 10000.0f;

    // row-major: (rows, cols)
    int dim = d_model / n_heads;
    int rows = seq_len;
    int cols = dim / 2;
    float* cos = mat_new(rows, cols, TYPE_F32);
    float* sin = mat_new(rows, cols, TYPE_F32);

    // base freqs
    float* freqs = malloc(cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        freqs[i] = powf(theta, -(float) i / (float) d_model);
    }

    // outer product
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < cols; j++) {
            float angle = (float) i / (i * (1.0f / freqs[j]));
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
