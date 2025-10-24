/**
 * @file bin/matrix.c
 */

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "linear/lehmer.h"
#include "linear/q8.h"
#include "linear/type.h"
#include "linear/quant.h"

int main(void) {
    lehmer_init(42);

    int rows = 4;
    int cols = 8;

    // --- Create and quantize input ---

    // create a vector
    float* x = calloc(cols, sizeof(float));
    // populate the vector
    for (int i = 0; i < cols; i++) {
        x[i] = lehmer_float();
    }
    quant8_t xq = q8_vec_new(cols);
    q8_vec_encode(&xq, x, cols);

    // print the vector
    printf("x (before quant8):\n");
    for (int i = 0; i < cols; i++) {
        printf("% .5f ", (double) x[i]);
    }
    printf("\n");

    // --- Create and quantize weight ---

    // create a flat row-major matrix
    float* W = calloc(rows * cols, sizeof(float));  // in buffer
    // populate the matrix
    for (int i = 0; i < rows * cols; i++) {
        W[i] = lehmer_float();
    }

    // quantize input weight by row
    quant8_t* Wq = q8_mat_new(rows, cols);
    q8_mat_encode(Wq, W, rows, cols);

    // print the matrix
    printf("W (before quant8):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("% .5f ", (double) W[i * cols + j]);
        }
        printf("\n");
    }

    // --- Matrix multiplication: y = W * x ---

    // Create output buffer
    float* y = calloc(rows, sizeof(float));

    // Decode the input vector
    float* v = calloc(cols, sizeof(float));
    // Dequanitze x once
    dequant_vec(v, &xq, cols, TYPE_Q8);

    // Intermediate buffer for dequant matrix
    float* w = calloc(cols, sizeof(float));  // out buffer

    // Calculate mat mul
    for (int r = 0; r < rows; r++) {
        // dequantize quant rows back into new matrix
        dequant_vec(w, &Wq[r], cols, TYPE_Q8);

        // Calculate dot product
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += w[c] * v[c];
        }

        // Update accumulated sum
        y[r] = sum;
    }

    // Print dequantized output y
    printf("y = W * x (after quant8):\n");
    for (int r = 0; r < rows; r++) {
        printf("% .5f\n", (double) y[r]);
    }
    printf("\n");

    // --- Clean up ---

    // inputs
    free(x);
    q8_vec_free(&xq);

    // weights
    q8_mat_free(Wq, rows);
    free(W);

    // buffers
    free(w);
    free(y);
    free(v);
    return 0;
}
