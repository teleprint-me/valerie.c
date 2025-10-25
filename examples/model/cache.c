/**
 * @file examples/model/cache.c
 */

#include <stdlib.h>
#include <stdio.h>
#include "linear/lehmer.h"

int main(void) {
    lehmer_init(42);

    int seq_len = 4;
    int kv_dim = 8;

    float* K = calloc(seq_len * kv_dim, sizeof(float));
    for (int i = 0; i < seq_len * kv_dim; i++) {
        K[i] = lehmer_xavier(seq_len, kv_dim);
    }

    printf("keys matrix:\n");
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < kv_dim; ++j) {
            printf("% .5f ", (double) K[i * kv_dim + j]);
        }
        printf("\n");
    }
    printf("\n");

    // position 2 (3rd row)
    int pos = 2;
    float* K_cache = K + pos * kv_dim;

    printf("Aliased cache row (pos = %d):\n", pos);
    for (int j = 0; j < kv_dim; ++j) {
        printf("%d -> % .5f\n", j, (double) K_cache[j]);
    }

    // Write through the alias (simulate update)
    for (int j = 0; j < kv_dim; ++j) {
        K_cache[j] = (float) j * 0.1f;
    }

    printf("\nUpdated base matrix (after writing through alias):\n");
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < kv_dim; ++j) {
            printf("% .5f ", (double) K[i * kv_dim + j]);
        }
        printf("\n");
    }

    free(K);
    return 0;
}
