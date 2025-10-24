/**
 * @file examples/core/alias.c
 */

#include "linear/lehmer.h"

#include <stdlib.h>
#include <stdio.h>

int main(void) {
    lehmer_init(42);

    float* x = calloc(10, sizeof(float));

    printf("x:\n");
    for (int i = 0; i < 10; i++) {
        x[i] = lehmer_float();
        printf("%d -> %1.3f\n", i, (double) x[i]);
    }
    printf("\n");

    float* y = x + 5;
    
    printf("y (aliased to x):\n");
    for (int i = 0; i < 5; i++) {
        printf("%d -> %1.3f\n", i, (double) y[i]);
    }

    free(x);
    return 0;
}
