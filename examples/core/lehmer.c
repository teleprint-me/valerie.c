/**
 * @file examples/core/lehmer.c
 */

#include <stdio.h>
#include <omp.h>
#include "core/lehmer.h"

#define LIMIT 10

int main(void) {
    lehmer_init(1337);

    printf("lehmer 32-bit int:\n");
    for (unsigned i = 0; i < LIMIT; i++) {
        printf("%d -> %d\n", i, lehmer_int32());
    }
    printf("\n");

    printf("lehmer 32-bit float:\n");
    for (unsigned i = 0; i < LIMIT; i++) {
        printf("%d -> % .6f\n", i, (double) lehmer_float());
    }
    printf("\n");

    lehmer_init(1337);

    printf("lehmer 32-bit int (OpenMP):\n");
#pragma omp parallel for
    for (unsigned i = 0; i < LIMIT; i++) {
        printf("%d -> %d\n", i, lehmer_int32());
    }
    printf("\n");

    printf("lehmer 32-bit float (OpenMP):\n");
#pragma omp parallel for
    for (unsigned i = 0; i < LIMIT; i++) {
        printf("%d -> % .6f\n", i, (double) lehmer_float());
    }

    return 0;
}
