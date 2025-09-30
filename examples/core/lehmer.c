/**
 * @file examples/core/lehmer.c
 */

#include "core/lehmer.h"

#include <stdio.h>

int main(void) {
    lehmer_init(1337);

    printf("lehmer 32-bit int:\n");
    for (unsigned i = 0; i < 20; i++) {
        printf("%d -> %d\n", i, lehmer_int32());
    }
    printf("\n");

    printf("lehmer 32-bit float:\n");
    for (unsigned i = 0; i < 20; i++) {
        printf("%d -> % .6f\n", i, (double) lehmer_float());
    }

    return 0;
}
