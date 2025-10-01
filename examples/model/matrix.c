/**
 * @file examples/model/matrix.c
 */

#include <stdlib.h>
#include <stdio.h>
#include "core/lehmer.h"
#include "model/matrix.h"

int main(void) {
    lehmer_init(1337);

    DataTypeId dtype = TYPE_FLOAT32;
    unsigned dsize = data_type_size(dtype);
    const char* dname = data_type_name(dtype);
    printf("sizeof(%s) = %u\n", dname, dsize);

    unsigned rows = 3;
    unsigned cols = 4;
    float* W = mat_new(rows, cols, dtype);
    mat_xavier(W, rows, cols, dtype);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float w = W[i * cols + j];
            printf("% .6f ", (double) w);
        }
        printf("\n");
    }
    free(W);
    return 0;
}
