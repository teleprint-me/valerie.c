#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct Array {
    void* data;
    size_t size;
    size_t count;
    size_t capacity;
} Array;

Array* array_create(size_t n, size_t size) {
    Array* a = malloc(sizeof(Array));
    a->data = calloc(n, size);
    a->count = n;
    a->size = size;
    a->capacity = n * size;
    return a;
}

void array_realloc(Array* a, size_t n, size_t size) {
    return;
}

void array_free(Array* a) {
    if (a) {
        if (a->data) {
            free(a->data);
        }
        free(a);
    }
}

// insert

// append

// sort

int main(void) {
    return 0;
}
