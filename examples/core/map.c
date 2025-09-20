
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/map.h"

int main(void) {
    HashMap* map = hash_map_create(1, HASH_STR);
    char* k = strdup("test");
    int* v = malloc(sizeof(int));

    if (!map) {
        fprintf(stderr, "Failed to create hash map.\n");
        goto fail;
    }

    if (!k) {
        fprintf(stderr, "Failed to allocate key.\n");
        goto fail;
    }

    if (!v) {
        fprintf(stderr, "Failed to allocate value.\n");
        goto fail;
    }

    *v = 42;

    HashState state = hash_map_insert(map, k, v);  // crashes here
    if (state != HASH_SUCCESS) {
        fprintf(stderr, "Failed to insert kv pair into map.\n");
        goto fail;
    }

    free(k);
    free(v);
    hash_map_free(map);

    return EXIT_SUCCESS;

fail:
    free(k);
    free(v);
    hash_map_free(map);

    return EXIT_FAILURE;
}
