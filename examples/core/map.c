#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/map.h"

#define MAP_CHECK(EXPR, MSG) \
    do { if (!(EXPR)) { \
        fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, MSG); \
        goto fail; \
    } } while (0)

int main(void) {
    HashMap* map = hash_map_create(2, HASH_STR);
    MAP_CHECK(map, "Failed to create map");

    char* k1 = strdup("alpha");
    int* v1 = malloc(sizeof(int)); if (v1) *v1 = 1;
    char* k2 = strdup("beta");
    int* v2 = malloc(sizeof(int)); if (v2) *v2 = 2;
    char* k3 = strdup("gamma");
    int* v3 = malloc(sizeof(int)); if (v3) *v3 = 3;

    MAP_CHECK(k1 && v1 && k2 && v2 && k3 && v3, "Out of memory");

    MAP_CHECK(hash_map_insert(map, k1, v1) == HASH_SUCCESS, "Insert k1/v1 failed");
    MAP_CHECK(hash_map_insert(map, k2, v2) == HASH_SUCCESS, "Insert k2/v2 failed");
    MAP_CHECK(hash_map_insert(map, k3, v3) == HASH_SUCCESS, "Insert k3/v3 failed");

    int* found = hash_map_search(map, "alpha");
    MAP_CHECK(found && *found == 1, "Search 'alpha' failed or wrong value");
    found = hash_map_search(map, "beta");
    MAP_CHECK(found && *found == 2, "Search 'beta' failed or wrong value");
    found = hash_map_search(map, "gamma");
    MAP_CHECK(found && *found == 3, "Search 'gamma' failed or wrong value");

    int* v1b = malloc(sizeof(int)); if (v1b) *v1b = 111;
    char* k_dup = strdup("alpha");
    MAP_CHECK(k_dup && v1b, "Out of memory (dup)");
    MAP_CHECK(hash_map_insert(map, k_dup, v1b) == HASH_EXISTS, "Duplicate insert should return EXISTS");
    free(k_dup); free(v1b);

    MAP_CHECK(hash_map_delete(map, "beta") == HASH_SUCCESS, "Delete 'beta' failed");
    found = hash_map_search(map, "beta");
    MAP_CHECK(!found, "Deleted key 'beta' still found");

    // Do NOT call hash_map_clear(map) here!
    // Instead, just free all entries and the map in one go:
    hash_iter_free_all(map, free, free);
    map = NULL;

    printf("[PASS] All core smoke tests succeeded.\n");
    return EXIT_SUCCESS;

fail:
    if (map) hash_iter_free_all(map, free, free);
    return EXIT_FAILURE;
}
