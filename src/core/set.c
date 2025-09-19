/// @file set.c
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/logger.h"
#include "core/hash.h"

typedef struct HashSet {
    Hash hash; /** < Bundles a key type and size with hash and compare functions. */

    pthread_mutex_t thread_lock; /**< Mutex for thread safety. */

    void* data; /**< Array of keys. */
    size_t count; /**< Current number of entries in the set. */
    size_t capacity; /**< Total capacity of the set. */
} HashSet;

bool hash_set_is_valid(HashSet* set) {
    return set && set->data && set->capacity > 0;
}

uint64_t hash_set_index(HashSet* set, const void* key, size_t i) {
    return set->hash.function(key, set->capacity, i);
}

uint8_t* hash_set_value(HashSet* set, const void* key, size_t i) {
    uint64_t index = hash_set_index(set, key, i);
    return (uint8_t*) set->data + (index * set->hash.size);
}

/**
 * HashSet Search
 * @{
 */

void* hash_set_search_internal(HashSet* set, const void* key) {
    if (!hash_set_is_valid(set)) {
        LOG_ERROR("Invalid set for search internal.");
        return NULL;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return NULL;
    }

    for (size_t i = 0; i < set->capacity; i++) {
        void* v = hash_set_value(set, key, i);

        if (!v) {
            return NULL;
        }

        if (0 == set->hash.compare(v, key)) {
            return v;
        }
    }

    return NULL;
}

int32_t* hash_set_int32_search(HashSet* set, const void* key) {
    return (int32_t*) hash_set_search_internal(set, key);
}

int64_t* hash_set_int64_search(HashSet* set, const void* key) {
    return (int64_t*) hash_set_search_internal(set, key);
}

uintptr_t hash_set_ptr_search(HashSet* set, const void* key) {
    return (uintptr_t) hash_set_search_internal(set, key);
}

char* hash_set_str_search(HashSet* set, const void* key) {
    return (char*) hash_set_search_internal(set, key);
}

/** @} */
