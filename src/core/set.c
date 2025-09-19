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

    void* data; /**< Array of unique elements. */
    size_t count; /**< Current number of elements in the set. */
    size_t capacity; /**< Total capacity of the set. */
} HashSet;

/**
 * Set Lifecycle
 * @{
 */

HashSet* hash_set_create(uint64_t initial_capacity, HashType type) {
    HashSet* set = malloc(sizeof(HashSet));
    if (!set) {
        LOG_ERROR("Failed to allocate memory for HashSet.");
        return NULL;
    }

    set->count = 0;
    set->capacity = initial_capacity > 0 ? initial_capacity : 10;
    set->hash.type = type;

    switch (set->hash.type) {
        case HASH_INT32:
            set->hash.function = hash_int32;
            set->hash.compare = hash_int32_cmp;
            set->hash.size = sizeof(int32_t);
            break;
        case HASH_INT64:
            set->hash.function = hash_int64;
            set->hash.compare = hash_int64_cmp;
            set->hash.size = sizeof(int64_t);
            break;
        case HASH_PTR:
            set->hash.function = hash_ptr;
            set->hash.compare = hash_ptr_cmp;
            set->hash.size = sizeof(uintptr_t);
            break;
        case HASH_STR:
            set->hash.function = hash_str;
            set->hash.compare = hash_str_cmp;
            set->hash.size = sizeof(char);
            break;
        default:
            LOG_ERROR("Invalid HashType given.");
            free(set);
            return NULL;
    }

    set->data = calloc(set->capacity, set->hash.size);
    if (!set->data) {
        LOG_ERROR("Failed to allocate memory for HashMap data.");
        free(set);
        return NULL;
    }

    // Initialize the mutex for thread safety
    int error_code = pthread_mutex_init(&set->thread_lock, NULL);
    if (0 != error_code) {
        LOG_ERROR("Failed to initialize mutex with error: %d", error_code);
        free(set->data);
        free(set);
        return NULL;
    }

    return set;
}

void hash_set_free(HashSet* set) {
    if (set) {
        // Destroy the mutex before freeing memory
        pthread_mutex_destroy(&set->thread_lock);

        if (set->data) {
            free(set->data);
        }

        free(set);
        set = NULL;
    }
}

/** @} */

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
