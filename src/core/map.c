/**
 * Copyright © 2023 Austin Berrio
 *
 * @file map.c
 * @brief Minimalistic hash map implementation providing mapping between integers, strings, and
 * memory addresses.
 *
 * The Hash Interface is designed to provide a minimal mapping between integers, strings, and
 * memory addresses, much like a dictionary in Python. Users can map integers, strings, and
 * memory addresses to other data types, supporting insertion, search, deletion, and map clearing.
 *
 * @note Comparison functions used with the HashMap must:
 * - Return 0 for equality.
 * - Return a non-zero value for inequality.
 *
 * @note Thread Safety:
 * - The hash map is designed to be thread-safe using mutexes. Ensure that all operations on the
 * hash map are performed within critical sections to avoid race conditions.
 *
 * @note Supported Keys:
 * - Integers (`int32_t` and `int64_t`)
 * - Strings (`char*`)
 * - Memory addresses (`uintptr_t`)
 *
 * @note Probing:
 * - Linear probing is used to handle collisions.
 */

#include "core/logger.h"
#include "core/strext.h"
#include "core/hash.h"
#include "core/map.h"

/**
 * @section Life-cycle management
 * @{
 */

HashMap* hash_map_create(size_t capacity, HashType type) {
    return hash_create(capacity, type);
}

void hash_map_free(HashMap* map) {
    hash_free(map);
}

/** @} */

/**
 * @section External functions
 */

HashState hash_map_insert(HashMap* map, void* key, void* value) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for insert.");
        return HASH_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_ERROR;
    }

    HashState state;
    pthread_mutex_lock(&map->lock);

    if ((double) map->count / map->capacity > 0.75) {
        if (HASH_SUCCESS != hash_resize(map, map->capacity * 2)) {
            state = HASH_ERROR;
            goto exit;
        }
    }
    state = hash_insert(map, key, value);

exit:
    pthread_mutex_unlock(&map->lock);
    return state;
}

HashState hash_map_resize(HashMap* map, uint64_t new_size) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for resize.");
        return HASH_ERROR;
    }

    HashState state;
    pthread_mutex_lock(&map->lock);
    state = hash_resize(map, new_size);
    pthread_mutex_unlock(&map->lock);
    return state;
}

HashState hash_map_delete(HashMap* map, const void* key) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for delete.");
        return HASH_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_ERROR;
    }

    HashState state;
    pthread_mutex_lock(&map->lock);
    state = hash_delete(map, key);
    pthread_mutex_unlock(&map->lock);
    return state;
}

HashState hash_map_clear(HashMap* map) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for clear.");
        return HASH_ERROR;
    }

    HashState state;
    pthread_mutex_lock(&map->lock);
    state = hash_clear(map);
    pthread_mutex_unlock(&map->lock);
    return state;
}

void* hash_map_search(HashMap* map, const void* key) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for search.");
        return NULL;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return NULL;
    }

    void* value = NULL;
    pthread_mutex_lock(&map->lock);
    value = hash_search(map, key);
    pthread_mutex_unlock(&map->lock);
    return value;
}

/** @} */
