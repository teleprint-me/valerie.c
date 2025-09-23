/**
 * @file map.c
 * @brief Minimalistic hash map interface (thin wrapper over hash).
 * @copyright Copyright © 2023 Austin Berrio
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
    if ((double) map->count / map->capacity > 0.75) {
        if (HASH_SUCCESS != hash_resize(map, map->capacity * 2)) {
            state = HASH_ERROR;
            goto exit;
        }
    }
    state = hash_insert(map, key, value);

exit:
    return state;
}

HashState hash_map_resize(HashMap* map, uint64_t new_size) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for resize.");
        return HASH_ERROR;
    }

    return hash_resize(map, new_size);
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

    return hash_delete(map, key);
}

HashState hash_map_clear(HashMap* map) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for clear.");
        return HASH_ERROR;
    }

    return hash_clear(map);
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

    return hash_search(map, key);
}

/** @} */
