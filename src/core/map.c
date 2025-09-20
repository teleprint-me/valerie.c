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
 * @section Internal functions
 */

static HashState hash_map_insert_internal(HashMap* map, void* key, void* value) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for insert internal.");
        return HASH_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_ERROR;
    }

    for (size_t i = 0; i < map->capacity; i++) {
        uint64_t index = map->fn(key, map->capacity, i);

        if (!map->entries[index].key) {
            map->entries[index].key = key;
            map->entries[index].value = value;  // values are optional
            map->count++;
            return HASH_SUCCESS;
        } else if (0 == map->cmp(map->entries[index].key, key)) {
            return HASH_EXISTS;
        }
    }

    return HASH_FULL;
}

static HashState hash_map_resize_internal(HashMap* map, size_t new_capacity) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for resize internal.");
        return HASH_ERROR;
    }

    if (new_capacity <= map->capacity) {
        return HASH_SUCCESS;
    }

    HashEntry* new_entries = calloc(new_capacity, sizeof(HashEntry));
    if (!new_entries) {
        LOG_ERROR("Failed to allocate memory for resized map.");
        return HASH_ERROR;
    }

    // Backup
    HashEntry* old_entries = map->entries;
    size_t old_capacity = map->capacity;

    // Swap
    map->entries = new_entries;
    map->capacity = new_capacity;

    // Probe entries
    size_t rehashed_count = 0;
    for (size_t i = 0; i < old_capacity; i++) {
        HashEntry* entry = &old_entries[i];
        if (hash_entry_is_valid(entry)) {
            HashState state = hash_map_insert_internal(map, entry->key, entry->value);
            if (HASH_SUCCESS != state) {
                LOG_ERROR("Failed to rehash key during resize.");
                free(new_entries);
                map->entries = old_entries;
                map->capacity = old_capacity;
                return state;
            }
            rehashed_count++;
        }
    }

    map->count = rehashed_count;
    free(old_entries);
    return HASH_SUCCESS;
}

static HashState hash_map_delete_internal(HashMap* map, const void* key) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for delete internal.");
        return HASH_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_ERROR;
    }

    for (size_t i = 0; i < map->capacity; i++) {
        uint64_t index = map->fn(key, map->capacity, i);
        HashEntry* entry = &map->entries[index];

        if (!hash_entry_is_valid(entry)) {
            return HASH_NOT_FOUND;  // Stop probing
        }

        if (0 == map->cmp(entry->key, key)) {
            // Delete entry
            entry->key = NULL;
            entry->value = NULL;
            map->count--;

            // Rehash the remainder of the probe sequence
            for (size_t j = i + 1; j < map->capacity; j++) {
                uint64_t rehash_index = map->fn(key, map->capacity, j);
                HashEntry* rehash_entry = &map->entries[rehash_index];

                if (!hash_entry_is_valid(rehash_entry)) {
                    break;
                }

                void* rehash_key = rehash_entry->key;
                void* rehash_value = rehash_entry->value;

                rehash_entry->key = NULL;
                rehash_entry->value = NULL;
                map->count--;

                // Reinsert into new position
                HashState state = hash_map_insert_internal(map, rehash_key, rehash_value);
                if (HASH_SUCCESS != state) {
                    LOG_ERROR("Failed to reinsert during delete.");
                    return HASH_ERROR;
                }
            }

            return HASH_SUCCESS;
        }
    }

    return HASH_NOT_FOUND;
}

static HashState hash_map_clear_internal(HashMap* map) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for clear internal.");
        return HASH_ERROR;
    }

    for (size_t i = 0; i < map->capacity; i++) {
        map->entries[i].key = NULL;
        map->entries[i].value = NULL;
    }

    map->count = 0;
    return HASH_SUCCESS;
}

static void* hash_map_search_internal(HashMap* map, const void* key) {
    if (!hash_is_valid(map)) {
        LOG_ERROR("Invalid map for search internal.");
        return NULL;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return NULL;
    }

    for (size_t i = 0; i < map->capacity; i++) {
        uint64_t index = map->fn(key, map->capacity, i);
        HashEntry* entry = &map->entries[index];

        if (!hash_entry_is_valid(entry)) {
            return NULL;
        }

        if (0 == map->cmp(entry->key, key)) {
            return entry->value;
        }
    }

    return NULL;
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
        if (HASH_SUCCESS != hash_map_resize_internal(map, map->capacity * 2)) {
            state = HASH_ERROR;
            goto exit;
        }
    }
    state = hash_map_insert_internal(map, key, value);

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
    state = hash_map_resize_internal(map, new_size);
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
    state = hash_map_delete_internal(map, key);
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
    state = hash_map_clear_internal(map);
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
    value = hash_map_search_internal(map, key);
    pthread_mutex_unlock(&map->lock);
    return value;
}

int32_t* hash_map_search_int32(HashMap* map, const void* key) {
    return (int32_t*) hash_map_search_internal(map, key);
}

int64_t* hash_map_search_int64(HashMap* map, const void* key) {
    return (int64_t*) hash_map_search_internal(map, key);
}

char* hash_map_search_str(HashMap* map, const void* key) {
    return (char*) hash_map_search_internal(map, key);
}

void* hash_map_search_ptr(HashMap* map, const void* key) {
    return (void*) hash_map_search_internal(map, key);
}

/** @} */
