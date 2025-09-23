/**
 * @file      hash.c
 * @brief     General-purpose hash and comparison functions for sets/maps in C.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This source provides hash and compare function pointer types, supported key types,
 * and basic hash functions for use in generic hash-based containers.
 *
 * - Supports int32, int64, pointer, and null-terminated string keys.
 * - Designed to be extensible for additional types as needed.
 * - For use in hash set, hash map, and similar data structures.
 */

#include <stdlib.h>
#include <string.h>

#include "core/logger.h"
#include "core/hash.h"

/**
 * @section Hash int
 */

uint64_t hash_int32(const void* key, uint64_t size, uint64_t i) {
    const int32_t* k = (int32_t*) key;
    uint64_t hash = *k * HASH_KNUTH;  // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_int32_cmp(const void* key1, const void* key2) {
    return *(const int32_t*) key1 - *(const int32_t*) key2;
}

/** @} */

/**
 * @section Hash long
 */

uint64_t hash_int64(const void* key, uint64_t size, uint64_t i) {
    const int64_t* k = (int64_t*) key;
    uint64_t hash = *k * HASH_KNUTH;  // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_int64_cmp(const void* key1, const void* key2) {
    return *(const int64_t*) key1 - *(const int64_t*) key2;
}

/** @} */

/**
 * @section Hash char ptr
 */

uint64_t hash_djb2(const char* string) {
    uint64_t hash = 5381;
    int c;

    while ((c = *string++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}

uint64_t hash_str(const void* key, uint64_t size, uint64_t i) {
    const char* string = (const char*) key;
    return (hash_djb2(string) + i) % size;
}

int hash_str_cmp(const void* key1, const void* key2) {
    return strcmp((const char*) key1, (const char*) key2);
}

/** @} */

/**
 * @section Hash unsigned long ptr
 */

uint64_t hash_ptr(const void* key, uint64_t size, uint64_t i) {
    uintptr_t addr = (uintptr_t) key;
    uint64_t hash = addr * HASH_KNUTH;  // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_ptr_cmp(const void* key1, const void* key2) {
    intptr_t a = (intptr_t) key1;
    intptr_t b = (intptr_t) key2;
    return (a > b) - (a < b);
}

/** @} */

/**
 * @section Hash life-cycle
 */

Hash* hash_create(size_t capacity, HashType type) {
    Hash* h = malloc(sizeof(Hash));
    if (!h) {
        LOG_ERROR("Failed to allocate memory for HashSet.");
        return NULL;
    }

    h->count = 0;
    h->capacity = capacity > 0 ? capacity : 10;
    h->type = type;

    switch (h->type) {
        case HASH_INT32:
            h->fn = hash_int32;
            h->cmp = hash_int32_cmp;
            h->size = sizeof(int32_t);
            break;
        case HASH_INT64:
            h->fn = hash_int64;
            h->cmp = hash_int64_cmp;
            h->size = sizeof(int64_t);
            break;
        case HASH_PTR:
            h->fn = hash_ptr;
            h->cmp = hash_ptr_cmp;
            h->size = sizeof(uintptr_t);
            break;
        case HASH_STR:
            h->fn = hash_str;
            h->cmp = hash_str_cmp;
            h->size = sizeof(char);
            break;
        default:
            LOG_ERROR("Invalid HashType given.");
            free(h);
            return NULL;
    }

    h->entries = calloc(h->capacity, sizeof(HashEntry));
    if (!h->entries) {
        LOG_ERROR("Failed to allocate memory for HashMap data.");
        free(h);
        return NULL;
    }

    // Initialize the mutex for thread safety
    int error_code = pthread_mutex_init(&h->lock, NULL);
    if (0 != error_code) {
        LOG_ERROR("Failed to initialize mutex with error: %d", error_code);
        free(h->entries);
        free(h);
        return NULL;
    }

    return h;
}

void hash_free(Hash* h) {
    if (h) {
        // Destroy the mutex before freeing memory
        pthread_mutex_destroy(&h->lock);

        if (h->entries) {
            free(h->entries);
        }

        free(h);
        h = NULL;
    }
}

/** @} */

/**
 * @section Atomic operations
 * @{
 */

int hash_lock(Hash* h) {
    return pthread_mutex_lock(&h->lock);
}

int hash_unlock(Hash* h) {
    return pthread_mutex_unlock(&h->lock);
}

void hash_lock_pair(Hash* a, Hash* b) {
    if (a < b) {
        hash_lock(a);
        hash_lock(b);
    } else {
        hash_lock(b);
        hash_lock(a);
    }
}

void hash_unlock_pair(Hash* a, Hash* b) {
    hash_unlock(a);
    hash_unlock(b);
}

/** @} */

/**
 * @section Hash queries
 */

size_t hash_count(const Hash* h) {
    return h ? h->count : 0;
}

size_t hash_capacity(const Hash* h) {
    return h ? h->capacity : 0;
}

size_t hash_size(const Hash* h) {
    return h ? h->size : 0;
}

HashType hash_type(const Hash* h) {
    return h ? h->type : HASH_UNK;
}

bool hash_is_empty(Hash* h) {
    return hash_count(h) == 0;
}

bool hash_is_valid(const Hash* h) {
    return h && h->entries && h->capacity > 0;
}

bool hash_entry_is_valid(const HashEntry* e) {
    return e && e->key;
}

bool hash_cmp_is_valid(const Hash* a, const Hash* b) {
    return a->type == b->type;
}

bool hash_type_is_valid(const Hash* h) {
    if (!h) {
        return false;
    }

    switch (h->type) {
        case HASH_INT32:
        case HASH_INT64:
        case HASH_STR:
        case HASH_PTR:
            return true;
        case HASH_UNK:
        default:
            return false;
    }
}

/** @} */

/**
 * @section Hash operations
 */

HashState hash_insert(Hash* h, void* key, void* value) {
    if (!hash_is_valid(h)) {
        LOG_ERROR("Invalid hash for internal insert.");
        return HASH_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_ERROR;
    }

    for (size_t i = 0; i < h->capacity; i++) {
        uint64_t index = h->fn(key, h->capacity, i);

        if (!h->entries[index].key) {
            h->entries[index].key = key;
            h->entries[index].value = value;  // values are optional
            h->count++;
            return HASH_SUCCESS;
        } else if (0 == h->cmp(h->entries[index].key, key)) {
            return HASH_EXISTS;
        }
    }

    return HASH_FULL;
}

HashState hash_resize(Hash* h, size_t new_capacity) {
    if (!hash_is_valid(h)) {
        LOG_ERROR("Invalid hash for internal resize.");
        return HASH_ERROR;
    }

    if (new_capacity <= h->capacity) {
        return HASH_SUCCESS;
    }

    HashEntry* new_entries = calloc(new_capacity, sizeof(HashEntry));
    if (!new_entries) {
        LOG_ERROR("Failed to allocate memory for resized hash.");
        return HASH_ERROR;
    }

    // Backup
    HashEntry* old_entries = h->entries;
    size_t old_capacity = h->capacity;

    // Swap
    h->entries = new_entries;
    h->capacity = new_capacity;

    // Probe entries
    size_t rehashed_count = 0;
    for (size_t i = 0; i < old_capacity; i++) {
        HashEntry* entry = &old_entries[i];
        if (hash_entry_is_valid(entry)) {
            HashState state = hash_insert(h, entry->key, entry->value);
            if (HASH_SUCCESS != state) {
                LOG_ERROR("Failed to rehash key during resize.");
                free(new_entries);
                h->entries = old_entries;
                h->capacity = old_capacity;
                return state;
            }
            rehashed_count++;
        }
    }

    h->count = rehashed_count;
    free(old_entries);
    return HASH_SUCCESS;
}

HashState hash_delete(Hash* h, const void* key) {
    if (!hash_is_valid(h)) {
        LOG_ERROR("Invalid hash for internal delete.");
        return HASH_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_ERROR;
    }

    for (size_t i = 0; i < h->capacity; i++) {
        uint64_t index = h->fn(key, h->capacity, i);
        HashEntry* entry = &h->entries[index];

        if (!hash_entry_is_valid(entry)) {
            return HASH_NOT_FOUND;  // Stop probing
        }

        if (0 == h->cmp(entry->key, key)) {
            // Delete entry
            entry->key = NULL;
            entry->value = NULL;
            h->count--;

            // Rehash the remainder of the probe sequence
            for (size_t j = i + 1; j < h->capacity; j++) {
                uint64_t rehash_index = h->fn(key, h->capacity, j);
                HashEntry* rehash_entry = &h->entries[rehash_index];

                if (!hash_entry_is_valid(rehash_entry)) {
                    break;
                }

                void* rehash_key = rehash_entry->key;
                void* rehash_value = rehash_entry->value;

                rehash_entry->key = NULL;
                rehash_entry->value = NULL;
                h->count--;

                // Reinsert into new position
                HashState state = hash_insert(h, rehash_key, rehash_value);
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

HashState hash_clear(Hash* h) {
    if (!hash_is_valid(h)) {
        LOG_ERROR("Invalid hash for internal clear.");
        return HASH_ERROR;
    }

    for (size_t i = 0; i < h->capacity; i++) {
        h->entries[i].key = NULL;
        h->entries[i].value = NULL;
    }

    h->count = 0;
    return HASH_SUCCESS;
}

void* hash_search(Hash* h, const void* key) {
    if (!hash_is_valid(h)) {
        LOG_ERROR("Invalid hash for internal search.");
        return NULL;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return NULL;
    }

    for (size_t i = 0; i < h->capacity; i++) {
        uint64_t index = h->fn(key, h->capacity, i);
        HashEntry* entry = &h->entries[index];

        if (!hash_entry_is_valid(entry)) {
            return NULL;
        }

        if (0 == h->cmp(entry->key, key)) {
            return entry->value;
        }
    }

    return NULL;
}

/** @} */

/**
 * @section Hash iterator
 * {@
 */

HashIt hash_iter(Hash* h) {
    return (HashIt) {.table = h, .index = 0};
}

bool hash_iter_is_valid(HashIt* it) {
    return it && it->table && it->table->entries;
}

HashEntry* hash_iter_next(HashIt* it) {
    if (!hash_iter_is_valid(it)) {
        return NULL;
    }

    while (it->index < it->table->capacity) {
        HashEntry* entry = &it->table->entries[it->index++];
        if (hash_entry_is_valid(entry)) {
            return entry;
        }
    }

    return NULL;
}

void hash_iter_log(Hash* h) {
    if (!hash_type_is_valid(h)) {
        LOG_ERROR("Error: Invalid hash object: %p", (void*) h);
        return;
    }

    LOG_INFO("size: %zu", h->size);
    LOG_INFO("capacity: %zu", h->capacity);
    LOG_INFO("count: %zu", h->count);
    LOG_INFO("type: %d", h->type);

    HashEntry* entry;
    HashIt it = hash_iter(h);
    while ((entry = hash_iter_next(&it))) {
        switch (h->type) {
            case HASH_INT32:
                LOG_INFO("key: %d", *(int32_t*) entry->key);
                break;
            case HASH_INT64:
                LOG_INFO("key: %ld", *(int64_t*) entry->key);
                break;
            case HASH_STR:
                LOG_INFO("key: %s", (uint8_t*) entry->key);
                break;
            case HASH_PTR:
                LOG_INFO("key: %p", (uint8_t*) entry->key);
                break;
            default:
                LOG_ERROR("Error: Invalid hash type: %d", h->type);
                break;
        }
    }
}

void hash_iter_free_kv(Hash* h, HashValueFree value_free) {
    if (h) {
        HashEntry* entry;
        HashIt it = hash_iter(h);
        while ((entry = hash_iter_next(&it))) {
            // Keys are always allocated
            free(entry->key);  // Restricted by HashType

            // Values are optional (May be NULL)
            if (value_free) {
                value_free(entry->value);  // custom() or free()
            }
        }
    }
}

void hash_iter_free_all(Hash* h, HashValueFree value_free) {
    hash_iter_free_kv(h, value_free);
    hash_free(h);
}

/** @} */
