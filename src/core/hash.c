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
 * @section Hash utils
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

bool hash_is_valid(const Hash* h) {
    return h && h->entries && h->capacity > 0;
}

bool hash_entry_is_valid(const HashEntry* e) {
    return e && e->key;
}

bool hash_type_is_valid(const Hash* a, const Hash* b) {
    return a->type == b->type;
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
