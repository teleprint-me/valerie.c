/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/map.c
 * @brief Minimalistic hash table implementation providing mapping between integers, strings, and
 * memory addresses.
 *
 * The Hash Interface is designed to provide a minimal mapping between integers, strings, and memory
 * addresses, much like a dictionary in Python. Users can map integers, strings, and memory
 * addresses to other data types, supporting insertion, search, deletion, and table clearing.
 *
 * @note Comparison functions used with the HashMap must:
 * - Return 0 for equality.
 * - Return a non-zero value for inequality.
 *
 * @note Thread Safety:
 * - The hash table is designed to be thread-safe using mutexes. Ensure that all operations on the
 * hash table are performed within critical sections to avoid race conditions.
 *
 * @note Supported Keys:
 * - Integers (`uint64_t`)
 * - Strings (`char*`)
 * - Memory addresses (`uintptr_t`)
 *
 * @note Probing:
 * - Linear probing is used to handle collisions.
 */

#include "core/memory.h"
#include "core/logger.h"
#include "core/strext.h"
#include "core/map.h"

/**
 * @section Hash Life-cycle
 */

HashMap* hash_map_create(uint64_t initial_size, HashMapKeyType key_type) {
    HashMap* table = memory_alloc(sizeof(HashMap), alignof(HashMap));
    if (!table) {
        LOG_ERROR("Failed to allocate memory for HashMap.");
        return NULL;
    }

    table->count = 0;
    table->size = initial_size > 0 ? initial_size : 10;
    table->type = key_type;

    switch (table->type) {
        case HASH_MAP_KEY_TYPE_STRING:
            table->hash = hash_string;
            table->compare = hash_string_compare;
            break;
        case HASH_MAP_KEY_TYPE_INTEGER:
            table->hash = hash_integer;
            table->compare = hash_integer_compare;
            break;
        case HASH_MAP_KEY_TYPE_ADDRESS:
            table->hash = hash_address;
            table->compare = hash_address_compare;
            break;
        default:
            LOG_ERROR("Invalid HashMapKeyType given.");
            memory_free(table);
            return NULL;
    }

    table->entries = memory_calloc(table->size, sizeof(HashMapEntry), alignof(HashMapEntry));
    if (!table->entries) {
        LOG_ERROR("Failed to allocate memory for HashMap entries.");
        memory_free(table);
        return NULL;
    }

    // Initialize the mutex for thread safety
    int error_code = pthread_mutex_init(&table->thread_lock, NULL);
    if (0 != error_code) {
        LOG_ERROR("Failed to initialize mutex with error: %d", error_code);
        memory_free(table->entries);
        memory_free(table);
        return NULL;
    }

    return table;
}

void hash_map_free(HashMap* table) {
    if (table) {
        // Destroy the mutex before freeing memory
        pthread_mutex_destroy(&table->thread_lock);

        if (table->entries) {
            memory_free(table->entries);
        }

        memory_free(table);
        table = NULL;
    }
}

/**
 * @section Private Functions
 */

static HashMapState hash_map_insert_internal(HashMap* table, const void* key, void* value) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for insert internal.");
        return HASH_MAP_STATE_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_MAP_STATE_ERROR;
    }

    if (!value) {
        LOG_ERROR("Value is NULL.");
        return HASH_MAP_STATE_ERROR;
    }

    for (uint64_t i = 0; i < table->size; i++) {
        uint64_t index = table->hash(key, table->size, i);

        if (!table->entries[index].key) {
            table->entries[index].key = (void*) key;
            table->entries[index].value = value;
            table->count++;
            return HASH_MAP_STATE_SUCCESS;
        } else if (0 == table->compare(table->entries[index].key, key)) {
            return HASH_MAP_STATE_KEY_EXISTS;
        }
    }

    return HASH_MAP_STATE_FULL;
}

static HashMapState hash_map_resize_internal(HashMap* table, uint64_t new_size) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for resize internal.");
        return HASH_MAP_STATE_ERROR;
    }

    if (new_size <= table->size) {
        return HASH_MAP_STATE_SUCCESS;
    }

    HashMapEntry* new_entries = (HashMapEntry*) calloc(new_size, sizeof(HashMapEntry));
    if (!new_entries) {
        LOG_ERROR("Failed to allocate memory for resized table.");
        return HASH_MAP_STATE_ERROR;
    }

    // Backup
    HashMapEntry* old_entries = table->entries;
    uint64_t old_size = table->size;

    // Swap
    table->entries = new_entries;
    table->size = new_size;

    // Probe entries
    uint64_t rehashed_count = 0;
    for (uint64_t i = 0; i < old_size; i++) {
        HashMapEntry* entry = &old_entries[i];
        if (entry->key) {
            HashMapState state = hash_map_insert_internal(table, entry->key, entry->value);
            if (HASH_MAP_STATE_SUCCESS != state) {
                LOG_ERROR("Failed to rehash key during resize.");
                memory_free(new_entries);
                table->entries = old_entries;
                table->size = old_size;
                return state;
            }
            rehashed_count++;
        }
    }

    table->count = rehashed_count;
    memory_free(old_entries);
    return HASH_MAP_STATE_SUCCESS;
}

static HashMapState hash_map_delete_internal(HashMap* table, const void* key) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for delete internal.");
        return HASH_MAP_STATE_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_MAP_STATE_ERROR;
    }

    for (uint64_t i = 0; i < table->size; i++) {
        uint64_t index = table->hash(key, table->size, i);
        HashMapEntry* entry = &table->entries[index];

        if (!entry->key) {
            return HASH_MAP_STATE_KEY_NOT_FOUND; // Stop probing
        }

        if (0 == table->compare(entry->key, key)) {
            // Delete entry
            entry->key = NULL;
            entry->value = NULL;
            table->count--;

            // Rehash the remainder of the probe sequence
            for (uint64_t j = i + 1; j < table->size; j++) {
                uint64_t rehash_index = table->hash(key, table->size, j);
                HashMapEntry* rehash_entry = &table->entries[rehash_index];

                if (!rehash_entry->key) {
                    break;
                }

                void* rehash_key = rehash_entry->key;
                void* rehash_value = rehash_entry->value;

                rehash_entry->key = NULL;
                rehash_entry->value = NULL;
                table->count--;

                // Reinsert into new position
                HashMapState state = hash_map_insert_internal(table, rehash_key, rehash_value);
                if (HASH_MAP_STATE_SUCCESS != state) {
                    LOG_ERROR("Failed to reinsert during delete.");
                    return HASH_MAP_STATE_ERROR;
                }
            }

            return HASH_MAP_STATE_SUCCESS;
        }
    }

    return HASH_MAP_STATE_KEY_NOT_FOUND;
}

static HashMapState hash_map_clear_internal(HashMap* table) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for clear internal.");
        return HASH_MAP_STATE_ERROR;
    }

    for (uint64_t i = 0; i < table->size; i++) {
        table->entries[i].key = NULL;
        table->entries[i].value = NULL;
    }

    table->count = 0;
    return HASH_MAP_STATE_SUCCESS;
}

static void* hash_map_search_internal(HashMap* table, const void* key) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for search internal.");
        return NULL;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return NULL;
    }

    for (uint64_t i = 0; i < table->size; i++) {
        uint64_t index = table->hash(key, table->size, i);
        HashMapEntry* entry = &table->entries[index];

        if (!entry->key) {
            return NULL;
        }

        if (0 == table->compare(entry->key, key)) {
            return entry->value;
        }
    }

    return NULL;
}

/**
 * @section Hash Functions
 */

HashMapState hash_map_insert(HashMap* table, const void* key, void* value) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for insert.");
        return HASH_MAP_STATE_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_MAP_STATE_ERROR;
    }

    if (!value) {
        LOG_ERROR("Value is NULL.");
        return HASH_MAP_STATE_ERROR;
    }

    HashMapState state;
    pthread_mutex_lock(&table->thread_lock);

    if ((double) table->count / table->size > 0.75) {
        if (HASH_MAP_STATE_SUCCESS != hash_map_resize_internal(table, table->size * 2)) {
            state = HASH_MAP_STATE_ERROR;
            goto exit;
        }
    }
    state = hash_map_insert_internal(table, key, value);

exit:
    pthread_mutex_unlock(&table->thread_lock);
    return state;
}

HashMapState hash_map_resize(HashMap* table, uint64_t new_size) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for resize.");
        return HASH_MAP_STATE_ERROR;
    }

    HashMapState state;
    pthread_mutex_lock(&table->thread_lock);
    state = hash_map_resize_internal(table, new_size);
    pthread_mutex_unlock(&table->thread_lock);
    return state;
}

HashMapState hash_map_delete(HashMap* table, const void* key) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for delete.");
        return HASH_MAP_STATE_ERROR;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return HASH_MAP_STATE_ERROR;
    }

    HashMapState state;
    pthread_mutex_lock(&table->thread_lock);
    state = hash_map_delete_internal(table, key);
    pthread_mutex_unlock(&table->thread_lock);
    return state;
}

HashMapState hash_map_clear(HashMap* table) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for clear.");
        return HASH_MAP_STATE_ERROR;
    }

    HashMapState state;
    pthread_mutex_lock(&table->thread_lock);
    state = hash_map_clear_internal(table);
    pthread_mutex_unlock(&table->thread_lock);
    return state;
}

void* hash_map_search(HashMap* table, const void* key) {
    if (!table || !table->entries || table->size == 0) {
        LOG_ERROR("Invalid table for search.");
        return NULL;
    }

    if (!key) {
        LOG_ERROR("Key is NULL.");
        return NULL;
    }

    void* value = NULL;
    pthread_mutex_lock(&table->thread_lock);
    value = hash_map_search_internal(table, key);
    pthread_mutex_unlock(&table->thread_lock);
    return value;
}

uint64_t hash_map_size(HashMap* table) {
    if (table) {
        return table->size;
    }

    return 0;
}

uint64_t hash_map_count(HashMap* table) {
    if (table) {
        return table->count;
    }

    return 0;
}

/**
 * @section Hash Iterator
 * {@
 */

HashMapIterator hash_map_iter(HashMap* table) {
    HashMapIterator iter = { .table = table, .index = 0 };
    return iter;
}

HashMapEntry* hash_map_next(HashMapIterator* iter) {
    if (!iter || !iter->table || !iter->table->entries) {
        return NULL;
    }

    while (iter->index < iter->table->size) {
        HashMapEntry* entry = &iter->table->entries[iter->index++];
        if (entry->key != NULL) {
            return entry;
        }
    }

    return NULL;
}

uint64_t hash_map_iter_count(HashMap* table) {
    uint64_t count = 0;

    if (table) {
        HashMapEntry* entry;
        HashMapIterator it = hash_map_iter(table);
        while ((entry = hash_map_next(&it))) {
            count++;
        }
    }

    return count;
}

void hash_map_iter_free_kv(HashMap* table, void (*value_free)(void*)) {
    if (table) {
        HashMapEntry* entry;
        HashMapIterator it = hash_map_iter(table);
        while ((entry = hash_map_next(&it))) {
            memory_free(entry->key); // keys are restricted
            if (value_free) {
                value_free(entry->value); // custom allocation
            } else {
                memory_free(entry->value); // allocated segment
            }
        }
    }
}

void hash_map_iter_free_all(HashMap* table, void (*value_free)(void*)) {
    hash_map_iter_free_kv(table, value_free);
    hash_map_free(table);
}

/** @} */

/**
 * @section Hash Integers
 */

uint64_t hash_integer(const void* key, uint64_t size, uint64_t i) {
    const int32_t* k = (int32_t*) key;
    uint64_t hash = *k * 2654435761U; // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_integer_compare(const void* key1, const void* key2) {
    return *(const int32_t*) key1 - *(const int32_t*) key2;
}

int32_t* hash_integer_search(HashMap* table, const void* key) {
    return (int32_t*) hash_map_search_internal(table, key);
}

/**
 * @section Hash Strings
 */

uint64_t hash_djb2(const char* string) {
    uint64_t hash = 5381;
    int c;

    while ((c = *string++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}

uint64_t hash_string(const void* key, uint64_t size, uint64_t i) {
    const char* string = (const char*) key;
    return (hash_djb2(string) + i) % size;
}

int hash_string_compare(const void* key1, const void* key2) {
    return string_compare((const char*) key1, (const char*) key2);
}

char* hash_string_search(HashMap* table, const void* key) {
    return (char*) hash_map_search_internal(table, key);
}

/**
 * @section Hash Addresses
 */

uint64_t hash_address(const void* key, uint64_t size, uint64_t i) {
    uintptr_t addr = (uintptr_t) key;
    uint64_t hash = addr * 2654435761U; // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_address_compare(const void* key1, const void* key2) {
    intptr_t a = (intptr_t) key1;
    intptr_t b = (intptr_t) key2;
    return (a > b) - (a < b);
}

void* hash_address_search(HashMap* table, const void* key) {
    return (void*) hash_map_search_internal(table, key);
}
