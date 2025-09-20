/**
 * Copyright © 2023 Austin Berrio
 *
 * @file map.h
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

#ifndef HASH_MAP_LINEAR_H
#define HASH_MAP_LINEAR_H

#include <stdint.h>
#include <pthread.h>

/// @note See hash.h for details
#include "core/hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef HashMap
 * @brief Alias for the base Hash structure, representing a generic hash map (key-value store).
 *
 * This typedef allows map-specific interfaces to use @c HashMap for clarity,
 * while relying on the underlying @c Hash implementation.
 *
 * The Hash structure contains:
 *   - @c HashEntry* entries:   Array of pointers to entries (key-value pairs)
 *   - @c HashType type:        Key type tag (e.g. int32, int64, str, ptr)
 *   - @c size_t count:         Number of active entries
 *   - @c size_t capacity:      Total capacity of the entries array
 *   - @c size_t size:          Key size in bytes
 *   - @c pthread_mutex_t lock: Mutex for thread safety
 *   - @c HashFn fn:            Hash function pointer
 *   - @c HashCmp cmp:          Comparison function pointer
 *
 * @note All logic for allocation, lookup, insertion, removal, and cleanup is
 *       provided by map.c using this alias; @c HashMap is never defined separately.
 */
typedef struct Hash HashMap;

/**
 * @name Life-cycle Management
 * @{
 */

/**
 * @brief Creates a new hash map.
 *
 * @param capacity Initial capacity of the map.
 * @param type Type of keys (integer, string, or address).
 * @return Pointer to the new hash map, or NULL on failure.
 * @note Hash objects do not own key-value pairs.
 */
HashMap* hash_map_create(size_t capacity, HashType type);

/**
 * @brief Frees a hash map and all associated memory.
 *
 * @param map Pointer to the hash map to free.
 * @note Hash objects do not free any key-value pairs.
 */
void hash_map_free(HashMap* map);

/** @} */

/**
 * @name Core Hash Operations
 * @note Thread-safe: acquires internal lock during operation.
 * @{
 */

/**
 * @brief Inserts a key-value pair into the hash map.
 *
 * If the map exceeds its load factor threshold, it will resize
 * automatically before insertion.
 *
 * @param map Pointer to the hash map.
 * @param key Pointer to the key to insert.
 * @param value Pointer to the value to associate with the key.
 * @return HASH_MAP_STATE_SUCCESS on success, or an error code.
 *
 * @note Automatically resizes if capacity is insufficient.
 */
HashState hash_map_insert(HashMap* map, void* key, void* value);

/**
 * @brief Resizes the hash map to a new capacity.
 *
 * @param map Pointer to the hash map.
 * @param new_size Desired new capacity.
 * @return HASH_MAP_STATE_SUCCESS on success, HASH_MAP_STATE_ERROR on failure.
 */
HashState hash_map_resize(HashMap* map, size_t new_capacity);

/**
 * @brief Deletes a key and its associated value from the hash map.
 *
 * @param map Pointer to the hash map.
 * @param key Pointer to the key to delete.
 * @return HASH_MAP_STATE_SUCCESS if deletion succeeded, HASH_MAP_STATE_KEY_NOT_FOUND if not found.
 */
HashState hash_map_delete(HashMap* map, const void* key);

/**
 * @brief Removes all entries from the hash map.
 *
 * @param map Pointer to the hash map.
 * @return HASH_MAP_STATE_SUCCESS on success, HASH_MAP_STATE_ERROR on failure.
 */
HashState hash_map_clear(HashMap* map);

/**
 * @brief Searches for a key in the hash map.
 *
 * @param map Pointer to the hash map.
 * @param key Pointer to the key to search.
 * @return Pointer to the associated value, or NULL if not found.
 */
void* hash_map_search(HashMap* map, const void* key);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // HASH_MAP_LINEAR_H
