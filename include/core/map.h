/**
 * Copyright © 2023 Austin Berrio
 *
 * @file map.h
 * @brief Minimalistic hash map interface (thin wrapper over hash).
 *
 * Users can map integers, strings, and memory addresses to other data types,
 * supporting insert, resize, delete, and clear operations.
 *
 * @note
 * Comparison functions used with the hash API **must**:
 *   - Return `0` for equality.
 *   - Return non-zero for inequality.
 *
 * @note
 * Supported key types:
 *   - Integers (`int32_t`, `int64_t`)
 *   - Strings (`char*`, null-terminated)
 *   - Memory addresses (`uintptr_t`)
 *
 * @note
 * Thread Safety:
 *   - The hash interface itself is thread-agnostic.
 *   - **Consumers** must ensure all hash operations are protected with locks as needed.
 *   - See hash_lock() and hash_unlock() in hash.h for details.
 *
 * @note
 * Collision Handling:
 *   - Linear probing is used for collision resolution.
 *
 * @note
 * Memory Management:
 * - By default, the hash table does not deep-copy keys or values; it stores pointers.
 * - User is responsible for allocating all key/value memory.
 *   - i.e. Freeing, via iteration/free helpers.
 */

#ifndef HASH_MAP_H
#define HASH_MAP_H

#include <stdint.h>
#include <pthread.h>

/// @note See hash.h for details
#include "core/hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef HashMap
 * @brief Opaque alias for the base Hash structure, representing a generic map (key-value store).
 *
 * @note
 * All logic for allocation, lookup, insertion, removal, and cleanup is provided
 * by hash.c aliasing @c Hash as @c HashMap is never defined separately.
 *
 * Use only via the map interface functions below.
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

#endif  // HASH_MAP_H
