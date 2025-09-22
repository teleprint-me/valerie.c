/**
 * Copyright © 2023 Austin Berrio
 *
 * @file set.h
 * @brief Minimalistic hash set interface (wrapper over hash map).
 *
 * @note Thread Safety:
 *   - All set operations are thread-safe (via internal mutex in HashSet).
 *   - Do not mutate the same set from multiple threads without synchronization.
 *
 * @note Supported key types:
 *   - int32_t, int64_t, uintptr_t, char*
 *
 * @note The null set is a valid set object with count 0.
 *   - NULL pointers are never valid sets.
 *
 * @ref https://discrete.openmathbooks.org/dmoi3.html
 * @ref https://discrete.openmathbooks.org/dmoi3/sec_intro-sets.html
 */

#ifndef HASH_SET_H
#define HASH_SET_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "core/hash.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HASH_SET_VALUE ((void*) 1)

/**
 * @typedef HashSet
 * @brief Opaque alias for the base Hash structure, representing a generic set.
 *
 * Use only via the set interface functions below.
 */
typedef struct Hash HashSet;

/** Life-cycle  **/

/**
 * @brief Creates a new empty set with specified capacity and key type.
 * @param capacity Initial capacity (minimum 1).
 * @param type     HashType of the keys (see hash.h).
 * @return Pointer to a new HashSet, or NULL on failure.
 */
HashSet* hash_set_create(size_t capacity, HashType type);

/**
 * @brief Frees a set and all internal memory.
 * @param set Set to free.
 */
void hash_set_free(HashSet* set);

/**  Basic Queries  **/

/**
 * @brief Returns the number of elements in the set.
 * @param set Set to query.
 * @return Element count, or 0 if set is invalid.
 */
size_t hash_set_count(HashSet* set);

/**
 * @brief Returns true if set is valid and empty.
 */
bool hash_set_is_empty(HashSet* set);

/**
 * @brief Returns true if set contains the given key.
 * @param set Set to query.
 * @param value Pointer to key (not NULL).
 */
bool hash_set_contains(HashSet* set, void* value);

/**  Core Operations  **/

/**
 * @brief Adds a key to the set.
 * @param set Set to modify.
 * @param value Pointer to key (not NULL).
 * @return true on success, false on error.
 */
bool hash_set_add(HashSet* set, void* value);

/**
 * @brief Removes a key from the set.
 * @param set Set to modify.
 * @param value Pointer to key (not NULL).
 * @return true on success, false if not present or error.
 */
bool hash_set_remove(HashSet* set, void* value);

/**
 * @brief Removes all keys from the set (empties set, does not free).
 * @param set Set to clear.
 * @return true on success, false on error.
 */
bool hash_set_clear(HashSet* set);

/**  Set Algebra  **/

/**
 * @brief Returns true if every element of a is in b (A ⊆ B).
 */
bool hash_set_is_subset(HashSet* a, HashSet* b);

/**
 * @brief Returns true if two sets have the same keys (A = B).
 */
bool hash_set_is_equal(HashSet* a, HashSet* b);

/**
 * @brief Returns a new set with a shallow copy of the input set.
 * @param set Set to clone.
 * @return Pointer to new set, or NULL on error.
 */
HashSet* hash_set_clone(HashSet* set);

/**
 * @brief Returns the union of two sets (A ∪ B).
 * @param a First set.
 * @param b Second set.
 * @return New set (union), or NULL on error.
 */
HashSet* hash_set_union(HashSet* a, HashSet* b);

/**
 * @brief Returns the intersection of two sets (A ∩ B).
 * @param a First set.
 * @param b Second set.
 * @return New set (intersection), or NULL on error.
 */
HashSet* hash_set_intersection(HashSet* a, HashSet* b);

/**
 * @brief Returns the set difference (A \ B).
 * @param a First set.
 * @param b Second set.
 * @return New set (difference), or NULL on error.
 */
HashSet* hash_set_difference(HashSet* a, HashSet* b);

#ifdef __cplusplus
}
#endif

#endif  // HASH_SET_H
