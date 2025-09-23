/**
 * @file set.h
 * @brief Minimalistic hash set interface (wrapper over hash map).
 * @copyright Copyright © 2023 Austin Berrio
 *
 * @note
 * Null Sets:
 * The null set is a set object with a count of 0. NULL is never a null set.
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
 * @note
 * All logic for allocation, lookup, insertion, removal, and cleanup is provided
 * by hash.c aliasing @c Hash as @c HashSet is never defined separately.
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
