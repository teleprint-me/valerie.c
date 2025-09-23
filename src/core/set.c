/**
 * @file set.c
 * @brief Minimalistic hash set interface (wrapper over hash map).
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/logger.h"
#include "core/hash.h"
#include "core/map.h"
#include "core/set.h"

/**
 * HashSet life-cycle
 * @{
 */

// Braces are used to enclose the elements of a set.
// e.g. {1, 2, 3} is the set containing 1, 2, and 3.
HashSet* hash_set_create(size_t capacity, HashType type) {
    return hash_create(capacity, type);
}

void hash_set_free(HashSet* set) {
    hash_free(set);
}

/** @} */

/**
 * HashSet operations
 * @{
 */

// 2 ∈ {1, 2, 3} asserts that 2 is an element of the set {1, 2, 3}.
bool hash_set_contains(HashSet* set, void* value) {
    if (!hash_is_valid(set)) {
        return false;  // undefined behavior
    }
    if (hash_is_empty(set)) {
        return false;  // null set
    }
    return hash_map_search(set, value) == HASH_SET_VALUE;
}

// A ⊆ B asserts that A is a subset of B: every element of A is also an element of B.
bool hash_set_is_subset(HashSet* a, HashSet* b) {
    if (!hash_is_valid(a) || !hash_is_valid(b)) {
        return false;  // undefined behavior
    }
    if (!hash_cmp_is_valid(a, b)) {
        return false;  // hash types do not match
    }
    if (hash_is_empty(a)) {
        return true;  // the empty set is a subset of every set, including the empty set
    }
    if (hash_is_empty(b)) {
        return false;  // a non-empty set is not a subset of the empty set
    }
    if (hash_count(a) > hash_count(b)) {
        return false;  // every element of A cannot be in B
    }

    // { ∀ a ∈ A : a ∈ B }
    HashEntry* entry;
    HashIt it = hash_iter(a);
    while ((entry = hash_iter_next(&it))) {
        if (!hash_set_contains(b, entry->key)) {
            return false;
        }
    }

    // A is a subset of B
    return true;
}

// A = B asserts that A and B are equal
bool hash_set_is_equal(HashSet* a, HashSet* b) {
    if (!hash_is_valid(a) || !hash_is_valid(b)) {
        return false;  // undefined behavior
    }
    if (!hash_cmp_is_valid(a, b)) {
        return false;  // hash types do not match
    }
    if (a == b) {
        return true;  // same ptr, must be equal
    }
    if (hash_is_empty(a) && hash_is_empty(b)) {
        return true;  // two empty sets are always equal
    }
    if (hash_is_empty(a) || hash_is_empty(b)) {
        return false;  // if one is empty, one is not, then they are not equal
    }
    if (hash_count(a) != hash_count(b)) {
        return false;  // sets must be equal
    }
    // if all elements in a are in b, and counts are equal, then they're equal
    return hash_set_is_subset(a, b);
}

// Add a new value to the set.
bool hash_set_add(HashSet* set, void* value) {
    if (!hash_is_valid(set) || !value) {
        return false;
    }
    HashState state = hash_map_insert(set, value, HASH_SET_VALUE);
    return state == HASH_SUCCESS || state == HASH_EXISTS;
}

// Remove an existing value from the set.
bool hash_set_remove(HashSet* set, void* value) {
    if (!hash_is_valid(set) || !value) {
        return false;  // undefined behavior
    }
    return hash_map_delete(set, value) == HASH_SUCCESS;
}

// Clear all existing values from the set.
bool hash_set_clear(HashSet* set) {
    if (!hash_is_valid(set)) {
        return false;  // undefined behavior
    }
    return hash_map_clear(set) == HASH_SUCCESS;
}

// Create a shallow copy of the given set.
HashSet* hash_set_clone(HashSet* set) {
    if (!hash_is_valid(set)) {
        return NULL;  // undefined behavior
    }

    HashSet* new_set = hash_set_create(set->capacity, set->type);
    if (!new_set) {
        return NULL;  // failed to alloc
    }

    // { x : x ∈ A }
    HashEntry* entry;
    HashIt it = hash_iter(set);
    while ((entry = hash_iter_next(&it))) {
        if (!hash_set_add(new_set, entry->key)) {
            hash_set_free(new_set);
            return NULL;  // failed to add element
        }
    }

    return new_set;
}

/// A ∪ B is the union of A and B: the set containing all
/// elements which are elements of A or B or both.
/// proof: A ∪ ∅ = { x : x ∈ A or x ∈ ∅} = { x : x ∈ A } = A
/// @note x ∈ ∅ is always false and is redundant.
/// @ref https://math.stackexchange.com/q/1124251
HashSet* hash_set_union(HashSet* a, HashSet* b) {
    if (!hash_is_valid(a) || !hash_is_valid(b)) {
        LOG_ERROR("HashSet is invalid!");
        return NULL;  // undefined behavior
    }
    if (!hash_cmp_is_valid(a, b)) {
        LOG_ERROR("HashSet types do not match!");
        return NULL;  // hash types do not match
    }

    // Both empty: result is empty set
    if (hash_is_empty(a) && hash_is_empty(b)) {
        return hash_set_create(1, a->type);
    }

    // a empty: A ∪ B = B (clone B)
    if (hash_is_empty(a)) {
        return hash_set_clone(b);
    }

    // b empty: A ∪ B = A (clone A)
    if (hash_is_empty(b)) {
        return hash_set_clone(a);
    }

    // Both non-empty: union logic
    size_t new_capacity = hash_capacity(a) + hash_capacity(b);
    HashSet* new_set = hash_set_create(new_capacity, a->type);
    if (!new_set) {
        LOG_ERROR("Failed to create a new HashSet.");
        return NULL;
    }

    // Add all elements from A
    HashEntry* entry;
    HashIt it = hash_iter(a);
    while ((entry = hash_iter_next(&it))) {
        if (!hash_set_add(new_set, entry->key)) {
            LOG_ERROR("Failed to add element from set A to set C: %p", entry->key);
            hash_set_free(new_set);
            return NULL;  // failed to add element
        }
    }

    // Add all elements from B
    entry = NULL;
    it = hash_iter(b);
    while ((entry = hash_iter_next(&it))) {
        if (!hash_set_add(new_set, entry->key)) {
            LOG_ERROR("Failed to add element from set B to set C: %p", entry->key);
            hash_set_free(new_set);
            return NULL;  // failed to add element
        }
    }

    // return the union of A and B
    return new_set;
}

// A ∩ B is the intersection of A and B:
// the set containing all elements which are elements of both A and B.
// A ∩ B = { x : x ∈ A ∧ x ∈ B }
HashSet* hash_set_intersection(HashSet* a, HashSet* b) {
    if (!hash_is_valid(a) || !hash_is_valid(b)) {
        return NULL;
    }
    if (!hash_cmp_is_valid(a, b)) {
        return NULL;
    }

    // If either is empty, intersection is empty.
    if (hash_is_empty(a) || hash_is_empty(b)) {
        return hash_set_create(1, a->type);
    }

    // Make new set (max possible = min(a,b) count)
    size_t min_capacity = hash_count(a) < hash_count(b) ? hash_count(a) : hash_count(b);

    // Allocate min of A and B
    HashSet* new_set = hash_set_create(min_capacity > 0 ? min_capacity : 1, a->type);
    if (!new_set) {
        return NULL;
    }

    // Iterate A, add to result if also in B.
    HashEntry* entry;
    HashIt it = hash_iter(a);
    while ((entry = hash_iter_next(&it))) {
        if (hash_set_contains(b, entry->key)) {
            if (!hash_set_add(new_set, entry->key)) {
                hash_set_free(new_set);
                return NULL;
            }
        }
    }

    // Return the intersection of A and B
    return new_set;
}

// A \ B is set difference between A and B: the set containing
// all elements of A which are not elements of B.
// A ∖ B = { x ∈ A : x ∉ B }
HashSet* hash_set_difference(HashSet* a, HashSet* b) {
    if (!hash_is_valid(a) || !hash_is_valid(b)) {
        return NULL;
    }
    if (!hash_cmp_is_valid(a, b)) {
        return NULL;
    }
    if (a == b) {
        // The difference of a set with itself is the empty set
        return hash_set_create(1, a->type);
    }
    if (hash_is_empty(a)) {
        return hash_set_create(1, a->type);
    }
    if (hash_is_empty(b)) {
        return hash_set_clone(a);  // ∀ of x ∈ A are not in B
    }

    HashSet* new_set = hash_set_create(a->capacity, a->type);
    if (!new_set) {
        return NULL;
    }

    // Add all elements of A which are not in B
    HashEntry* entry;
    HashIt it = hash_iter(a);
    while ((entry = hash_iter_next(&it))) {
        if (!hash_set_contains(b, entry->key)) {
            if (!hash_set_add(new_set, entry->key)) {
                hash_set_free(new_set);
                return NULL;
            }
        }
    }

    // Return the difference between A and B
    return new_set;
}

/** @} */
