/**
 * Copyright © 2023 Austin Berrio
 *
 * @file set.h
 * @brief Minimalistic hash set implementation.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

// The cardinality (or size) of A is the number of elements in A.
size_t hash_set_count(HashSet* set) {
    return set ? set->count : 0;  // set is valid but empty
}

/// ∅ The empty set is the set which contains no elements.
bool hash_set_is_empty(HashSet* set) {
    return !set || set->count == 0;  // set is null
}

// 2 ∈ {1, 2, 3} asserts that 2 is an element of the set {1, 2, 3}.
bool hash_set_contains(HashSet* set, void* value) {
    if (hash_set_is_empty(set)) {
        return false;  // null set
    }
    return hash_map_search(set, value) == HASH_SET_VALUE;
}

// A ⊆ B asserts that A is a subset of B: every element of A is also an element of B.
bool hash_set_is_subset(HashSet* a, HashSet* b) {
    if (hash_set_is_empty(a) || hash_set_is_empty(b)) {
        return true;  // the empty set is a subset of any set
    }
    if (hash_set_count(a) > hash_set_count(b)) {
        return false;  // every element of a cannot be in b
    }

    HashEntry* entry;
    HashIt it = hash_iter(a);
    while ((entry = hash_iter_next(&it))) {
        if (!hash_set_contains(b, entry->key)) {
            return false;
        }
    }

    return true;
}

bool hash_set_is_equal(HashSet* a, HashSet* b) {
    if (a == b) {
        return true;  // same ptr, must be equal
    }
    if (hash_set_is_empty(a) || hash_set_is_empty(b)) {
        return false;  // invalid comparison (UB)
    }
    if (hash_set_count(a) != hash_set_count(b)) {
        return false;  // sets must be equal
    }
    // if all elements in a are in b, and counts are equal, then they're equal
    return hash_set_is_subset(a, b);
}

// Add a new value to the set.
bool hash_set_add(HashSet* set, void* value) {
    return hash_map_insert(set, value, HASH_SET_VALUE) == HASH_SUCCESS;
}

// Remove an existing value from the set.
bool hash_set_remove(HashSet* set, void* value) {
    return hash_map_delete(set, value) == HASH_SUCCESS;
}

bool hash_set_clear(HashSet* set) {
    return hash_map_clear(set) == HASH_SUCCESS;
}

// A ∪ B is the union of A and B: the set containing all
// elements which are elements of A or B or both.
HashSet* set_union(HashSet* a, HashSet* b) {
    // Handle null sets
    if (hash_set_is_empty(a) || hash_set_is_empty(b)) {
        return NULL;  // null sets have no unions (UB)
    }

    // Start with max capacity for all elements (max of both)
    size_t new_capacity = a->capacity + b->capacity;
    HashSet* new_set = hash_set_create(new_capacity, a->type);
    if (!new_set) {
        return NULL;
    }

    // Add all elements from a
    HashEntry* entry;
    HashIt it = hash_iter(a);
    while ((entry = hash_iter_next(&it))) {
        hash_set_add(new_set, entry->key);
    }

    // Add all elements from b
    it = hash_iter(b);
    while ((entry = hash_iter_next(&it))) {
        hash_set_add(new_set, entry->key);
    }

    // return the union of a and b
    return new_set;
}

/** @} */
