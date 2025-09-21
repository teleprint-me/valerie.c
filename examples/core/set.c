/**
 * @file examples/core/set.c
 * @brief Driver for an (un)ordered set of elements, offering a flexible set interface.
 *
 * This implementation allows any object to be placed into the set and operates as expected.
 * It is a naive implementation and is not optimized for performance. A HashSet can be
 * used as a fallback if this experiment fails.
 */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/logger.h"

// Define a Set struct for handling a collection of objects.
// Each set stores a shallow copy of objects and does not free any
// resources referenced by those objects.
typedef struct Set {
    void* elements;  // Flat buffer storing the capacity * size
    size_t count;  // Current number of elements
    size_t capacity;  // Total capacity in terms of number of elements (not bytes)
    size_t size;  // Size of each object (in bytes)
} Set;

// Braces are used to enclose the elements of a set.
// e.g. {1, 2, 3} is the set containing 1, 2, and 3.
Set* set_create(size_t capacity, size_t size) {
    Set* set = malloc(sizeof(Set));
    if (!set) {
        LOG_ERROR("Failed to create Set.");
        return NULL;
    }

    if (!(size > 0)) {
        LOG_ERROR("Size must be sizeof(type)!");
        free(set);
        return NULL;
    }

    set->size = size;
    set->capacity = capacity > 0 ? capacity : 1;
    set->count = 0;
    set->elements = malloc(set->capacity * set->size);
    if (!set->elements) {
        LOG_ERROR("Failed to allocate %zu bytes", set->capacity);
        free(set);
        return NULL;
    }

    return set;
}

void set_free(Set* set) {
    if (set) {
        if (set->elements) {
            free(set->elements);
        }
        free(set);
    }
}

// The cardinality (or size) of A is the number of elements in A.
size_t set_count(Set* set) {
    return set ? set->count : 0;
}

/// @brief Calculates the column for a given element within the buffer.
/// @note C naturally aligns with row-major ordering.
/// @ref https://stackoverflow.com/a/14015582/15147156
uint8_t* set_element(Set* set, size_t i) {
    return set ? (uint8_t*) set->elements + i * set->size : NULL;
}

/// ∅ The empty set is the set which contains no elements.
bool set_is_empty(Set* set) {
    // not sure if this is valid yet. probably expects inverse bool checks.
    // maybe just return count instead?
    return !set || set->count == 0;
}

// start with naive linear search to keep it simple for now
// 2 ∈ {1, 2, 3} asserts that 2 is an element of the set {1, 2, 3}.
bool set_contains(Set* set, void* value) {
    if (set_is_empty(set)) {
        return false;  // null set
    }

    // not sure if this can be parallelized yet.
    for (size_t i = 0; i < set->count; i++) {
        // compare elements against value
        if (memcmp(set_element(set, i), value, set->size) == 0) {
            return true;
        }
    }
    return false;
}

// A ⊆ B asserts that A is a subset of B: every element of A is also an element of B.
bool set_is_subset(Set* a, Set* b) {
    for (size_t i = 0; i < a->count; i++) {
        if (!set_contains(b, set_element(a, i))) {
            return false;
        }
    }
    return true;
}

bool set_is_equal(Set* a, Set* b) {
    // null sets are equal
    if (set_is_empty(a) && set_is_empty(b)) {
        return true;
    }

    // not equal if one set is null and the other is not
    if (!set_is_empty(a) || !set_is_empty(b)) {
        return false;
    }

    // sets are not equal
    if (a->count != b->count) {
        return false;
    }

    // compare elements. O(n^2) is okay for now.
    return set_is_subset(a, b) && set_is_subset(b, a);
}

// technically, this just appends a new value into the sequence.
bool set_add(Set* set, void* value) {
    // catch duplicate values
    if (set_contains(set, value)) {
        return false;  // enums might be more useful, but this is simple
    }

    // out of memory
    if (set->count == set->capacity) {
        size_t new_capacity = set->capacity * 2;
        void* temp = realloc(set->elements, new_capacity * set->size);
        if (!temp) {
            return false;
        }
        set->elements = temp;
        set->capacity = new_capacity;
    }

    /// insert value into set
    /// @note memmove is safer with overlapping ops
    memmove(set_element(set, set->count), value, set->size);
    set->count++;
    return true;
}

// not sure how to handle non-existant values
size_t set_index(Set* set, void* value) {
    if (!set) {
        return SIZE_MAX;  // null set
    }

    for (size_t i = 0; i < set->count; i++) {
        if (memcmp(set_element(set, i), value, set->size) == 0) {
            return i;
        }
    }

    return SIZE_MAX;  // nothing exists
}

bool set_remove(Set* set, void* value) {
    if (!set_contains(set, value)) {
        return false;  // nothing to remove
    }

    // get the index for the given value
    size_t index = set_index(set, value);
    if (index == SIZE_MAX) {
        return false;  // not found
    }

    // shift elements after index left by one
    if (index < set->count - 1) {
        void* dst = set_element(set, index);
        void* src = set_element(set, index + 1);
        size_t bytes = (set->count - index - 1) * set->size;
        /// @note memmove is safer with overlapping ops
        memmove(dst, src, bytes);
    }

    set->count--;  // update!
    return true;
}

bool set_clear(Set* set) {
    // null sets are cleared already
    if (set_is_empty(set)) {
        return false;
    }
    // zero out memory for safety
    memset(set->elements, 0, set->capacity * set->size);
    // reset number of elements
    set->count = 0;
    return true;  // ok
}

// A ∪ B is the union of A and B: the set containing all
// elements which are elements of A or B or both.
Set* set_union(Set* a, Set* b) {
    // Handle null sets
    if (set_is_empty(a) && set_is_empty(b)) {
        // If both are empty, return an empty set
        return set_create(1, a ? a->size : (b ? b->size : sizeof(char)));
    }
    if (set_is_empty(a)) {
        return set_create(b->capacity, b->size);
    }
    if (set_is_empty(b)) {
        return set_create(a->capacity, a->size);
    }

    // Start with enough capacity for all elements (max of both)
    size_t new_capacity = a->count + b->count;
    Set* new_set = set_create(new_capacity, a->size);
    if (!new_set) {
        return NULL;
    }

    // Add all elements from a
    for (size_t i = 0; i < a->count; i++) {
        set_add(new_set, set_element(a, i));  // set_add ensures no duplicates
    }

    // Add all elements from b
    for (size_t i = 0; i < b->count; i++) {
        set_add(new_set, set_element(b, i));
    }

    // return the union of a and b
    return new_set;
}

int main(void) {
    Set* set = set_create(1, sizeof(int));
    if (!set) {
        return EXIT_FAILURE;
    }

    int a = 3;
    int b = 2;
    int c = 5;

    assert(set_add(set, &a));
    assert(set_add(set, &b));
    assert(set_add(set, &c));

    assert(set_contains(set, &a));

    assert(set_remove(set, &c));
    assert(!set_contains(set, &c));

    printf("All assertions passed!\n");
    set_free(set);
    return 0;
}
