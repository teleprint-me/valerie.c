/**
 * @file examples/core/set.c
 * @brief driver for handling an (un)ordered set of elements.
 * HashMap can regulate referenence management.
 * HashMap would be equivalent to using PDS (aka PDO) to store values as keys.
 * This is not as a simple as it initially appears. Especially when seeking out flexibility.
 * The keys of a map are usually restricted which makes sense when appropriately considered.
 * Instead, the point here is to experiment with a naive set implementation and to allow it
 * to organically evolve to see what happens. A HashSet can be used as a fallback if this
 * experiment fails.
 * The goal of the experiment is to implement a flexible set interface.
 * Any object should be able to be placed into the set and it should operate as expected.
 */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/logger.h"

typedef struct Set {
    void* elements;  // flat buffer stored as capacity * size
    size_t count;  // current number of elements
    size_t capacity;  // total capacity in number of elements (not bytes)
    size_t size;  // size of the object (in bytes)
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
        LOG_ERROR("Size must be greater than 0!");
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

/// @note this is equivalent for checking validaty, e.g. is_valid()
/// maybe rename this to is valid then wrap this as a negated return value instead?
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
        // get current element
        void* element = set_element(set, i);
        // compare elements against value
        if (memcmp(element, value, set->size) == 0) {
            return true;
        }
    }
    return false;
}

// A ⊆ B asserts that A is a subset of B: every element of A is also an element of B.
bool set_is_subset(Set* a, Set* b) {
    for (size_t i = 0; i < a->count; i++) {
        void* element = set_element(a, i);
        if (!set_contains(b, element)) {
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

    // insert value into set
    void* element = set_element(set, set->count);
    memmove(element, value, set->size);  // memmove is safer with inline ops
    set->count++;
    return true;
}

// not sure how to handle non-existant values
size_t set_index(Set* set, void* value) {
    if (!set) {
        return SIZE_MAX;  // null set
    }

    for (size_t i = 0; i < set->count; i++) {
        void* element = set_element(set, i);
        if (memcmp(element, value, set->size) == 0) {
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
        memmove(dst, src, bytes);
    }

    set->count--;  // update!
    return true;
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

    set_free(set);
    return 0;
}
