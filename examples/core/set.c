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

// We use these braces to enclose the elements of a set.
// So {1, 2, 3} is the set containing 1, 2, and 3.
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
    // a set may be null mathematically, but this may conflict with C itself.
    // implementation is to be decided. this is fine for now to enforce simplicity.
    if (!a || !b) {
        return false;
    }

    // sets are not equal
    if (a->count != b->count) {
        return false;
    }

    // compare elements
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

int main(void) {
    Set* set = set_create(1, sizeof(int));
    if (!set) {
        return EXIT_FAILURE;
    }

    set_free(set);
    return 0;
}
