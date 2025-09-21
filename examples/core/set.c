/// @file examples/core/set.c
/// @brief driver for handling an (un)ordered set of elements.
/// HashMap can regulate referenence management.
/// HashMap would be equivalent to using PDS (aka PDO) to store values as keys.
/// This is not as a simple as it initially appears. Especially when seeking out flexibility.
/// The keys of a map are usually restricted which makes sense when appropriately considered.
/// Instead, the point here is to experiment with a naive set implementation and to allow it
/// to organically evolve to see what happens. A HashSet can be used as a fallback if this
/// experiment fails.
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/logger.h"

/// there's a way to make elements flat. i just can't recall how atm.
typedef struct Set {
    void** elements;  // objects stored in the set
    size_t count;  // number of elements
    size_t capacity;  // total capacity in bytes
    size_t size;  // size of the object in bytes
} Set;

// this is convoluted, but i'm running with it for now.
// i'm sure this can be simplified.
Set* set_create(size_t n, size_t size) {
    Set* set = malloc(sizeof(Set));
    if (!set) {
        LOG_ERROR("Failed to create Set.");
        return NULL;
    }

    set->size = size;
    set->count = n > 0 ? n : 1;
    set->capacity = set->count * size;
    set->elements = malloc(set->capacity);
    if (!set->elements) {
        LOG_ERROR("Failed to allocated %zu bytes", set->capacity);
        free(set);
        return NULL;
    }

    return set;
}

void set_free(Set* set) {
    if (set) {
        if (set->elements) {
            free(set);
        }
        free(set);
    }
}

/// @note this is equivalent for checking validaty, e.g. is_valid()
/// maybe rename this to is valid then wrap this as a negated return value instead?
bool set_is_empty(Set* set) {
    // not sure if this is valid yet. probably expects inverse bool checks.
    // maybe just return count instead?
    return set && set->size > 0 && set->capacity > 0 && set->count > 0;
}

// start with naive linear search to keep it simple for now
bool set_contains(Set* set, void* value) {
    // not sure if this can be parallelized yet.
    for (size_t i = 0; i < set->count; i++) {
        // believe it or not, indexing is slower than shifting the pointer manually.
        // this is negligble with small sets, but significant with large sets.
        if (memcmp(set->elements + (i * set->size), value, set->size)) {
            return true;
        }
    }
    return false;
}

bool set_add(Set* set, void* value) {
    // catch duplicate values
    if (set_contains(set, value)) {
        return false;  // enums might be more useful, but this is simple
    }

    // resize set to fit input
    void** temp = realloc(set->elements, set->size * (set->count + 1));
    if (!temp) {
        return false;  // out of memory?
    }

    // insert value into set
    set->elements = temp;
    set->elements[set->count++] = value;
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
