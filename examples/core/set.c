/**
 * @file examples/core/set.c
 * @brief Driver for an unordered set of elements, offering a flexible set interface.
 *
 * @ref https://discrete.openmathbooks.org/dmoi3.html
 * @ref https://discrete.openmathbooks.org/dmoi3/sec_intro-sets.html
 */

#include <assert.h>
#include <stdio.h>

#include "core/set.h"

void hash_set_print(Hash* h) {
    if (!h) {
        printf("Error: Invalid hash object: %p\n", (void*) h);
        return;
    }

    printf("size: %zu\n", h->size);
    printf("capacity: %zu\n", h->capacity);
    printf("count: %zu\n", h->count);
    printf("type: %d\n", h->type);

    bool valid_type = true;

    HashEntry* entry;
    HashIt it = hash_iter(h);
    while ((entry = hash_iter_next(&it))) {
        switch (h->type) {
            case HASH_INT32:
                printf("key: %d\n", *(int32_t*) entry->key);
                break;
            case HASH_INT64:
                printf("key: %ld\n", *(int64_t*) entry->key);
                break;
            case HASH_STR:
                printf("key: %s\n", (uint8_t*) entry->key);
                break;
            case HASH_PTR:
                printf("key: %p\n", (uint8_t*) entry->key);
                break;
            default:
                printf("Error: Invalid hash type!\n");
                valid_type = false;
                break;
        }
        if (!valid_type) {
            break;
        }
    }
    printf("\n");
}

int main(void) {
    // Integer set
    HashSet* s1 = hash_set_create(8, HASH_INT32);
    HashSet* s2 = hash_set_create(8, HASH_INT32);

    int a = 1, b = 2, c = 3, d = 4;
    hash_set_add(s1, &a);
    hash_set_add(s1, &b);
    hash_set_add(s1, &c);
    hash_set_add(s2, &b);
    hash_set_add(s2, &c);
    hash_set_add(s2, &d);

    hash_set_print(s1);
    hash_set_print(s2);

    // Test union
    HashSet* uni = hash_set_union(s1, s2);
    hash_set_print(uni);
    assert(hash_set_count(uni) == 4);

    // Test intersection
    HashSet* isect = hash_set_intersection(s1, s2);
    hash_set_print(isect);
    assert(hash_set_count(isect) == 2);
    assert(hash_set_contains(isect, &b));
    assert(hash_set_contains(isect, &c));

    // Test difference
    HashSet* diff = hash_set_difference(s1, s2);
    hash_set_print(diff);
    assert(hash_set_count(diff) == 1);
    assert(hash_set_contains(diff, &a));

    // Test subset/equality
    assert(hash_set_is_subset(isect, uni));
    assert(!hash_set_is_equal(s1, s2));
    assert(hash_set_is_equal(s1, s1));

    // Empty set behavior
    HashSet* empty = hash_set_create(1, HASH_INT32);
    HashSet* diff2 = hash_set_difference(s1, empty);
    assert(hash_set_count(diff2) == hash_set_count(s1));
    assert(hash_set_is_subset(empty, s1));
    assert(hash_set_is_equal(empty, empty));

    // Clean up
    hash_set_free(s1);
    hash_set_free(s2);
    hash_set_free(uni);
    hash_set_free(isect);
    hash_set_free(diff);
    hash_set_free(diff2);
    hash_set_free(empty);

    printf("All set algebra smoke tests passed!\n");
    return 0;
}
