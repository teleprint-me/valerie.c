/**
 * @file      hash.c
 * @brief     General-purpose hash and comparison functions for sets/maps in C.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This source provides hash and compare function pointer types, supported key types,
 * and basic hash functions for use in generic hash-based containers.
 *
 * - Supports int32, int64, pointer, and null-terminated string keys.
 * - Designed to be extensible for additional types as needed.
 * - For use in hash set, hash map, and similar data structures.
 */

#include <string.h>

#include "core/hash.h"

/**
 * @section Hash int
 */

uint64_t hash_int32(const void* key, uint64_t size, uint64_t i) {
    const int32_t* k = (int32_t*) key;
    uint64_t hash = *k * HASH_KNUTH;  // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_int32_cmp(const void* key1, const void* key2) {
    return *(const int32_t*) key1 - *(const int32_t*) key2;
}

/**
 * @section Hash long
 */

uint64_t hash_int64(const void* key, uint64_t size, uint64_t i) {
    const int64_t* k = (int64_t*) key;
    uint64_t hash = *k * HASH_KNUTH;  // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_int64_cmp(const void* key1, const void* key2) {
    return *(const int64_t*) key1 - *(const int64_t*) key2;
}

/**
 * @section Hash ptr
 */

uint64_t hash_ptr(const void* key, uint64_t size, uint64_t i) {
    uintptr_t addr = (uintptr_t) key;
    uint64_t hash = addr * HASH_KNUTH;  // Knuth's multiplicative
    return (hash + i) % size;
}

int hash_ptr_cmp(const void* key1, const void* key2) {
    intptr_t a = (intptr_t) key1;
    intptr_t b = (intptr_t) key2;
    return (a > b) - (a < b);
}

/**
 * @section Hash char
 */

uint64_t hash_djb2(const char* string) {
    uint64_t hash = 5381;
    int c;

    while ((c = *string++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}

uint64_t hash_str(const void* key, uint64_t size, uint64_t i) {
    const char* string = (const char*) key;
    return (hash_djb2(string) + i) % size;
}

int hash_str_cmp(const void* key1, const void* key2) {
    return strcmp((const char*) key1, (const char*) key2);
}
