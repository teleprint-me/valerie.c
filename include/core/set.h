/**
 * Copyright © 2023 Austin Berrio
 *
 * @file set.h
 * @brief Minimalistic hash set implementation.
 *
 * @note Thread Safety:
 * - The hash set is designed to be thread-safe using mutexes. Ensure that all operations on the
 * hash set are performed within critical sections to avoid race conditions.
 *
 * @note Supported Keys:
 * - Integers (`int32_t` and `int64_t`)
 * - Strings (`char*`)
 * - Memory addresses (`uintptr_t`)
 *
 * @note Probing:
 * - Linear probing is used to handle collisions.
 */

#ifndef HASH_SET_H
#define HASH_SET_H

#include <stdint.h>
#include <pthread.h>

/// @note See hash.h for details
#include "core/hash.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HASH_SET_VALUE ((void*) 1)

/**
 * @typedef HashSet
 * @brief Alias for the base Hash structure, representing a generic hash set (key-value store).
 *
 * This typedef allows set-specific interfaces to use @c HashSet for clarity,
 * while relying on the underlying @c Hash implementation.
 *
 * The Hash structure contains:
 *   - @c HashEntry* entries:   Array of pointers to entries (key-value pairs)
 *   - @c HashType type:        Key type tag (e.g. int32, int64, str, ptr)
 *   - @c size_t count:         Number of active entries
 *   - @c size_t capacity:      Total capacity of the entries array
 *   - @c size_t size:          Key size in bytes
 *   - @c pthread_mutex_t lock: Mutex for thread safety
 *   - @c HashFn fn:            Hash function pointer
 *   - @c HashCmp cmp:          Comparison function pointer
 *
 * @note All logic for allocation, lookup, insertion, removal, and cleanup is
 *       provided by hash.c using this alias; @c HashSet is never defined separately.
 */
typedef struct Hash HashSet;

#ifdef __cplusplus
}
#endif

#endif  // HASH_SET_H
