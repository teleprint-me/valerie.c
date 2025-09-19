/**
 * @file      hash.h
 * @brief     General-purpose hash and comparison functions for sets/maps in C.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This header provides hash and compare function pointer types, supported key types,
 * and basic hash functions for use in generic hash-based containers.
 *
 * - Supports int32, int64, pointer, and null-terminated string keys.
 * - Designed to be extensible for additional types as needed.
 * - For use in hash set, hash map, and similar data structures.
 */

#ifndef HASH_H
#define HASH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Knuth's multiplicative hash constant.
#define HASH_KNUTH 2654435761U

/**
 * @enum HashType
 * @brief Key type for hash and compare dispatch.
 */
typedef enum HashType {
    HASH_INT32, /**< 32-bit signed integer keys */
    HASH_INT64, /**< 64-bit signed integer keys */
    HASH_PTR, /**< Pointer keys */
    HASH_STR /**< Null-terminated string keys */
} HashType;

/**
 * @typedef HashFn
 * @brief Hash function pointer type.
 * @param key   Pointer to key to hash.
 * @param size  Size of the hash table (bucket count).
 * @param i     Probe number (for collision resolution).
 * @return Hash value in range [0, size-1].
 */
typedef uint64_t (*HashFn)(const void* key, uint64_t size, uint64_t i);

/**
 * @typedef HashCmp
 * @brief Comparison function pointer type.
 * @param a  Pointer to first key.
 * @param b  Pointer to second key.
 * @return 0 if equal, <0 if a < b, >0 if a > b.
 */
typedef int (*HashCmp)(const void* a, const void* b);

/**
 * @struct Hash
 * @brief Bundles key type, hash/compare function pointers, and key size for generic use.
 */
typedef struct Hash {
    HashFn function; /**< Hash function pointer */
    HashCmp compare; /**< Comparison function pointer */
    HashType type; /**< Key type tag (e.g. int32, str, ptr) */
    size_t size; /**< Key size in bytes */
} Hash;

/** Hash Functions for Supported Types */

/**
 * @brief Hash and compare for 32-bit int keys.
 */
uint64_t hash_int32(const void* key, uint64_t size, uint64_t i);
int hash_int32_cmp(const void* a, const void* b);

/**
 * @brief Hash and compare for 64-bit int keys.
 */
uint64_t hash_int64(const void* key, uint64_t size, uint64_t i);
int hash_int64_cmp(const void* a, const void* b);

/**
 * @brief Hash and compare for pointer keys.
 */
uint64_t hash_ptr(const void* key, uint64_t size, uint64_t i);
int hash_ptr_cmp(const void* a, const void* b);

/**
 * @brief Hash and compare for null-terminated string keys (djb2).
 */
uint64_t hash_str(const void* key, uint64_t size, uint64_t i);
int hash_str_cmp(const void* a, const void* b);

#ifdef __cplusplus
}
#endif

#endif  // HASH_H
