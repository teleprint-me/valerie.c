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

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Knuth's multiplicative hash constant.
#define HASH_KNUTH 2654435761U

/**
 * @enum HashState
 * @brief Possible outcomes for hash operations.
 */
typedef enum HashState {
    HASH_SUCCESS, /**< Operation completed successfully. */
    HASH_ERROR, /**< General error occurred during operation. */
    HASH_KEY_EXISTS, /**< Duplicate key insertion attempted. */
    HASH_KEY_NOT_FOUND, /**< Key not found. */
    HASH_FULL /**< Reached maximum capacity. */
} HashState;

/**
 * @enum HashType
 * @brief Key type for hash and compare dispatch.
 */
typedef enum HashType {
    HASH_INT32, /**< 32-bit signed integer keys */
    HASH_INT64, /**< 64-bit signed integer keys */
    HASH_PTR, /**< Pointer keys */
    HASH_STR, /**< Null-terminated string keys */
    HASH_UNK /**< Unsupported data type */
} HashType;

/**
 * @struct HashEntry
 * @brief Represents a key-value pair entry in a hash table.
 * @note All key-value pairs are managed by the interface or user.
 *       Sets should be agnostic and only handles keys.
 *       Maps should be agnostic and handles both key-value pairs.
 *       By default, users are responsible for allocations of keys and/or values.
 *       Allocation details may be implementation dependent.
 */
typedef struct HashEntry {
    void* key; /**< Pointer to the key (HashType). */
    void* value; /**< Pointer to the associated value (Any). */
} HashEntry;

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
    HashEntry* entries; /**< Array of entries */

    HashType type; /**< Key type tag (e.g. int32, str, ptr) */

    size_t count; /**< Number of active entries. */
    size_t capacity; /**< Total capacity of all entries */
    size_t size; /**< Key size in bytes */

    pthread_mutex_t lock; /**< Mutex for thread safety. */

    HashFn fn; /**< Hash function pointer */
    HashCmp cmp; /**< Comparison function pointer */
} Hash;


/**
 * @section Hash life-cycle
 */

/**
 * @brief Creates a new hash table.
 *
 * @param capacity Initial capacity of the table.
 * @param type Type of keys (integer, string, or address).
 * @return Pointer to the new hash table, or NULL on failure.
 */
Hash* hash_create(size_t capacity, HashType type);

/**
 * @brief Frees a hash table and all associated memory.
 *
 * @param table Pointer to the hash table to free.
 */
void hash_free(Hash* h);

/** @} */

/**
 * @section Hash utils
 */

size_t hash_count(Hash* h);

size_t hash_capacity(Hash* h);

size_t hash_size(Hash* h);

HashType hash_type(Hash* h);

bool hash_is_valid(Hash* h);

bool hash_entry_is_valid(HashEntry* e);

/** @} */

/** 
 * @section Hash Functions for Supported Types
 */

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

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // HASH_H
