/**
 * @file      hash.h
 * @brief     General-purpose hash table API for sets and maps in C.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * This header defines the public API for generic hash-based containers,
 * supporting flexible hash and comparison function pointers for multiple key types.
 *
 * Features:
 * - Supports `int32_t`, `int64_t`, C-strings, and pointer keys.
 * - Extensible design for adding custom key types as needed.
 * - Usable for both hash sets and hash maps.
 *
 * @note
 * Comparison functions used with the hash API **must**:
 *   - Return `0` for equality.
 *   - Return non-zero for inequality.
 *
 * @note
 * Supported key types:
 *   - Integers (`int32_t`, `int64_t`)
 *   - Strings (`char*`, null-terminated)
 *   - Memory addresses (`uintptr_t`)
 *
 * @note
 * Thread Safety:
 *   - The hash interface itself is thread-agnostic.
 *   - **Consumers** must ensure all hash operations are protected with locks as needed.
 *
 * @note
 * Collision Handling:
 *   - Linear probing is used for collision resolution.
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
    HASH_EXISTS, /**< Duplicate key insertion attempted. */
    HASH_NOT_FOUND, /**< Key not found. */
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
 * @brief Iterator for traversing active entries in a hash table.
 */
typedef struct HashIt {
    Hash* table; /**< Pointer to the hash table being iterated. */
    size_t index; /**< Current index within the table. */
} HashIt;

/**
 * @brief Callback type for freeing hash values.
 *
 * Defines a function pointer type for freeing value pointers
 * during hash table iteration or cleanup operations.
 *
 * @param value Pointer to the value to free.
 */
typedef void (*HashValueFree)(void*);

/**
 * @defgroup functions Functions for Supported Types
 * @{
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

/**
 * @defgroup lifecycle Table Life-cycle
 *  @{
 */

/**
 * @brief Create a new hash table with the specified capacity and key type.
 *
 * @param capacity  Initial capacity (number of slots). Minimum is 10.
 * @param type      HashType enum (e.g., HASH_INT32, HASH_STR, etc).
 * @return          Pointer to new Hash instance, or NULL on allocation failure.
 */
Hash* hash_create(size_t capacity, HashType type);

/**
 * @brief Free all memory for a hash table, including its internal storage.
 *
 * @param h Pointer to the hash table to free.
 */
void hash_free(Hash* h);

/** @} */

/**
 * @defgroup queries Table Queries
 *  @{
 */

/**
 * @brief Return the current number of valid entries in the hash table.
 *
 * @param h Pointer to the hash table.
 * @return  Number of active entries, or 0 if NULL.
 */
size_t hash_count(const Hash* h);

/**
 * @brief Return the current capacity (number of slots) of the hash table.
 *
 * @param h Pointer to the hash table.
 * @return  Capacity, or 0 if NULL.
 */
size_t hash_capacity(const Hash* h);

/**
 * @brief Return the size in bytes of a single key, according to its type.
 *
 * @param h Pointer to the hash table.
 * @return  Key size in bytes, or 0 if NULL.
 */
size_t hash_size(const Hash* h);

/**
 * @brief Return the key type (HashType enum) of the hash table.
 *
 * @param h Pointer to the hash table.
 * @return  HashType value.
 */
HashType hash_type(const Hash* h);

/**
 * @brief Check if a hash table is valid (non-NULL, has storage, capacity > 0).
 *
 * @param h Pointer to the hash table.
 * @return  true if valid; false otherwise.
 */
bool hash_is_valid(const Hash* h);

/**
 * @brief Check if a hash table entry is valid (non-NULL key).
 *
 * @param e Pointer to the HashEntry.
 * @return  true if entry is valid; false otherwise.
 */
bool hash_entry_is_valid(const HashEntry* e);

/**
 * @brief Check if two hash tables are comparable (same HashType).
 *
 * @param a Pointer to first hash table.
 * @param b Pointer to second hash table.
 * @return  true if types match; false otherwise.
 */
bool hash_cmp_is_valid(const Hash* a, const Hash* b);

/**
 * @brief Check if a hash table has a recognized and supported HashType.
 *
 * @param h Pointer to the hash table.
 * @return  true if type is valid; false otherwise.
 */
bool hash_type_is_valid(const Hash* h);

/** @} */

/**
 * @defgroup operations Table Operations
 *  @{
 */

/**
 * @brief Insert a key-value pair into the hash table.
 *
 * @param h     Pointer to the hash table.
 * @param key   Pointer to key data (must match table's key type).
 * @param value Pointer to value data (may be NULL for set-like usage).
 * @return      HashState code (HASH_SUCCESS, HASH_EXISTS, etc).
 */
HashState hash_insert(Hash* h, void* key, void* value);

/**
 * @brief Resize the hash table to a new capacity (rehashes all keys).
 *
 * @param h            Pointer to the hash table.
 * @param new_capacity New capacity (must be > current).
 * @return             HASH_SUCCESS on success; HASH_ERROR or other state on failure.
 */
HashState hash_resize(Hash* h, size_t new_capacity);

/**
 * @brief Remove a key (and value) from the hash table.
 *
 * @param h   Pointer to the hash table.
 * @param key Pointer to key data.
 * @return    HASH_SUCCESS if deleted, HASH_NOT_FOUND if not present.
 */
HashState hash_delete(Hash* h, const void* key);

/**
 * @brief Remove all entries from the hash table, but do not free the table.
 *
 * @param h Pointer to the hash table.
 * @return  HASH_SUCCESS on success; HASH_ERROR on invalid input.
 */
HashState hash_clear(Hash* h);

/**
 * @brief Search for a key in the hash table and return its value.
 *
 * @param h   Pointer to the hash table.
 * @param key Pointer to key data.
 * @return    Pointer to value if found; NULL otherwise.
 */
void* hash_search(Hash* h, const void* key);

/** @} */

/**
 * @defgroup iterator Table Iterator
 *  @{
 */

/**
 * @brief Create a new iterator for traversing hash table entries.
 *
 * @param h Pointer to the hash table.
 * @return  HashIt iterator struct.
 */
HashIt hash_iter(Hash* h);

/**
 * @brief Check if an iterator is valid (table and entries are non-NULL).
 *
 * @param it Pointer to the iterator.
 * @return   true if valid; false otherwise.
 */
bool hash_iter_is_valid(HashIt* it);

/**
 * @brief Get the next valid entry in the hash table (iterates forward).
 *
 * @param it Pointer to the iterator.
 * @return   Pointer to HashEntry, or NULL if end reached.
 */
HashEntry* hash_iter_next(HashIt* it);

/**
 * @brief Print debug info and all valid keys in the table to the logger.
 *
 * @param h Pointer to the hash table.
 */
void hash_iter_log(Hash* h);

/**
 * @brief Free all keys (always allocated) and values (using value_free) in the table.
 *
 * @param h          Pointer to the hash table.
 * @param value_free Function pointer to free value memory (NULL to ignore values).
 */
void hash_iter_free_kv(Hash* h, HashValueFree value_free);

/**
 * @brief Free all keys/values in the table and then free the table itself.
 *
 * @param h          Pointer to the hash table.
 * @param value_free Function pointer to free value memory (NULL to ignore values).
 */
void hash_iter_free_all(Hash* h, HashValueFree value_free);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // HASH_H
