/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/map.h
 * @brief Minimalistic hash table implementation providing mapping between integers, strings, and
 * memory addresses.
 *
 * The Map Interface provides a minimal dictionary-like API supporting insertion, search, deletion,
 * and clearing for keys of type integer, string, or memory address.
 *
 * @note Comparison functions must return 0 for equality, non-zero otherwise.
 * @note Thread Safety: Uses mutexes for thread-safe operations; calls must be within critical
 * sections.
 * @note Collision resolution: Uses linear probing.
 */

#ifndef HASH_MAP_LINEAR_H
#define HASH_MAP_LINEAR_H

#include <stdint.h>
#include <pthread.h>

/**
 * @brief Possible outcomes for hash table operations.
 */
typedef enum HashMapState {
    HASH_MAP_STATE_SUCCESS, /**< Operation completed successfully. */
    HASH_MAP_STATE_ERROR, /**< General error occurred during operation. */
    HASH_MAP_STATE_KEY_EXISTS, /**< Duplicate key insertion attempted. */
    HASH_MAP_STATE_KEY_NOT_FOUND, /**< Key not found in the table. */
    HASH_MAP_STATE_FULL /**< Hash table has reached maximum capacity. */
} HashMapState;

/**
 * @brief Types of keys supported by the hash table.
 */
typedef enum HashMapKeyType {
    HASH_MAP_KEY_TYPE_INTEGER, /**< Keys are integers (uint64_t). */
    HASH_MAP_KEY_TYPE_STRING, /**< Keys are null-terminated strings. */
    HASH_MAP_KEY_TYPE_ADDRESS /**< Keys are memory addresses (uintptr_t). */
} HashMapKeyType;

/**
 * @brief Represents a key-value pair entry in the hash table.
 */
typedef struct HashMapEntry {
    void* key; /**< Pointer to the key (type depends on HashMapKeyType). */
    void* value; /**< Pointer to the associated value. */
} HashMapEntry;

/**
 * @brief Core hash table structure.
 */
typedef struct HashMap {
    HashMapEntry* entries; /**< Array of hash entries. */
    uint64_t count; /**< Current number of entries in the table. */
    uint64_t size; /**< Total capacity of the hash table. */
    HashMapKeyType type; /**< Type of keys stored. */
    pthread_mutex_t thread_lock; /**< Mutex for thread safety. */

    uint64_t (*hash)(const void* key, uint64_t size, uint64_t i); /**< Hash function with probing. */
    int (*compare)(const void* key1, const void* key2); /**< Key comparison function. */
} HashMap;

/**
 * @brief Iterator for traversing active entries in a hash map.
 */
typedef struct HashMapIterator {
    HashMap* table; /**< Pointer to the hash table being iterated. */
    uint64_t index; /**< Current index within the table. */
} HashMapIterator;

/**
 * @brief Callback type for freeing hash map values.
 *
 * Defines a function pointer type for freeing value pointers
 * during hash map iteration or cleanup operations.
 *
 * @param value Pointer to the value to free.
 */
typedef void (*HashMapValueFree)(void*);

/**
 * @name Life-cycle Management
 * @{
 */

/**
 * @brief Creates a new hash table.
 *
 * @param initial_size Initial capacity of the table.
 * @param key_type Type of keys (integer, string, or address).
 * @return Pointer to the new hash table, or NULL on failure.
 */
HashMap* hash_map_create(uint64_t initial_size, HashMapKeyType key_type);

/**
 * @brief Frees a hash table and all associated memory.
 *
 * @param table Pointer to the hash table to free.
 */
void hash_map_free(HashMap* table);

/** @} */

/**
 * @name Core Hash Operations
 * @note Thread-safe: acquires internal lock during operation.
 * @{
 */

/**
 * @brief Inserts a key-value pair into the hash table.
 *
 * If the table exceeds its load factor threshold, it will resize
 * automatically before insertion.
 *
 * @param table Pointer to the hash table.
 * @param key Pointer to the key to insert.
 * @param value Pointer to the value to associate with the key.
 * @return HASH_MAP_STATE_SUCCESS on success, or an error code.
 *
 * @note Automatically resizes if capacity is insufficient.
 */
HashMapState hash_map_insert(HashMap* table, const void* key, void* value);

/**
 * @brief Resizes the hash table to a new capacity.
 *
 * @param table Pointer to the hash table.
 * @param new_size Desired new capacity.
 * @return HASH_MAP_STATE_SUCCESS on success, HASH_MAP_STATE_ERROR on failure.
 */
HashMapState hash_map_resize(HashMap* table, uint64_t new_size);

/**
 * @brief Deletes a key and its associated value from the hash table.
 *
 * @param table Pointer to the hash table.
 * @param key Pointer to the key to delete.
 * @return HASH_MAP_STATE_SUCCESS if deletion succeeded, HASH_MAP_STATE_KEY_NOT_FOUND if not found.
 */
HashMapState hash_map_delete(HashMap* table, const void* key);

/**
 * @brief Removes all entries from the hash table.
 *
 * @param table Pointer to the hash table.
 * @return HASH_MAP_STATE_SUCCESS on success, HASH_MAP_STATE_ERROR on failure.
 */
HashMapState hash_map_clear(HashMap* table);

/**
 * @brief Searches for a key in the hash table.
 *
 * @param table Pointer to the hash table.
 * @param key Pointer to the key to search.
 * @return Pointer to the associated value, or NULL if not found.
 */
void* hash_map_search(HashMap* table, const void* key);

/**
 * @brief Returns the number of buckets allocated in the hash map.
 *
 * The "size" reflects the total number of buckets, not the current number
 * of elements. Used for capacity/resize logic and iterator bounds.
 *
 * @param table Pointer to the hash map, or NULL.
 * @return Number of buckets allocated, or 0 if table is NULL.
 */
uint64_t hash_map_size(HashMap* table);

/**
 * @brief Returns the number of active (non-empty) entries in the hash map.
 *
 * The "count" tracks how many key-value pairs are stored, not the capacity.
 * Updated on insert/delete. Prefer this over iterating for fast lookups.
 *
 * @param table Pointer to the hash map, or NULL.
 * @return Number of stored key-value pairs, or 0 if table is NULL.
 */
uint64_t hash_map_count(HashMap* table);

/** @} */

/**
 * @name Hash Iterator
 * @warning Iterators are not thread-safe. External locking is required if the hash map may be
 * mutated concurrently during iteration.
 * @{
 */

/**
 * @brief Initializes a hash map iterator.
 *
 * Returns an iterator positioned at the start of the table.
 * Pass to hash_map_next() to traverse key-value entries.
 *
 * @param table Pointer to the hash map to iterate.
 * @return Initialized iterator.
 */
HashMapIterator hash_map_iter(HashMap* table);

/**
 * @brief Advances the iterator and returns the next valid entry.
 *
 * Skips empty or deleted buckets. Returns entries in arbitrary order
 * (not sorted). Returns NULL when all entries are exhausted.
 *
 * @param iter Pointer to an iterator. Must not be NULL.
 * @return Pointer to the next valid HashMapEntry, or NULL if done.
 */
HashMapEntry* hash_map_next(HashMapIterator* iter);

/**
 * @brief Counts the number of valid entries via iteration.
 *
 * Slower than hash_map_count(), but useful if external filtering or
 * validation is needed. Mostly a convenience wrapper.
 *
 * @param table Pointer to the hash map, or NULL.
 * @return Number of valid entries found by iteration.
 */
uint64_t hash_map_iter_count(HashMap* table);

/**
 * @brief Iterates and frees all keys and values in the hash map.
 *
 * Frees each key using memory_free(), and each value with either
 * the provided value_free() function (if non-NULL), or memory_free().
 * Does not deallocate the hash map structure itself.
 *
 * @param table Pointer to the hash map.
 * @param value_free Optional callback to free value pointers (may be NULL).
 */
void hash_map_iter_free_kv(HashMap* table, HashMapValueFree value_free);

/**
 * @brief Frees all keys and values, then deallocates the entire map.
 *
 * Calls hash_map_iter_free_kv() with the standard free() function,
 * then releases the table with hash_map_free().
 *
 * @param table Pointer to the hash map.
 * @param value_free Optional callback to free value pointers (may be NULL).
 */
void hash_map_iter_free_all(HashMap* table, HashMapValueFree value_free);

/** @} */

/**
 * @name Integer Key Support
 * @{
 */

/**
 * @brief Hash function for integer keys with linear probing.
 *
 * @param key Pointer to the integer key.
 * @param size Size of the hash table.
 * @param i Probe index for collision resolution.
 * @return Hash index for the given key.
 */
uint64_t hash_integer(const void* key, uint64_t size, uint64_t i);

/**
 * @brief Compares two integer keys.
 *
 * @param key1 Pointer to first integer key.
 * @param key2 Pointer to second integer key.
 * @return 0 if equal, non-zero otherwise.
 */
int hash_integer_compare(const void* key1, const void* key2);

/**
 * @brief Searches for an integer key in the hash table.
 *
 * @param table Pointer to the hash table.
 * @param key Pointer to the integer key to search.
 * @return Pointer to associated value, or NULL if not found.
 */
int32_t* hash_integer_search(HashMap* table, const void* key);

/** @} */

/**
 * @name String Key Support
 * @{
 */

/**
 * @brief Computes DJB2 hash for a string key.
 *
 * @param string Null-terminated string key.
 * @return Hash value of the string.
 */
uint64_t hash_djb2(const char* string);

/**
 * @brief Hash function for string keys with linear probing.
 *
 * @param key Pointer to the string key.
 * @param size Size of the hash table.
 * @param i Probe index.
 * @return Hash index for the given key.
 */
uint64_t hash_string(const void* key, uint64_t size, uint64_t i);

/**
 * @brief Compares two string keys.
 *
 * @param key1 Pointer to first string key.
 * @param key2 Pointer to second string key.
 * @return 0 if equal, non-zero otherwise.
 */
int hash_string_compare(const void* key1, const void* key2);

/**
 * @brief Searches for a string key in the hash table.
 *
 * @param table Pointer to the hash table.
 * @param key Pointer to the string key to search.
 * @return Pointer to associated value, or NULL if not found.
 */
char* hash_string_search(HashMap* table, const void* key);

/** @} */

/**
 * @name Address Key Support
 * @{
 */

/**
 * @brief Hash function for address keys with linear probing.
 *
 * @param key Pointer to the address key.
 * @param size Size of the hash table.
 * @param i Probe index.
 * @return Hash index for the given key.
 */
uint64_t hash_address(const void* key, uint64_t size, uint64_t i);

/**
 * @brief Compares two address keys.
 *
 * @param key1 Pointer to first address key.
 * @param key2 Pointer to second address key.
 * @return 0 if equal, non-zero otherwise.
 */
int hash_address_compare(const void* key1, const void* key2);

/**
 * @brief Searches for an address key in the hash table.
 *
 * @param table Pointer to the hash table.
 * @param key Pointer to the address key to search.
 * @return Pointer to associated value, or NULL if not found.
 */
void* hash_address_search(HashMap* table, const void* key);

/** @} */

#endif  // HASH_MAP_LINEAR_H
