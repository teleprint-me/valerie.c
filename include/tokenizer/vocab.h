/**
 * @file      tokenizer/vocab.h
 * @brief     Vocabulary mapping and serialization utilities for tokenization pipelines.
 * @copyright Copyright © 2025 Austin Berrio
 *
 * Provides utilities for building, saving, loading, and printing vocabulary maps
 * using a string-to-int frequency mapping (HashMap).
 *
 * Format and serialization is designed for compact storage and fast loading.
 */

#ifndef TOKENIZER_VOCAB_H
#define TOKENIZER_VOCAB_H

#include <stdbool.h>
#include <stddef.h>

#include "core/map.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def VOCAB_MAGIC
 * @brief Magic number for vocab file format identification ("syms", little-endian).
 */
#define VOCAB_MAGIC 0x73796D73

/**
 * @def VOCAB_VERSION
 * @brief Current version of the vocab file format.
 */
#define VOCAB_VERSION 1

/**
 * @defgroup VocabMapUtils Vocab Map Utilities
 * @brief General utilities for working with vocab hash maps.
 * @{
 */

/**
 * @brief Creates a copy of a given hash map containing word frequencies.
 *
 * The caller is responsible for freeing the memory of the returned hash map.
 *
 * @param m The input hash map to be copied.
 *
 * @return A new hash map with the same key-value pairs as the input hash map.
 */
HashMap* vocab_map_copy(HashMap* m);

/**
 * @brief Frees all memory associated with a vocab map.
 *
 * Calls hash_map_iter_free_all() on the map. Safe to call with NULL.
 *
 * @param m Pointer to the vocab map (HashMap*), or NULL.
 */
void vocab_map_free(HashMap* m);

/**
 * @brief Prints all tokens and frequencies in a vocab map to stdout.
 *
 * Format: <token> -> <frequency>
 *
 * @param m Pointer to the vocab map (HashMap*), or NULL.
 */
void vocab_map_log(HashMap* m);

/** @} */

/**
 * @defgroup VocabSerialization Vocab Map Serialization
 * @brief Save/load vocabulary maps to and from disk.
 * @{
 */

/**
 * @brief Saves a vocab map to a binary file.
 *
 * Writes a compact, versioned binary format:
 *   [int32] magic ('syms')
 *   [int32] version (currently 1)
 *   [int32] count (number of entries)
 *   [int32] size  (hash map capacity)
 *   For each entry:
 *     [int32] token length (bytes)
 *     [char[]] token bytes
 *     [int32] frequency
 *
 * All values are little-endian.
 *
 * @param m Pointer to vocab map (HashMap*), must not be NULL.
 * @param path Path to the file to write.
 * @return true on success, false on error.
 */
bool vocab_map_save(HashMap* m, const char* path);

/**
 * @brief Loads a vocab map from a binary file.
 *
 * Expects the format written by vocab_map_save().
 *
 * @param path Path to the binary vocab file.
 * @return Pointer to newly-allocated vocab map (HashMap*), or NULL on error.
 */
HashMap* vocab_map_load(const char* path);

/** @} */

/**
 * @defgroup VocabTextIO Text File Utilities
 * @brief Reading plain text files for vocabulary extraction.
 * @{
 */

/**
 * @brief Reads the contents of a text file into a newly-allocated buffer.
 *
 * @param path Path to the text file.
 * @return Null-terminated string buffer, or NULL on failure.
 *         Caller must free the returned buffer.
 */
char* vocab_read_text(const char* path);

/** @} */

/**
 * @defgroup VocabFrequencies Frequency Mapping
 * @brief Construct frequency hash maps from text or symbol lists.
 * @{
 */

/**
 * @brief Creates a word-frequency map from plain text.
 *
 * Tokenizes input on whitespace and counts unique word occurrences.
 *
 * @param text Null-terminated input text.
 * @return HashMap* mapping (word string) -> (int* frequency), or NULL on failure.
 */
HashMap* vocab_create_frequencies(const char* text);

/**
 * @brief Creates a symbol-frequency map from a word-frequency map.
 *
 * Splits each word into its constituent symbols (one per codepoint),
 * joins them as space-separated strings, and aggregates frequencies.
 *
 * @param words HashMap* mapping (word string) -> (int* frequency).
 * @return HashMap* mapping (symbol sequence) -> (int* frequency).
 */
HashMap* vocab_create_symbols(HashMap* words);

/** @} */

/**
 * @defgroup VocabFactory Vocabulary Factory
 * @brief Pipeline for building vocabularies from text or file.
 * @{
 */

/**
 * @brief Tokenizes input text into a symbol-frequency vocab map.
 *
 * Equivalent to: vocab_create_frequencies() + vocab_create_symbols().
 *
 * @param text Null-terminated input text.
 * @return HashMap* mapping (symbol sequence) -> (int* frequency), or NULL on failure.
 */
HashMap* vocab_tokenize(const char* text);

/**
 * @brief Builds a vocab map directly from a plain text file.
 *
 * Reads the file, tokenizes, and returns the resulting symbol-frequency map.
 *
 * @param path Path to the plain text file.
 * @return HashMap* mapping (symbol sequence) -> (int* frequency), or NULL on failure.
 */
HashMap* vocab_build(const char* path);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // TOKENIZER_VOCAB_H
