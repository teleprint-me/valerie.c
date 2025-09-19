/**
 * @file      tokenizer/bpe.h
 * @brief     Byte-Pair Encoding (BPE) merges and model API.
 * @copyright Copyright © 2025 Austin Berrio
 *
 * This module implements the core BPE model and merge operations,
 * including training, serialization, and manipulation of BPE merges.
 *
 * @note All APIs are designed for use with space-delimited symbol strings
 *       (e.g., "l o o k e d"). Ownership of returned pointers is specified.
 */

#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include <stdbool.h>
#include <stddef.h>

#include "core/map.h"

/**
 * @def BPE_MAGIC
 * @brief Magic number for BPE merges file format identification ("pair", little-endian).
 */
#define BPE_MAGIC 0x70616972

/**
 * @def BPE_VERSION
 * @brief Current version of the BPE merges file format.
 */
#define BPE_VERSION 1

/**
 * @struct BPEMerge
 * @brief Represents a single merge operation in BPE: a pair of symbols and their frequency.
 *
 * @var BPEMerge::pair
 *   Space-delimited string of the merged symbol pair (e.g., "th e").
 *   Caller is responsible for managing memory.
 * @var BPEMerge::freq
 *   Frequency of this merge pair when selected.
 */
typedef struct BPEMerge {
    char* pair;  ///< Space-delimited symbol pair (heap-allocated, NUL-terminated).
    int freq;  ///< Frequency of the pair at merge time.
} BPEMerge;

/**
 * @struct BPEModel
 * @brief Represents a learned BPE model (sequence of merges).
 *
 * @var BPEModel::merges
 *   Dynamic array of BPEMerge records (heap-allocated).
 * @var BPEModel::count
 *   Number of valid merges in the array.
 * @var BPEModel::capacity
 *   Allocated capacity of the merges array (for growth).
 */
typedef struct BPEModel {
    BPEMerge* merges;  ///< Array of learned merges.
    size_t count;  ///< Number of merge steps.
    size_t capacity;  ///< Allocated capacity.
} BPEModel;

/**
 * @name BPE serialization
 * @brief Save/load BPE models from disk (binary format).
 * @{
 */

/**
 * @brief Save a BPE model to a binary file.
 *
 * @param model    Pointer to BPEModel to save.
 * @param path     Output file path (NUL-terminated string).
 * @return true on success, false on error.
 */
bool bpe_save(BPEModel* model, const char* path);

/**
 * @brief Load a BPE model from a binary file.
 *
 * @param path     Input file path (NUL-terminated string).
 * @return Newly allocated BPEModel pointer, or NULL on error. Caller must free with bpe_free().
 */
BPEModel* bpe_load(const char* path);

/** @} */

/**
 * @name BPE merge operations
 * @brief Train and apply BPE merges on vocabularies.
 * @{
 */

/**
 * @brief Collect all adjacent symbol pairs and their frequencies from a vocabulary.
 *
 * @param vocab   HashMap* of words to frequency (tokens as space-delimited symbol strings).
 * @return HashMap* mapping pair string ("A B") to int* frequency. Caller must free with
 * vocab_map_free().
 */
HashMap* bpe_pairs(HashMap* vocab);

/**
 * @brief Select the highest-frequency symbol pair from a pairs map.
 *
 * @param pairs      HashMap* mapping pair strings to int* frequency.
 * @param out_freq   (Optional) Pointer to int to receive the best frequency.
 * @return Heap-allocated NUL-terminated string for the best pair ("A B"), or NULL if empty. Caller
 * must free.
 */
char* bpe_best(HashMap* pairs, int* out_freq);

/**
 * @brief Merge all occurrences of the given best pair in the vocabulary.
 *
 * @param vocab      HashMap* vocabulary (token string -> int* freq).
 * @param best_pair  NUL-terminated string for the pair to merge ("A B").
 * @return New HashMap* with merged pairs. Caller must free with vocab_map_free().
 */
HashMap* bpe_merges(HashMap* vocab, const char* best_pair);

/**
 * @brief Train a BPE model with a fixed number of merge steps.
 *
 * @param vocab      Input vocabulary (token string -> int* freq). Not consumed or freed.
 * @param n_merges   Number of merges to perform.
 * @param verbose    If true, prints intermediate steps.
 * @return Pointer to a new BPEModel on success; NULL on error. Caller must free with bpe_free().
 */
BPEModel* bpe_train(HashMap* vocab, size_t n_merges, bool verbose);

/**
 * @brief Free all memory associated with a BPEModel.
 *
 * @param model   Pointer to BPEModel to free. Safe to pass NULL.
 */
void bpe_free(BPEModel* model);

/** @} */

#endif  // TOKENIZER_BPE_H
