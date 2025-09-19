/**
 * @copyright Copyright © 2025 Austin Berrio
 * @file      tokenizer/bpe.h
 * @brief
 *
 */

#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include <stdbool.h>
#include <stddef.h>

#include "core/map.h"

/**
 * @def BPE_MAGIC
 * @brief Magic number for merges file format identification ("pair", little-endian).
 */
#define BPE_MAGIC 0x70616972

/**
 * @def BPE_VERSION
 * @brief Current version of the merges file format.
 */
#define BPE_VERSION 1

typedef struct BPEMerge {
    char* pair;
    int freq;
} BPEMerge;

typedef struct BPEModel {
    BPEMerge* merges;
    size_t count;
    size_t capacity;
} BPEModel;

/**
 * BPE serialization
 * @{
 */

bool bpe_save(BPEModel* model, const char* path);

BPEModel* bpe_load(const char* path);

/** @} */

// collect vocab pairs
// once all pairs have been exhausted,
// the pairs function must return NULL to indicate the end of operation
HashMap* bpe_pairs(HashMap* vocab);

char* bpe_best(HashMap* pairs, int* out_freq);

HashMap* bpe_merges(HashMap* vocab, const char* best_pair);

BPEModel* bpe_train(HashMap* vocab, size_t n_merges, bool verbose);

void bpe_free(BPEModel* model);

#endif  // TOKENIZER_BPE_H
