/**
 * Copyright © 2023 Austin Berrio
 *
 * @file examples/tokenizer.c
 * @brief Valerie uses a custom GPE tokenizer built from scratch in pure C.
 * @todo Need methods for creating tokenizers from a raw UTF-8 corpus.
 * @ref arXiv:1508.07909v5 [cs.CL] 10 Jun 2016
 * @ref arXiv:2505.24689 [cs.CL] 30 May 2025
 * @ref https://aclanthology.org/2025.coling-main.400/
 */

#include "logger.h"  // logging macros LOG_ERROR, etc.
#include "strext.h"
#include "map.h"  // HashMap* map

#include <stdint.h>
#include <stdio.h>  // IO
#include <stdlib.h>
#include <sys/types.h>  // ssize_t
#include <sys/mman.h>  // mmap/munmap

/**
 * Tokenizer Blueprint
 * @{
 */

#define VTKN_MAGIC 0x56544B4E
#define VTKN_VERSION 1

typedef struct TokenEntry {
    char* token;
    float score;
} TokenEntry;

typedef struct TokenSpecial {
    int bos_id;  // <s>
    int eos_id;  // </s>
    int pad_id;  // <pad>
} TokenSpecial;

typedef struct Tokenizer {
    int magic;
    int version;
    ssize_t vocab_size;  // vocab size
    TokenSpecial special;
    HashMap* token_to_id;
    HashMap* id_to_token;
} Tokenizer;

// @brief Load a tokenizer from a binary model file.
Tokenizer* tokenizer_create(const char* filepath);
void tokenizer_free(Tokenizer* t);

// @brief Convert a token id to its UTF-8 string.
int tokenizer_token_to_id(Tokenizer* t, const char* token);
// @brief Convert a UTF-8 token string to its id.
char* tokenizer_id_to_token(Tokenizer* t, const int id);

// @brief Encode a UTF-8 string into a sequence of token ids using greedy BPE.
int* tokenizer_prompt(Tokenizer* t, char* text, int* out_size);

/** @} */

/**
 * Map Corpus to memory
 * @{
 */

char* tokenizer_corpus_mmap(const char* filepath, ssize_t* out_size) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        LOG_ERROR("[Tokenizer] Failed to read corpus: '%s'", filepath);
        return NULL;
    }

    if (-1 == fseek(file, 0, SEEK_END)) {
        LOG_ERROR("[Tokenizer] Failed to seek end of corpus.");
        fclose(file);
        return NULL;
    }

    *out_size = ftell(file);
    if (-1 == *out_size) {
        LOG_ERROR("[Tokenizer] Failed to get corpus file size.");
        fclose(file);
        return NULL;
    }
    rewind(file);

    char* data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (!data || MAP_FAILED == data) {
        LOG_ERROR("[Tokenizer] Failed to map %ld bytes of corpus to memory", *out_size);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return data;
}

void tokenizer_corpus_unmap(char* corpus, ssize_t size) {
    if (corpus && 0 < size) {
        munmap(corpus, size);
    }
}

/** @} */

/**
 * Vocab clean up
 * @{
 */

void tokenizer_vocab_free(HashMap* vocab) {
    if (vocab) {
        hash_map_iter_free(vocab, NULL);
        hash_map_free(vocab);
    }
}

/** @} */

/**
 * Create Initial Vocab
 * @{
 */

HashMap* tokenizer_vocab_create(const char* data) {
    HashMap* vocab = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);
    if (!vocab) {
        return NULL;
    }

    size_t count = strlen(data);
    for (size_t i = 0; i < count - 1; i++) {
        // copy current
        char* current = calloc(2, sizeof(char));
        current[0] = data[i];
        current[1] = '\0';

        // copy next
        char* next = calloc(2, sizeof(char));
        next[0] = data[i + 1];
        next[1] = '\0';

        // Create the initial merge pair
        char* key = string_concat(current, next);

        // Insert into map
        int* freq = hash_map_search(vocab, key);
        if (!freq) {
            int* value = calloc(1, sizeof(int));
            *value = 1;
            hash_map_insert(vocab, key, value);
        } else {
            (*freq)++;
            free(key);  // already present, so free our duplicate
        }

        // Clean up scratch buffers
        free(current);
        free(next);

        // if the i-th key is freed here, it incites a double-free
    }
    return vocab;
}

/** @} */

/**
 * Create Vocab Stats
 * @{
 */

HashMap* tokenizer_pairs_create(HashMap* vocab) {
    HashMap* pairs = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);
    if (!vocab) {
        return NULL;
    }

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        char* key = (char*) entry->key;

        size_t sym_count = 0;
        char** syms = string_split(key, &sym_count);
        if (sym_count < 2) {
            string_split_free(syms, sym_count);
            continue;  // dropout
        }

        for (size_t i = 0; i < sym_count - 1; i++) {
            // Build pair key (e.g. {"a", "b"} -> "ab")
            char* new_key = string_join((char*[]) {syms[i], syms[i + 1]}, 2, (char*) "");

            // insert into map
            int* freq = hash_map_search(pairs, new_key);
            if (!freq) {
                int* value = calloc(1, sizeof(int));
                value = memcpy(value, (int*) entry->value, sizeof(int));
                hash_map_insert(pairs, new_key, value);
            } else {
                (*freq) += *(int*) entry->value;  // Update frequency
                free(new_key);  // Already present
            }
        }

        string_split_free(syms, sym_count);
    }

    return pairs;
}

/** @} */

/**
 * Create Vocab Merges
 * @{
 */

HashMap* tokenizer_merges_create(HashMap* vocab, const char* pair) {
    HashMap* new_vocab = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);

    // Split the pair into two symbols
    size_t pair_len = 0;
    char** pair_syms = string_split(pair, &pair_len);
    if (!pair_syms) {
        hash_map_free(new_vocab);
        return NULL;  // Failed to create pair
    }
    if (pair_len != 2) {
        hash_map_free(new_vocab);
        string_split_free(pair_syms, pair_len);
        return NULL;
    }

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        char* key = (char*) entry->key;

        size_t sym_count = 0;
        char** syms = string_split(key, &sym_count);

        // Build new symbol array after merging (pessimistic)
        size_t new_count = 0;
        char** new_syms = calloc(sym_count + 1, sizeof(char*));

        for (uint64_t i = 0; i < sym_count;) {
            // Try to merge at this position
            if (i + 1 < sym_count && 0 == string_compare(syms[i], pair_syms[0])
                && 0 == string_compare(syms[i + 1], pair_syms[1])) {
                // Merge the pair
                char* merged = string_concat(syms[i], syms[i + 1]);
                new_syms[new_count++] = merged;
                i += 2;  // Skip next symbol
            } else {
                new_syms[new_count++] = string_copy(syms[i]);
                i += 1;
            }
        }

        // Join new symbol array back into a string (space-delimited, if needed)
        char* new_key = string_join(new_syms, new_count, "");

        // Insert or update frequency in new vocab
        int* freq = hash_map_search(new_vocab, new_key);
        if (!freq) {
            int* value = calloc(1, sizeof(int));
            memcpy(value, entry->value, sizeof(int));
            hash_map_insert(new_vocab, new_key, value);
        } else {
            (*freq) += *(int*) entry->value;
            free(new_key);  // Already present
        }

        // Cleanup
        for (uint64_t j = 0; j < new_count; j++) {
            free(new_syms[j]);
        }
        free(new_syms);
        for (uint64_t j = 0; j < sym_count; j++) {
            free(syms[j]);
        }
        free(syms);
        // new_key: ownership passed to hash_map, or freed above if duplicate
    }

    string_split_free(pair_syms, pair_len);
    return new_vocab;
}

/** @} */

/**
 * Command Line Interface
 * @{
 */

void tokenizer_usage(const char* argv) {
    fprintf(stderr, "Usage: %s input.txt num_merges\n", argv);
}

/** @} */

int main(int argc, char* argv[]) {
    if (argc < 2) {
        tokenizer_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Read the input corpus from a plaintext file
    char* corpus_path = argv[1];

    // Get the number of merges
    uint64_t num_merges = 10;
    if (argc == 3) {
        num_merges = atoi(argv[2]);
    }

    // Read the corpus into memory
    ssize_t corpus_size = 0;
    char* corpus = tokenizer_corpus_mmap(corpus_path, &corpus_size);
    printf("[corpus start]\n%s\n[corpus end]\n", corpus);

    // Build the initial vocab from the corpus
    HashMap* vocab = tokenizer_vocab_create(corpus);
    if (!vocab) {
        tokenizer_corpus_unmap(corpus, corpus_size);
        return EXIT_FAILURE;
    }
    printf("Initialized vocab.");

    HashMapEntry* entry = NULL;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        printf("key='%s', value=%d\n", (uint8_t*) entry->key, *(int*) entry->value);
    }

    for (uint64_t i = 0; i < num_merges; i++) {
        printf("\n[MERGE %lu]\n", i + 1);
        HashMap* pairs = tokenizer_pairs_create(vocab);

        int best_freq = 0;
        char* best_pair = NULL;
        it = hash_map_iter(pairs);
        while ((entry = hash_map_next(&it))) {
            int freq = *(int*) entry->value;
            if (freq > best_freq) {
                best_freq = freq;
                if (best_pair) {
                    free(best_pair);
                }
                best_pair = string_copy(entry->key);
            }
        }

        if (!best_pair) {
            free(best_pair);
            tokenizer_vocab_free(pairs);
            break;
        }

        printf("Merging pair: '%s' freq: %d\n", best_pair, best_freq);

        HashMap* new_vocab = tokenizer_merges_create(vocab, best_pair);
        free(best_pair);
        tokenizer_vocab_free(pairs);
        tokenizer_vocab_free(vocab);
        vocab = new_vocab;
    }

    entry = NULL;
    it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        printf(
            "entry[%lu], entry->key='%s', entry->value=%d\n",
            it.index,
            (char*) entry->key,
            *(int*) entry->value
        );
    }

    // Clean up
    tokenizer_vocab_free(vocab);
    tokenizer_corpus_unmap(corpus, corpus_size);
    return EXIT_SUCCESS;
}
