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

#include "memory.h"
#include "logger.h"  // logging macros LOG_ERROR, etc.
#include "map.h"  // HashMap* map
#include "utf8/byte.h"
#include "utf8/codepoint.h"
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
#define VTKN_META "\u2581"  // UTF-8 marker '▁'
#define VTKN_PRE \
    "('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+)"

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

// @brief Map a corpus from plaintext to memory for training.
char* tokenizer_corpus_mmap(const char* filepath, ssize_t* out_size);
// @brief Free mapped corpus from memory.
void tokenizer_corpus_unmap(char* corpus, ssize_t size);

// @brief Create a base vocab from the corpus
HashMap* tokenizer_vocab_create(char** tokens, uint64_t token_count);
// @brief Free vocab resources
void tokenizer_vocab_free(HashMap* vocab);

// @brief Calculate frequencies of pairs of adjacent symbols in the vocabulary.
HashMap* tokenizer_pairs_create(HashMap* vocab);
void tokenizer_pairs_free(HashMap* pairs);

// @brief Merge a given pair of symbols in the vocabulary.
HashMap* tokenizer_merges_create(HashMap* vocab, const char* pair);
void tokenizer_merges_free(HashMap* merges);

// @brief Load a tokenizer from a binary model file.
Tokenizer* tokenizer_create(const char* filepath);
// @brief Free tokenizer resources.
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

    char* data = mmap(NULL, *out_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(file), 0);
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
 * Create the tokenizer vocab
 * @{
 */

HashMap* tokenizer_vocab_create(char** tokens, uint64_t token_count) {
    HashMap* vocab = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);

    for (uint64_t i = 0; i < token_count; i++) {
        uint8_t* token = (uint8_t*) tokens[i];
        uint64_t cpts_count = 0;
        uint8_t** cpts = utf8_byte_split(token, &cpts_count);

        // Compute buffer size for the joined key
        size_t buf_size = 0;
        for (uint64_t j = 0; j < cpts_count; j++) {
            buf_size += utf8_byte_count(cpts[j]) + 1;  // +1 for space
        }
        buf_size += utf8_byte_count((const uint8_t*) "</w>") + 1;  // for "</w>" + null terminator

        // Build the key by joining codepoints with spaces, then appending "</w>"
        uint8_t* key = memory_alloc(buf_size, alignof(uint8_t));
        key[0] = '\0';

        for (uint64_t j = 0; j < cpts_count; j++) {
            uint8_t* prev = key;
            key = utf8_byte_cat(key, cpts[j]);
            memory_free(prev);

            if (j + 1 < cpts_count) {
                prev = key;
                key = utf8_byte_cat(key, (const uint8_t*) " ");
                memory_free(prev);
            }
        }
        // Append "</w>"
        uint8_t* prev = key;
        key = utf8_byte_cat(key, (const uint8_t*) "</w>");
        memory_free(prev);

        // Insert into map
        int* frequency = hash_map_search(vocab, (char*) key);
        if (!frequency) {
            int* value = memory_alloc(sizeof(int), alignof(int));
            *value = 1;
            hash_map_insert(vocab, (char*) key, value);
        } else {
            (*frequency)++;
            memory_free(key);  // already present, so free our duplicate
        }

        utf8_byte_split_free(cpts, cpts_count);
    }
    return vocab;
}

void tokenizer_vocab_free(HashMap* vocab) {
    if (vocab) {
        hash_map_iter_free(vocab, NULL);
        hash_map_free(vocab);
    }
}

/** @} */

/**
 * Create the vocab frequency pairs
 * @{
 */

HashMap* tokenizer_pairs_create(HashMap* vocab) {
    HashMap* pairs = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);
    HashMapIterator it = hash_map_iter(vocab);
    HashMapEntry* entry;

    while ((entry = hash_map_next(&it))) {
        uint8_t* word = (uint8_t*) entry->key;
        int frequency = *(int*) entry->value;

        uint64_t symbol_count = 0;
        uint8_t** symbols = utf8_byte_split_delim(word, (const uint8_t*) " ", &symbol_count);
        if (symbol_count < 2) {
            utf8_byte_split_free(symbols, symbol_count);
            continue;
        }

        for (uint64_t i = 0; i < symbol_count - 1; i++) {
            // Join symbols[i] + " " + symbols[i+1] as new string
            uint8_t* pair = utf8_byte_join((uint8_t*[]) {symbols[i], symbols[i + 1]}, 2, (const uint8_t*) " ");
            int* pair_freq = hash_map_search(pairs, pair);
            if (!pair_freq) {
                int* value = memory_alloc(sizeof(int), alignof(int));
                *value = frequency;
                hash_map_insert(pairs, pair, value);
            } else {
                *pair_freq += frequency;
                memory_free(pair);  // Already exists
            }
        }

        utf8_byte_split_free(symbols, symbol_count);
    }

    return pairs;
}

void tokenizer_pairs_free(HashMap* pairs) {
    if (pairs) {
        hash_map_iter_free(pairs, NULL);
        hash_map_free(pairs);
    }
}

/** @} */

/**
 * Create the merged frequency pairs
 * @{
 */

HashMap* tokenizer_merges_create(HashMap* vocab, const char* pair) {
    printf("pair='%s'\n", pair);
    // Split the pair (e.g., "l l" => ["l", "l"])
    uint64_t pair_len = 0;
    uint8_t** pair_syms = NULL;
    if (utf8_byte_count((const uint8_t*) pair) > 1 && strchr(pair, ' ')) {
        pair_syms = utf8_byte_split_delim((const uint8_t*) pair, (const uint8_t*) " ", &pair_len);
    } else {
        pair_syms = utf8_byte_split((const uint8_t*) pair, &pair_len);
    }

    HashMap* new_vocab = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        // Split word into symbols
        uint64_t sym_count = 0;
        uint8_t** syms = utf8_byte_split_delim(entry->key, (const uint8_t*) " ", &sym_count);

        // Build new symbol array after merging (pessimistic)
        uint8_t** new_syms = memory_alloc(sizeof(uint8_t*) * (sym_count + 1), alignof(uint8_t*));
        uint64_t new_count = 0;

        for (uint64_t i = 0; i < sym_count;) {
            // Try to merge at this position
            if (i + 1 < sym_count && 0 == utf8_byte_cmp(syms[i], pair_syms[0])
                && 0 == utf8_byte_cmp(syms[i + 1], pair_syms[1])) {
                // Merge the pair
                new_syms[new_count++] = utf8_byte_cat(syms[i], syms[i + 1]);
                i += 2;
            } else {
                new_syms[new_count++] = utf8_byte_copy(syms[i]);
                i += 1;
            }
        }

        // Join back into word (with spaces)
        uint8_t* new_word = utf8_byte_join(new_syms, new_count, (const uint8_t*) " ");

        // Insert into new_vocab
        int* freq = hash_map_search(new_vocab, new_word);
        if (!freq) {
            int* val = memory_alloc(sizeof(int), alignof(int));
            *val = *(int*) entry->value;
            hash_map_insert(new_vocab, new_word, val);
        } else {
            *freq += *(int*) entry->value;
            memory_free(new_word);
        }

        // Free temp arrays
        for (uint64_t j = 0; j < new_count; j++) {
            memory_free(new_syms[j]);
        }
        free(new_syms);
        utf8_byte_split_free(syms, sym_count);
    }

    utf8_byte_split_free(pair_syms, pair_len);
    return new_vocab;
}

/** @} */

/**
 * Command Line Interface
 * @{
 */

void tokenizer_usage(const char* argv) {
    fprintf(stderr, "Usage: %s [input.txt] [num merges]>\n", argv);
}

/** @} */

int main(int argc, char* argv[]) {
    if (argc < 2) {
        tokenizer_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Read the input corpus from a plaintext file
    char* corpus_path = argv[1];
    ssize_t corpus_size = 0;
    char* corpus = tokenizer_corpus_mmap(corpus_path, &corpus_size);
    printf("[corpus start]\n%s\n[corpus end]\n", corpus);

    // @todo Prepare the write the tokenizer model to a binary format
    //       matching it's outlined structure.
    // char* model_path = argv[2];

    // Pre-tokenize the input corpus
    uint64_t split_count = 0;
    char** corpus_split = (char**) utf8_byte_split_regex((const uint8_t*) corpus, (const uint8_t*) VTKN_PRE, &split_count);

    printf("split_count=%lu\n", split_count);
    for (uint64_t i = 0; i < split_count; i++) {
        printf("corpus_split[%lu]='%s'\n", i, corpus_split[i]);
    }

    // Build the vocab
    HashMap* vocab = tokenizer_vocab_create(corpus_split, split_count);
    HashMapEntry* entry = NULL;
    HashMapIterator it = hash_map_iter(vocab);

    printf("vocab=%p, split_count=%lu\n", (void*) vocab, split_count);
    while ((entry = hash_map_next(&it))) {
        printf(
            "entry[%lu], entry->key='%s', entry->value=%d\n",
            it.index,
            (char*) entry->key,
            *(int*) entry->value
        );
    }

    uint64_t num_merges = 2;
    if (argc == 3) {
        num_merges = atoi(argv[2]);
    }
    for (uint64_t i = 0; i < num_merges; i++) {
        // 1. Get all pairs and their frequencies
        HashMap* pairs = tokenizer_pairs_create(vocab);

        // 2. Find the best (most frequent) pair
        int best_freq = 0;
        char* best_pair = NULL;
        it = hash_map_iter(pairs);
        while ((entry = hash_map_next(&it))) {
            int freq = *(int*) entry->value;
            if (freq > best_freq) {
                best_freq = freq;
                if (best_pair) {
                    memory_free(best_pair);
                }
                best_pair = (char*) utf8_byte_copy((uint8_t*) entry->key);
            }
        }

        if (!best_pair) {
            memory_free(best_pair);
            tokenizer_pairs_free(pairs);
            break;  // No more pairs to merge
        }

        printf("Merging pair: '%s' freq: %d\n", best_pair, best_freq);

        // 3. Merge the vocab on this pair
        HashMap* new_vocab = tokenizer_merges_create(vocab, best_pair);

        // 4. Cleanup
        memory_free(best_pair);
        tokenizer_pairs_free(pairs);
        tokenizer_vocab_free(vocab);
        vocab = new_vocab;  // Use merged vocab for next round
    }

    // Clean up
    tokenizer_vocab_free(vocab);
    utf8_byte_split_free((uint8_t**) corpus_split, split_count);
    tokenizer_corpus_unmap(corpus, corpus_size);
    return EXIT_SUCCESS;
}
