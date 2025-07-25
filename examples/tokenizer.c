/**
 * Copyright © 2023 Austin Berrio
 *
 * @file examples/tokenizer.c
 * @brief Valerie uses a custom BPE tokenizer built from scratch in pure C.
 * @todo Need methods for creating tokenizers from a raw UTF-8 corpus.
 * @ref arXiv:1508.07909v5 [cs.CL] 10 Jun 2016
 */

#include "memory.h"
#include "logger.h"  // logging macros LOG_ERROR, etc.
#include "map.h"  // HashMap* map
#include "utf8/string.h"  // Process UTF-8 strings as char*
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

    char* data = mmap(
        NULL, *out_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(file), 0
    );
    if (!data || MAP_FAILED == data) {
        LOG_ERROR(
            "[Tokenizer] Failed to map %ld bytes of corpus to memory", *out_size
        );
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
        // Get the tokens codepoints
        char* token = tokens[i];
        uint64_t cpts_count = 0;
        char** cpts = utf8_split_char(token, &cpts_count);

        // Compute the buffer size
        size_t buf_size = 0;
        for (uint64_t j = 0; j < cpts_count; j++) {
            buf_size += utf8_len_bytes(cpts[j]) + 1;  // add space
        }
        buf_size += utf8_len_bytes("</w>") + 2;  // space + null

        // Join codepoints with spaces and append stop token
        char* key = memory_alloc(buf_size, alignof(char));
        key[0] = '\0';

        for (uint64_t j = 0; j < cpts_count; j++) {
            char* prev = key;
            key = utf8_concat(key, cpts[j]);
            memory_free(prev);

            prev = key;
            key = utf8_concat(key, " ");
            memory_free(prev);
        }

        char* prev = key;
        key = utf8_concat(key, "</w>");
        memory_free(prev);

        // Insert key-value pairs into map
        int* frequency = hash_map_search(vocab, key);
        if (!frequency) {
            int* value = memory_alloc(sizeof(int), alignof(int));
            *value = 1;
            hash_map_insert(vocab, key, value);
        } else {
            (*frequency)++;
            memory_free(key);  // key already exists
        }

        utf8_split_free(cpts, cpts_count);
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
        char* word = (char*) entry->key;
        int frequency = *(int*) entry->value;

        uint64_t symbol_count = 0;
        char** symbols = utf8_split(word, " ", &symbol_count);
        if (symbol_count < 2) {
            utf8_split_free(symbols, symbol_count);
            continue;
        }

        for (uint64_t i = 0; i < symbol_count - 1; i++) {
            // Join symbols[i] + " " + symbols[i+1] as new string
            char* pair = utf8_split_join(
                (char*[]) {symbols[i], symbols[i + 1]}, " ", 2
            );
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

        utf8_split_free(symbols, symbol_count);
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
    // Split the pair (e.g., "l l" => ["l", "l"])
    uint64_t pair_len = 0;
    char** pair_syms = utf8_split(pair, " ", &pair_len);

    HashMap* new_vocab = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        // Split word into symbols
        uint64_t sym_count = 0;
        char** syms = utf8_split(entry->key, " ", &sym_count);

        // Build new symbol array after merging
        char** new_syms = malloc(
            sizeof(char*) * (sym_count + 1)
        );  // pessimistic
        uint64_t new_count = 0;

        for (uint64_t i = 0; i < sym_count;) {
            // Try to merge at this position
            if (i + 1 < sym_count && strcmp(syms[i], pair_syms[0]) == 0
                && strcmp(syms[i + 1], pair_syms[1]) == 0) {
                // Merge the pair
                new_syms[new_count++] = utf8_concat(syms[i], syms[i + 1]);
                i += 2;
            } else {
                new_syms[new_count++] = strdup(syms[i]);
                i += 1;
            }
        }

        // Join back into word (with spaces)
        char* new_word = utf8_split_join(new_syms, " ", new_count);

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
        utf8_split_free(syms, sym_count);
    }

    utf8_split_free(pair_syms, pair_len);
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

    // @todo Prepare the write the tokenizer model to a binary format
    //       matching it's outlined structure.
    // char* model_path = argv[2];

    // Pre-tokenize the input corpus
    uint64_t split_count = 0;
    char** corpus_split = utf8_split_regex(corpus, VTKN_PRE, &split_count);

    // Build the vocab
    HashMap* vocab = tokenizer_vocab_create(corpus_split, split_count);

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
        HashMapIterator it = hash_map_iter(pairs);
        HashMapEntry* entry;
        while ((entry = hash_map_next(&it))) {
            int freq = *(int*) entry->value;
            if (freq > best_freq) {
                best_freq = freq;
                if (best_pair) {
                    memory_free(best_pair);
                }
                best_pair = utf8_copy((char*) entry->key);
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
    utf8_split_free(corpus_split, split_count);
    tokenizer_corpus_unmap(corpus, corpus_size);
    return EXIT_SUCCESS;
}
