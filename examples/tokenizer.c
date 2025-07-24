/**
 * Copyright © 2023 Austin Berrio
 *
 * @file examples/tokenizer.c
 * @brief Valerie uses a custom BPE tokenizer built from scratch in pure C.
 * @todo Need methods for creating tokenizers from a raw UTF-8 corpus.
 * @ref arXiv:1508.07909v5 [cs.CL] 10 Jun 2016
 */

#include "memory.h"
#include "logger.h" // logging macros LOG_ERROR, etc.
#include "map.h" // HashMap* map
#include "utf8/raw.h" // Process UTF-8 strings as char*
#include <stdint.h>
#include <stdio.h> // IO
#include <sys/types.h>  // ssize_t
#include <sys/mman.h> // mmap/munmap

/**
 * Tokenizer Blueprint
 * @{
 */

#define VTKN_MAGIC 0x56544B4E
#define VTKN_VERSION 1
#define VTKN_META "\u2581" // UTF-8 marker '▁'
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
HashMap* tokenizer_vocab_merge(HashMap* vocab, char* pair);

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
        // Get the tokens codepoints
        char* token = tokens[i];
        uint64_t cpts_count = 0;
        char** cpts = utf8_raw_split_char(token, &cpts_count);

        // Compute the buffer size
        size_t buf_size = 0;
        for (uint64_t j = 0; j < cpts_count; j++) {
            buf_size += utf8_raw_byte_count(cpts[j]) + 1; // add space
        }
        buf_size += utf8_raw_byte_count("</w>") + 2; // space + null

        // Join codepoints with spaces and append stop token
        char* key = memory_alloc(buf_size, alignof(char));
        key[0] = '\0';

        for (uint64_t j = 0; j < cpts_count; j++) {
            char* prev = key;
            key = utf8_raw_concat(key, cpts[j]);
            memory_free(prev);

            prev = key;
            key = utf8_raw_concat(key, " ");
            memory_free(prev);
        }

        char* prev = key;
        key = utf8_raw_concat(key, "</w>");
        memory_free(prev);

        // Insert key-value pairs into map
        int* frequency = hash_map_search(vocab, key);
        if(!frequency) {
            int* value = memory_alloc(sizeof(int), alignof(int));
            *value = 1;
            hash_map_insert(vocab, key, value);
        } else {
            (*frequency)++;
            memory_free(key); // key already exists
        }

        utf8_raw_split_free(cpts, cpts_count);
    }
    return vocab;
}

void tokenizer_vocab_free(HashMap* vocab) {
    if (vocab) {
        HashMapIterator it = hash_map_iter(vocab);
        HashMapEntry* entry;
        while ((entry = hash_map_next(&it))) {
            memory_free(entry->key);
            memory_free(entry->value);
        }
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
        char** symbols = utf8_raw_split(word, " ", &symbol_count);
        if (symbol_count < 2) {
            utf8_raw_split_free(symbols, symbol_count);
            continue;
        }

        for (uint64_t i = 0; i < symbol_count - 1; i++) {
            // Join symbols[i] + " " + symbols[i+1] as new string
            char* pair = utf8_raw_split_join((char*[]){symbols[i], symbols[i + 1]}, " ", 2);
            int* pair_freq = hash_map_search(pairs, pair);
            if (!pair_freq) {
                int* value = memory_alloc(sizeof(int), alignof(int));
                *value = frequency;
                hash_map_insert(pairs, pair, value);
            } else {
                *pair_freq += frequency;
                memory_free(pair); // Already exists
            }
        }

        utf8_raw_split_free(symbols, symbol_count);
    }

    return pairs;
}

void tokenizer_pairs_free(HashMap* pairs) {
    if (!pairs) return;
    HashMapIterator it = hash_map_iter(pairs);
    HashMapEntry* ent;
    while ((ent = hash_map_next(&it))) {
        memory_free(ent->key);
        memory_free(ent->value);
    }
    hash_map_free(pairs);
}

/** @} */

/**
 * Command Line Interface
 * @{
 */

void tokenizer_usage(const char* argv) {
    fprintf(stderr, "Usage: %s path/to/corpus.md path/to/tokenizer.model\n", argv);
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
    char** corpus_split = utf8_raw_split_regex(corpus, VTKN_PRE, &split_count);
    printf("Pre-tokenized corpus into %lu tokens.\n", split_count);

    // Build the vocab
    HashMap* vocab = tokenizer_vocab_create(corpus_split, split_count);
    // Pair frequent scores
    HashMap* pairs = tokenizer_pairs_create(vocab);

    HashMapIterator it = hash_map_iter(pairs);
    HashMapEntry* ent;
    while ((ent = hash_map_next(&it))) {
        printf("'%s': %d\n", (char*)ent->key, *(int*)ent->value);
    }

    // Clean up
    tokenizer_pairs_free(pairs);
    tokenizer_vocab_free(vocab);
    utf8_raw_split_free(corpus_split, split_count);
    tokenizer_corpus_unmap(corpus, corpus_size);
    return EXIT_SUCCESS;
}
