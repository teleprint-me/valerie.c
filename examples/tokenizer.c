/**
 * Copyright © 2023 Austin Berrio
 *
 * @file examples/tokenizer.c
 * @brief Valerie uses a custom BPE tokenizer built from scratch in pure C.
 * @todo Need methods for creating tokenizers from a raw UTF-8 corpus.
 */

#include "memory.h" // posix allocators
#include "logger.h" // logging macros LOG_ERROR, etc.
#include "map.h" // HashMap* map
#include "utf8/byte.h"
#include "utf8/raw.h"
#include <stdint.h>
#include <stdio.h> // IO
#include <stdlib.h>
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

// @brief Calculate frequencies of pairs of adjacent symbols in the vocabulary.
HashMap* tokenizer_vocab_score(HashMap* vocab);
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
    HashMap* vocab = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);
    for (uint64_t i = 0; i < split_count; i++) {
        char* token = corpus_split[i];

        uint64_t cpts_count = 0;
        char** cpts = utf8_raw_split_char(token, &cpts_count);
        for (uint64_t j = 0; j < cpts_count; j++) {
            printf("[%s]", cpts[j]);
        }

        utf8_raw_split_free(cpts, cpts_count);
    }

    // Clean up
    hash_map_free(vocab);
    utf8_raw_split_free(corpus_split, split_count);
    tokenizer_corpus_unmap(corpus, corpus_size);
    return EXIT_SUCCESS;
}
