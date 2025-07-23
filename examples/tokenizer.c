/**
 * @file examples/tokenizer.c
 * @brief Valerie uses a custom BPE tokenizer built from scratch in pure C.
 * @todo Need methods for creating tokenizers from a raw UTF-8 corpus.
 */

#include "logger.h"
#include "map.h"

#include <stdio.h>
#include <sys/types.h>  // ssize_t

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
    int vox_len;  // Max UTF-8 length of any token
    size_t vox_size;  // vocab size
    TokenSpecial special;
    HashMap* token_to_id;
    HashMap* id_to_token;
} Tokenizer;

// @brief Read a corpus from plaintext for training.
char* tokenizer_corpus_read(const char* filepath);

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

int main(void) {
    printf("Hello, world!");
    return 0;
}
