/**
 * @file      tokenizer/model.h
 * @brief     BPE tokenizer model interface for Valerie
 * @copyright Copyright © 2025 Austin Berrio
 *
 * @ref https://arxiv.org/abs/1508.07909v5
 * @ref https://arxiv.org/abs/2505.24689
 * @ref https://aclanthology.org/2025.coling-main.400/
 */

#ifndef TOKENIZER_MODEL_H
#define TOKENIZER_MODEL_H

#include <stddef.h>
#include <stdbool.h>

#include "tokenizer/bpe.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @name Version Macros
 *  @{
 */
#define TOKENIZER_MAGIC 0x766F7870 /**< Model file magic value. */
#define TOKENIZER_VERSION 1 /**< Model format version.   */

/** @} */

/**
 * @brief Special tokens struct for start/end/pad/unk markers.
 */
typedef struct SpecialToken {
    char* bos; /**< Beginning-of-sequence token (e.g. "<|bos|>"). */
    char* eos; /**< End-of-sequence token (e.g. "<|eos|>"). */
    char* pad; /**< Padding token (e.g. "<|pad|>"). */
    char* unk; /**< Unknown token (e.g. "<|unk|>"). */
} SpecialToken;

/**
 * @brief Tokenizer structure for Byte Pair Encoding models.
 */
typedef struct Tokenizer {
    SpecialToken* special; /**< Special token markers. */
    void* scores; /**< (Internal) Merge scores map. */
    void* token_to_id; /**< (Internal) Token-to-id hashmap. */
    char** id_to_token; /**< Array: index maps to token string. */
    int vocab_size; /**< Number of tokens in vocabulary. */
} Tokenizer;

/** @name Special Token API
 *  @{
 */

/**
 * @brief Create a SpecialToken struct with custom or default values.
 *
 * @param bos  BOS token string (or NULL for default).
 * @param eos  EOS token string (or NULL for default).
 * @param pad  PAD token string (or NULL for default).
 * @param unk  UNK token string (or NULL for default).
 * @return     Allocated SpecialToken struct.
 */
SpecialToken* token_special_create(char* bos, char* eos, char* pad, char* unk);

/**
 * @brief Free a SpecialToken struct and its strings.
 *
 * @param special SpecialToken struct to free.
 */
void token_special_free(SpecialToken* special);

/** @} */

/** @name Tokenizer API
 *  @{
 */

/**
 * @brief Create a Tokenizer from a BPE model and special tokens.
 *
 * @param model   Pointer to a loaded BPEModel.
 * @param special SpecialToken struct (ownership is transferred).
 * @return        Allocated Tokenizer struct, or NULL on failure.
 */
Tokenizer tokenizer_create(BPEModel* model, SpecialToken* special);

/**
 * @brief Free a Tokenizer and all associated memory.
 *
 * @param t Tokenizer struct to free.
 */
void tokenizer_free(Tokenizer* t);

/** @} */

/** @name Model Persistence
 *  @{
 */

/**
 * @brief Serialize a tokenizer to a file.
 *
 * @param t    Tokenizer struct to save.
 * @param path Path to output file.
 * @return     true on success, false on failure.
 */
bool tokenizer_save(Tokenizer* t, const char* path);

/**
 * @brief Load a Tokenizer from a file.
 *
 * @param path Path to saved model file.
 * @return     Allocated Tokenizer struct, or NULL on error.
 */
Tokenizer tokenizer_load(const char* path);

/** @} */

/** @name Encoding and Decoding
 *  @{
 */

/**
 * @brief Encode a UTF-8 string into an array of token ids.
 *
 * @param t       Tokenizer to use.
 * @param text    Input string (UTF-8, null-terminated).
 * @param seq_len Output number of token ids in result.
 * @param add_bos If true, add BOS token at start if defined.
 * @param add_eos If true, add EOS token at end if defined.
 * @return        Newly allocated array of token ids (caller frees), or NULL.
 */
int* tokenizer_encode(Tokenizer* t, char* text, int* seq_len, bool add_bos, bool add_eos);

/**
 * @brief Decode a sequence of token ids into a UTF-8 string.
 *
 * @param t         Tokenizer to use.
 * @param ids       Array of token ids.
 * @param seq_len   Input number of token ids in array.
 * @return          Newly allocated decoded string (caller frees), or NULL.
 */
char* tokenizer_decode(Tokenizer* t, int* ids, size_t seq_len);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // TOKENIZER_MODEL_H
