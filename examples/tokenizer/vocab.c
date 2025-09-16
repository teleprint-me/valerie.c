/**
 * Copyright © 2025 Austin Berrio
 * @file examples/tokenizer/vocab.c
 * @brief Test driver for handling transformer bpe vocab.
 */

#include <string.h>
#include <stdio.h>

#include "strext.h"
#include "path.h"
#include "logger.h"
#include "map.h"

/**
 * Vocab Map Utils
 * @note These can probably be integrated into HashMap
 * @{
 */

// Free the vocabulary mapping
void vocab_map_free(HashMap* m) {
    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(m);
    while ((entry = hash_map_next(&it))) {
        free(entry->key);
        free(entry->value);
    }
    hash_map_free(m);
}

// Flush vocab to standard output
void vocab_map_print(HashMap* m) {
    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(m);
    while ((entry = hash_map_next(&it))) {
        char* tok = entry->key;
        int* freq = entry->value;
        printf("tok=`%s` | freq=`%d`\n", tok, *freq);
    }
}

/** @} */

/**
 * Vocab serialization
 * Format:
 *   [int32] magic ('vox\0')
 *   [int32] version (currently 1)
 *   [int32] count (number of entries)
 *   [int32] size  (hash map capacity used for init)
 *   For each entry:
 *     [int32] token length (bytes)
 *     [char[]] token bytes
 *     [int32] frequency
 * All values little-endian, native encoding.
 * @{
 */

// Save the given vocab HashMap to a binary file
bool vocab_map_save(HashMap* m, const char* path) {
    // Get the current directory
    char* dirname = path_dirname(path);
    // Create the directory if it does not exist
    path_mkdir(dirname);  // returns 0 on success
    // Clean up
    free(dirname);

    // Open the vocab file for writing
    FILE* file = fopen(path, "wb");
    if (!file) {
        return false;
    }

    // Place holders for future macros
    // Using int keeps things simple for now
    int magic = 0x766F7800;  // vox magic
    fwrite(&magic, 1, sizeof(int), file);

    int version = 1;  // vox version
    fwrite(&version, 1, sizeof(int), file);

    // number of elements in the map (for reading)
    int count = m->count;  // vox has n elements
    fwrite(&count, 1, sizeof(int), file);

    // number of bytes allocated (for initialization)
    int size = m->size;  // vox has n bytes
    fwrite(&size, 1, sizeof(int), file);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(m);
    while ((entry = hash_map_next(&it))) {
        // Get the current token
        char* tok = entry->key;
        // Get the current token length
        int tok_len = strlen(tok);
        // Get the current frequency
        int* freq = entry->value;

        // Write kv mapping to disk
        fwrite(&tok_len, 1, sizeof(int), file);
        fwrite(tok, tok_len, sizeof(char), file);
        fwrite(freq, 1, sizeof(int), file);
    }

    // Clean up
    fclose(file);
    return true;  // ok
}

// Load the given binary file into a vocab HashMap
HashMap* vocab_map_load(const char* path) {
    // Check if path is a valid file
    if (!path_is_file(path)) {
        return NULL;  // file doesn't exist
    }

    // Attempt to open the alleged vocab file
    FILE* file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }

    // Read and validate vocab magic
    int magic = 0;
    fread(&magic, 1, sizeof(int), file);
    if (magic != 0x766F7800) {
        fclose(file);
        return NULL;
    }

    // Read and validate vocab version
    int version = 0;
    fread(&version, 1, sizeof(int), file);
    if (version != 1) {
        fclose(file);
        return NULL;
    }

    // Get the number of kv pairs in the vocab
    int count = 0;
    fread(&count, 1, sizeof(int), file);

    // Get the number of bytes required for the map
    int size = 0;
    fread(&size, 1, sizeof(int), file);

    // Allocate the map
    HashMap* m = hash_map_create(size, HASH_MAP_KEY_TYPE_STRING);
    if (!m) {
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        // Get the key length
        int k_len = 0;
        fread(&k_len, 1, sizeof(int), file);

        // Get the key
        char* k = calloc(k_len + 1, sizeof(char));
        fread(k, k_len, sizeof(char), file);
        k[k_len] = '\0';

        // Get the value
        int* v = calloc(1, sizeof(int));
        fread(v, 1, sizeof(int), file);

        // m : k -> v
        hash_map_insert(m, k, v);
    }

    fclose(file);
    return m;  // v : tok -> freq
}

/** @} */

/**
 * Read text into memory for vocab mapping
 * @note This should probably be generalized, but this is fine for now.
 * @{
 */

// Read a plain text file into memory from disk
char* vocab_read_text(const char* path) {
    // Ensure path is not null
    if (!path_is_valid(path)) {
        return NULL;
    }

    // Ensure path exists
    if (!path_exists(path)) {
        return NULL;
    }

    // Open the text file
    FILE* file = fopen(path, "r");
    if (!file) {
        return NULL;
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    if (length == 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);

    // Allocate memory to string
    char* text = calloc(length + 1, sizeof(char));
    if (!text) {
        fclose(file);
        return NULL;
    }

    // Read data into memory from disk
    fread(text, sizeof(char), length, file);
    fclose(file);
    if (!*text) {
        free(text);
        return NULL;
    }
    text[length] = '\0';  // null terminate

    return text;
}

/** @} */

/**
 * Map vocab frequencies
 * @{
 */

// Create the word frequencies
HashMap* vocab_create_frequencies(const char* text) {
    // Pre-tokenize the input text
    size_t pre_token_count = 0;
    // Split text by whitespace (0x09, 0x0A, 0x0D, 0x20)
    char** pre_tokens = string_split_space(text, &pre_token_count);

    // Build word frequencies from pre-tokens
    HashMap* freqs = hash_map_create(pre_token_count, HASH_MAP_KEY_TYPE_STRING);
    for (size_t i = 0; i < pre_token_count; i++) {
        int* value = hash_map_search(freqs, pre_tokens[i]);
        if (!value) {
            // Create a new key
            char* key = strdup(pre_tokens[i]);

            // Create a new value
            value = malloc(sizeof(int));
            *value = 1;

            // Insert new mapping
            hash_map_insert(freqs, key, value);
        } else {
            *value += 1;  // update current freq
        }
    }

    // Clean up pre-tokens
    string_split_free(pre_tokens, pre_token_count);

    // Return hash map
    return freqs;  // text : words -> freqs
}

// Create the symbol frequencies
HashMap* vocab_create_symbols(HashMap* words) {
    // Create the symbol-freq mapping
    HashMap* vocab = hash_map_create(words->size, HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(words);
    while ((entry = hash_map_next(&it))) {
        // get current word-freq mapping
        char* word = entry->key;  // tok -> cat
        int* freq = entry->value;  // freq -> 1

        // "cat" -> {"c", "a", "t"}
        size_t list_count = 0;
        char** list = string_split(word, &list_count);

        // {"c", "a", "t"} -> "c a t"
        char* symbols = string_join(list, list_count, " ");  // new word

        // clean up intermediate representation
        string_split_free(list, list_count);

        // handle word to symbol freq mapping
        int* sym_freq = hash_map_search(vocab, symbols);
        if (!sym_freq) {  // new sym-freq
            int* new_freq = malloc(sizeof(int));
            *new_freq = *freq;
            hash_map_insert(vocab, symbols, new_freq);
        } else {
            *sym_freq += 1;  // inc sym-freq
        }
    }

    // return hash map
    return vocab;  // words : syms -> freqs
}

/** @} */

/**
 * Vocab factory
 * @{
 */

// Pre-tokenize the vocabulary
HashMap* vocab_tokenize(const char* text) {
    // Create initial word-freq mapping
    HashMap* words = vocab_create_frequencies(text);
    if (!words) {
        return NULL;
    }

    // Create initial sym-freq mapping
    HashMap* vocab = vocab_create_symbols(words);
    if (!vocab) {
        vocab_map_free(words);
        return NULL;
    }

    // Clean up word-freq map
    vocab_map_free(words);

    // Return a newly initialized vocab map
    return vocab;  // v : syms -> freqs
}

// Build the initial vocabulary
HashMap* vocab_build(const char* path) {
    // Read the vocab from a plain text file into memory
    char* text = vocab_read_text(path);
    if (!text) {
        return NULL;
    }

    // Create the initial sym-freq map
    HashMap* vocab = vocab_tokenize(text);
    if (!vocab) {
        free(text);
        return NULL;
    }

    // Clean up plain text
    free(text);

    // Return a newly
    return vocab;
}

/** @} */

/**
 * Command-line interface
 * @{
 */

struct CLIParams {
    const char** argv;
    char* vocab_path;
    int argc;
};

void cli_usage(struct CLIParams cli) {
    printf("Usage: %s %s\n", cli.argv[0], "[--vocab S] ...");
    printf("--vocab S Plain text input file (default: samples/simple.txt)\n");
}

void cli_parse(struct CLIParams* cli) {
    if (cli->argc < 2) {
        cli_usage(*cli);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < cli->argc; i++) {
        if (strcmp(cli->argv[i], "--vocab") == 0 && i + 1 < cli->argc) {
            cli->vocab_path = strdup(cli->argv[++i]);
        } else if (strcmp(cli->argv[i], "--help") == 0 || strcmp(cli->argv[i], "-h") == 0) {
            cli_usage(*cli);
            exit(EXIT_SUCCESS);
        } else {
            printf("Unknown or incomplete option: %s", cli->argv[i]);
            cli_usage(*cli);
            exit(EXIT_FAILURE);
        }
    }
}

/** @} */

int main(int argc, const char* argv[]) {
    // Parse CLI arguments
    struct CLIParams cli = {.argc = argc, .argv = argv, .vocab_path = NULL};
    cli_parse(&cli);

    // Ensure vocab path is not null
    if (!cli.vocab_path) {
        cli.vocab_path = strdup("samples/simple.txt");
    }

    // Ensure vocab path exists
    if (!path_exists(cli.vocab_path)) {
        LOG_ERROR("Invalid vocab path detected: '%s'", cli.vocab_path);
        free(cli.vocab_path);
        exit(EXIT_FAILURE);
    }

    // Build word frequencies from text file
    HashMap* vocab = vocab_build(cli.vocab_path);

    // Observe mapped results
    vocab_map_print(vocab);

    // Clean up
    vocab_map_free(vocab);
    free(cli.vocab_path);
    return 0;
}
