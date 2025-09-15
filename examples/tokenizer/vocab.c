/**
 * @file examples/tokenizer/vocab.c
 * @brief Test driver for handling transformer bpe vocab.
 */

#include <string.h>
#include <stdio.h>

#include "strext.h"
#include "path.h"
#include "logger.h"
#include "map.h"

// Read the models vocabulary from disk
char* vocab_read(const char* path) {
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
    char* vocab = calloc(length + 1, sizeof(char));
    if (!vocab) {
        fclose(file);
        return NULL;
    }

    // Read data into memory from disk
    fread(vocab, sizeof(char), length, file);
    fclose(file);
    if (!*vocab) {
        return NULL;
    }
    vocab[length] = '\0';  // null terminate

    return vocab;
}

// Create the initial vocab frequencies
HashMap* vocab_freqs_create(const char* vocab) {
    // Pre-tokenize the vocab
    size_t pre_token_count = 0;
    char** pre_tokens = string_split_space(vocab, &pre_token_count);

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
    return freqs;  // f(v) : tokens -> freqs
}

HashMap* vocab_symbols_create(HashMap* freqs) {
    // Create the symbol-freq mapping
    HashMap* symbols = hash_map_create(freqs->size, HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(freqs);
    while ((entry = hash_map_next(&it))) {
        // get current word-freq mapping
        char* word = entry->key;  // tok -> cat
        int* freq = entry->value;  // freq -> 1

        // "cat" -> {"c", "a", "t"}
        size_t word_count = 0;
        char** word_split = string_split(word, &word_count);

        // {"c", "a", "t"} -> "c a t"
        char* syms = string_join(word_split, word_count, " ");  // new word

        // clean up intermediate representation
        string_split_free(word_split, word_count);

        // handle word to symbol freq mapping
        int* value = hash_map_search(symbols, syms);
        if (!value) {  // new freq
            int* new_freq = malloc(sizeof(int));
            *new_freq = *freq;
            hash_map_insert(symbols, syms, new_freq);
        } else {
            *value += 1;  // inc freq
        }
    }

    // return hash map
    return symbols;  // f : syms -> freqs
}

void vocab_map_free(HashMap* map) {
    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(map);
    while ((entry = hash_map_next(&it))) {
        free(entry->key);
        free(entry->value);
    }
    hash_map_free(map);
}

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

    // Read the vocab from disk into memory
    char* vocab = vocab_read(cli.vocab_path);
    if (!vocab) {
        LOG_ERROR("Failed to read vocab data: '%s'", cli.vocab_path);
        free(cli.vocab_path);
        exit(EXIT_FAILURE);
    }

    // Build word frequencies from pre-tokens
    HashMap* freqs = vocab_freqs_create(vocab);
    HashMap* symbols = vocab_symbols_create(freqs);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(symbols);
    while ((entry = hash_map_next(&it))) {
        char* word = entry->key;
        int* freq = entry->value;
        printf("syms = %s, freq = %d\n", word, *freq);
    }

    // Clean up
    vocab_map_free(symbols);
    vocab_map_free(freqs);
    free(vocab);
    free(cli.vocab_path);
    return 0;
}
