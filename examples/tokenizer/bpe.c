/**
 * Copyright © 2025 Austin Berrio
 * @file examples/tokenizer/bpe.c
 * @brief Test driver for handling transformer bpe merges.
 */

#include <stdio.h>

#include "core/path.h"
#include "core/strext.h"
#include "core/map.h"

#include "tokenizer/vocab.h"

// collect vocab pairs
// once all pairs have been exhausted,
// the pairs function must return NULL to indicate the end of operation
HashMap* bpe_pairs(HashMap* vocab) {
    HashMap* new_pairs = hash_map_create(hash_map_size(vocab), HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        size_t sym_count = 0;
        char** syms = string_split_delim(entry->key, " ", &sym_count);

        for (size_t i = 0; i < sym_count - 1; i++) {
            // Create the symbol tuple
            char* tuple[] = {syms[i], syms[i + 1]};

            // Create the symbol pair
            char* pair = string_join(tuple, 2, " ");

            // Check for pair existence
            int* freq = hash_map_search(new_pairs, pair);
            if (!freq) {  // pair does not exist
                int* new_freq = malloc(sizeof(int));
                *new_freq = *(int*) entry->value;
                hash_map_insert(new_pairs, pair, new_freq);
            } else {  // pair already exists
                *freq += *(int*) entry->value;
                free(pair);
            }
        }

        string_split_free(syms, sym_count);
    }

    return new_pairs;
}

char* bpe_best(HashMap* pairs, int* out_freq) {
    char* best_pair = NULL;
    int best_freq = -1;

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(pairs);
    while ((entry = hash_map_next(&it))) {
        char* pair = entry->key;
        int* freq = entry->value;

        if (*freq > best_freq) {
            best_pair = pair;
            best_freq = *freq;
        } else if (*freq == best_freq && best_pair && strcmp(pair, best_pair) < 0) {
            best_pair = pair;  // lexicographic tie-break
        }
    }

    if (out_freq) {
        *out_freq = best_freq;
    }

    // Caller must validate and free best pair
    return best_pair ? strdup(best_pair) : NULL;
}

HashMap* bpe_merges(HashMap* vocab, const char* best_pair) {
    if (!vocab || !best_pair) {
        return NULL;  // exhausted all pairs
    }

    // Parse tuple: (a, b)
    size_t tuple_count = 0;
    char** tuple = string_split_delim(best_pair, " ", &tuple_count);
    if (tuple_count != 2) {
        string_split_free(tuple, tuple_count);
        return NULL;
    }
    const char* a = tuple[0];
    const char* b = tuple[1];

    // New vocab map
    HashMap* new_vocab = hash_map_create(hash_map_size(vocab), HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        // Split word into symbols
        size_t sym_count = 0;
        char** syms = string_split_delim(entry->key, " ", &sym_count);

        // Prepare output buffer (never longer than input)
        size_t out_count = 0;
        char** out = calloc(sym_count, sizeof(char*));

        size_t i = 0;
        while (i < sym_count) {
            if (i + 1 < sym_count && strcmp(syms[i], a) == 0 && strcmp(syms[i + 1], b) == 0) {
                // Merge: concat a + b
                size_t merge_count = strlen(a) + strlen(b) + 1;
                char* merge = malloc(merge_count);
                strcpy(merge, a);
                strcat(merge, b);

                out[out_count++] = merge;
                i += 2;  // skip b
            } else {
                out[out_count++] = strdup(syms[i]);
                i += 1;
            }
        }

        // Join tokens into new word (space-delimited)
        char* new_word = string_join(out, out_count, " ");

        // Insert new word into new vocab
        int* freq = hash_map_search(new_vocab, new_word);
        if (!freq) {
            int* new_freq = malloc(sizeof(int));
            *new_freq = *(int*) entry->value;
            hash_map_insert(new_vocab, new_word, new_freq);
        } else {
            *freq += *(int*) entry->value;
            free(new_word);
        }

        // Free intermediate arrays
        string_split_free(out, out_count);
        string_split_free(syms, sym_count);
    }

    string_split_free(tuple, tuple_count);
    return new_vocab;
}

/**
 * Command-line interface
 * @{
 */

struct CLIParams {
    const char** argv;
    int argc;

    char* vocab_path;
    int merges;
    bool debug;
};

void cli_usage(struct CLIParams* cli) {
    printf("Usage: %s [--vocab S] ...\n", cli->argv[0]);
    printf("--vocab   S Plain text input file (default: samples/simple.txt)\n");
    printf("--merges  N Number of merges (default: 10)\n");
    printf("--verbose B Enables debug log (default: false)\n");
}

void cli_free(struct CLIParams* cli) {
    if (cli->vocab_path) {
        free(cli->vocab_path);
    }
}

void cli_parse(struct CLIParams* cli) {
    if (cli->argc < 2) {
        cli_usage(cli);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < cli->argc; i++) {
        if (strcmp(cli->argv[i], "--vocab") == 0 && i + 1 < cli->argc) {
            cli->vocab_path = strdup(cli->argv[++i]);
        } else if (strcmp(cli->argv[i], "--merges") == 0 && i + 1 < cli->argc) {
            cli->merges = atoi(cli->argv[++i]);
            if (1 > cli->merges) {
                cli->merges = 10;
            }
        } else if (strcmp(cli->argv[i], "--verbose") == 0) {
            cli->debug = true;
        } else if (strcmp(cli->argv[i], "--help") == 0 || strcmp(cli->argv[i], "-h") == 0) {
            cli_usage(cli);
            cli_free(cli);
            exit(EXIT_SUCCESS);
        } else {
            printf("Unknown or incomplete option: %s", cli->argv[i]);
            cli_usage(cli);
            cli_free(cli);
            exit(EXIT_FAILURE);
        }
    }
}

/** @} */

int main(int argc, const char* argv[]) {
    // Parse CLI arguments
    struct CLIParams cli = {
        .argc = argc,
        .argv = argv,
        .vocab_path = NULL,
        .merges = 1,
        .debug = false,
    };
    cli_parse(&cli);

    // Ensure vocab path is not null
    if (!cli.vocab_path) {
        cli.vocab_path = strdup("samples/simple.txt");
    }

    // Build word frequencies from text file
    HashMap* vocab = vocab_build(cli.vocab_path);
    if (cli.debug) {
        vocab_map_print(vocab);
    }

    // Collect the best merge pairs (used to build the model)
    size_t best_count = 0;
    char** best = calloc(1, sizeof(char*));
    for (int i = 0; i < cli.merges; i++) {
        // Build symbol pairs from vocab
        HashMap* pairs = bpe_pairs(vocab);
        if (cli.debug) {
            vocab_map_print(pairs);  // debug
        }

        // prep for best merges
        // collect the array of tuples
        // pairs : {syms[i], syms[i + 1]} -> freq
        int best_freq;
        char* best_pair = bpe_best(pairs, &best_freq);
        if (!best_pair) {
            printf("Exhausted all possible merge pairs at step %d.\n", i);
            vocab_map_free(pairs);
            break;  // exhausted all available pairs
        }

        // Observe best results
        printf("best_pair=%s | best_freq=%d\n", best_pair, best_freq);

        // Append the best pairs
        best = string_append(strdup(best_pair), best, &best_count);

        // Merge symbol pairs based on best freq
        HashMap* merges = bpe_merges(vocab, best_pair);
        if (cli.debug) {
            vocab_map_print(merges);  // debug
        }

        // Clean up
        free(best_pair);
        vocab_map_free(pairs);
        vocab_map_free(vocab);

        // Update
        vocab = merges;
    }

    // Clean up
    string_split_free(best, best_count);
    vocab_map_free(vocab);
    free(cli.vocab_path);
    return EXIT_SUCCESS;
}
