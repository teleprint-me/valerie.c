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

/**
 * Command-line interface
 * @{
 */

struct CLIParams {
    const char** argv;
    char* vocab_path;
    int argc;
};

void cli_usage(struct CLIParams* cli) {
    printf("Usage: %s [--vocab S] ...\n", cli->argv[0]);
    printf("--vocab S Plain text input file (default: samples/simple.txt)\n");
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
    struct CLIParams cli = {.argc = argc, .argv = argv, .vocab_path = NULL};
    cli_parse(&cli);

    // Ensure vocab path is not null
    if (!cli.vocab_path) {
        cli.vocab_path = strdup("samples/simple.txt");
    }

    // Build word frequencies from text file
    HashMap* vocab = vocab_build(cli.vocab_path);
    // Observe mapped results
    vocab_map_print(vocab);

    // prep for best merges
    // collect the array of tuples
    // pairs : {syms[i], syms[i + 1]} -> freq

    // collect vocab pairs
    // once all pairs have been exhausted,
    // the pairs function must return NULL to indicate the end of operation
    HashMap* new_pairs = hash_map_create(1, HASH_MAP_KEY_TYPE_STRING);

    // Clean up
    hash_map_free(new_pairs);
    vocab_map_free(vocab);
    free(cli.vocab_path);
    return 0;
}
