/**
 * @file      tokenizer/bpe.c
 * @brief     Byte-Pair Encoding (BPE) merges and model API.
 * @copyright Copyright © 2025 Austin Berrio
 *
 * This module implements the core BPE model and merge operations,
 * including training, serialization, and manipulation of BPE merges.
 *
 * @note All APIs are designed for use with space-delimited symbol strings
 *       (e.g., "l o o k e d"). Ownership of returned pointers is specified.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/map.h"

#include "tokenizer/vocab.h"
#include "tokenizer/bpe.h"

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

    BPEModel* model = bpe_train(vocab, cli.merges, cli.debug);
    if (!model) {
        vocab_map_free(vocab);
        return EXIT_FAILURE;
    }

    /// @todo Do stuff here...

    // Clean up
    bpe_free(model);
    vocab_map_free(vocab);
    free(cli.vocab_path);
    return EXIT_SUCCESS;
}
