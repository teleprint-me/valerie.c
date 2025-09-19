/**
 * Copyright © 2023 Austin Berrio
 *
 * @file examples/tokenizer/model.c
 * @brief Valerie uses a custom BPE tokenizer built from scratch in pure C.
 * @todo Need methods for creating tokenizers from a raw UTF-8 corpus.
 * @ref arXiv:1508.07909v5 [cs.CL] 10 Jun 2016
 * @ref arXiv:2505.24689 [cs.CL] 30 May 2025
 * @ref https://aclanthology.org/2025.coling-main.400/
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/strext.h"
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

    char* dir;  // output model file
    char* text;  // input text corpus
    int merges;  // number of merges
    bool debug;  // enable verbosity
};

void cli_usage(struct CLIParams* cli) {
    printf("Usage: %s [--vocab S] ...\n", cli->argv[0]);
    printf("--corpus  S Plain text input file (default: samples/simple.txt)\n");
    printf("--merges  N Number of merges (default: 10)\n");
    printf("--verbose B Enables debug log (default: false)\n");
}

void cli_free(struct CLIParams* cli) {
    if (cli->text) {
        free(cli->text);
    }
}

void cli_parse(struct CLIParams* cli) {
    if (cli->argc < 2) {
        cli_usage(cli);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < cli->argc; i++) {
        if (strcmp(cli->argv[i], "--vocab") == 0 && i + 1 < cli->argc) {
            cli->text = strdup(cli->argv[++i]);
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
        .text = NULL,
        .merges = 1,
        .debug = false,
    };
    cli_parse(&cli);

    // Ensure vocab path is not null
    if (!cli.text) {
        cli.text = strdup("samples/simple.txt");
    }

    // Build word frequencies from text file
    HashMap* vocab = vocab_build(cli.text);
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
    free(cli.text);
    return EXIT_SUCCESS;
}
