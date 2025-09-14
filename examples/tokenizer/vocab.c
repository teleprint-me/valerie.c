/**
 * @file examples/tokenizer/vocab.c
 * @brief Test driver for handling transformer bpe vocab.
 */

#include "logger.h"
#include "strext.h"
#include "path.h"
#include <stdio.h>

struct CLIParams {
    const char** argv;
    char* vocab_path;
    int argc;
};

void cli_usage(struct CLIParams cli) {
    printf("Usage: %s %s\n", cli.argv[0], "[--vocab S] ...");
    printf("--vocab S Plain text input file (default: data/vocab.txt)\n");
}

void cli_parse(struct CLIParams cli) {
    for (int i = 1; i < cli.argc; i++) {
        if (strcmp(cli.argv[i], "--vocab") == 0 && i + 1 < cli.argc) {
            cli.vocab_path = (char*) cli.argv[++i];
        } else if (strcmp(cli.argv[i], "--help") == 0 || strcmp(cli.argv[i], "-h") == 0) {
            cli_usage(cli);
            exit(EXIT_SUCCESS);
        } else {
            LOG_ERROR("Unknown or incomplete option: %s", cli.argv[i]);
            cli_usage(cli);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, const char* argv[]) {
    struct CLIParams cli = {.argc = argc, .argv = argv, .vocab_path = NULL};
    if (argc > 1) {
        cli_parse(cli);
    } else {
        cli_usage(cli);
        exit(EXIT_FAILURE);
    }

    if (!cli.vocab_path) {
        LOG_ERROR("NULL vocab path detected.");
        exit(EXIT_FAILURE);
    }

    return 0;
}
