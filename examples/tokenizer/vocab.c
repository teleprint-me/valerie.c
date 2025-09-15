/**
 * @file examples/tokenizer/vocab.c
 * @brief Test driver for handling transformer bpe vocab.
 */

#include <string.h>
#include <stdio.h>

#include "strext.h"
#include "path.h"
#include "logger.h"

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

    if (!path_exists(cli.vocab_path)) {
        LOG_ERROR("Invalid vocab path detected: '%s'", cli.vocab_path);
        free(cli.vocab_path);
        exit(EXIT_FAILURE);
    }

    char* vocab = vocab_read(cli.vocab_path);
    if (!vocab) {
        LOG_ERROR("Failed to read vocab data: '%s'", cli.vocab_path);
        free(cli.vocab_path);
        exit(EXIT_FAILURE);
    }

    printf("Vocab contents:\n%s\n", vocab);

    free(vocab);
    free(cli.vocab_path);
    return 0;
}
