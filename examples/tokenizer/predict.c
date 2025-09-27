/**
 * @file   examples/tokenizer/predict.c
 * @brief  Driver for loading and predicting model input ids
 * @copyright Copyright © 2025 Austin Berrio
 *
 * @ref https://arxiv.org/abs/1508.07909v5
 * @ref https://arxiv.org/abs/2505.24689
 * @ref https://aclanthology.org/2025.coling-main.400/
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

#include "core/path.h"

#include "tokenizer/model.h"

/**
 * @struct CLIParams
 * @brief Command-line parameters for tokenizer training.
 */
struct CLIParams {
    const char** argv;
    int argc;

    char* model_path;  ///< Input text corpus (plaintext)
    char* prompt;
    bool add_bos;
    bool add_eos;
    bool verbose;  ///< Enable verbose/debug output
};

/**
 * @brief Print usage instructions for the tokenizer trainer.
 */
void cli_usage(const char* prog) {
    printf("Usage: %s --model S [--verbose] [--help]\n", prog);
    printf("  --model    -m  Path to tokenizer model file (required)\n");
    printf("  --prompt   -p  Input text to encode and decode (required)\n");
    printf("  --add-bos  -b  Enable bos marker\n");
    printf("  --add-eos  -e  Enable eos marker\n");
    printf("  --verbose  -v  Enable debug/verbose output\n");
    printf("  --help     -h  Show this help message\n");
}

/**
 * @brief Free CLI parameter memory.
 */
void cli_free(struct CLIParams* cli) {
    free(cli->model_path);
}

bool cli_is_arg(const char* argv, const char* l, const char* s, int argc, int i) {
    return (strcmp(argv, l) == 0 || strcmp(argv, s) == 0) && i + 1 < argc;
}

bool cli_is_flag(const char* argv, const char* l, const char* s) {
    return strcmp(argv, l) == 0 || strcmp(argv, s) == 0;
}

/**
 * @brief Parse CLI arguments into CLIParams struct.
 */
void cli_parse(struct CLIParams* cli) {
    // Set defaults
    cli->model_path = NULL;
    cli->prompt = NULL;
    cli->add_bos = false;
    cli->add_eos = false;
    cli->verbose = false;

    for (int i = 1; i < cli->argc; ++i) {
        if (cli_is_arg(cli->argv[i], "--model", "-m", cli->argc, i)) {
            cli->model_path = strdup(cli->argv[++i]);
        } else if (cli_is_arg(cli->argv[i], "--prompt", "-p", cli->argc, i)) {
            cli->prompt = strdup(cli->argv[++i]);
        } else if (cli_is_flag(cli->argv[i], "--add-bos", "-b")) {
            cli->add_bos = true;
        } else if (cli_is_flag(cli->argv[i], "--verbose", "-v")) {
            cli->verbose = true;
        } else if (cli_is_flag(cli->argv[i], "--help", "-h")) {
            cli_usage(cli->argv[0]);
            cli_free(cli);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", cli->argv[i]);
            cli_usage(cli->argv[0]);
            cli_free(cli);
            exit(EXIT_FAILURE);
        }
    }

    if (!cli->model_path) {
        fprintf(stderr, "Error: --input and --output are required.\n");
        cli_usage(cli->argv[0]);
        cli_free(cli);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, const char* argv[]) {
    struct CLIParams cli = {.argc = argc, .argv = argv};
    cli_parse(&cli);

    // Validate input file
    if (!path_is_file(cli.model_path)) {
        fprintf(stderr, "Error: Model file does not exist: %s\n", cli.model_path);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    Tokenizer* t = tokenizer_load(cli.model_path);
    if (!t) {
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    if (cli.verbose) {
        printf("vocab size: %d\n", t->vocab_size);
        printf("model:\n");
        for (int i = 0; i < t->vocab_size; i++) {
            printf("  %03d -> %s\n", i, t->id_to_token[i]);
        }
    }

    printf("Prompt:\n");
    printf("%s\n\n", cli.prompt);

    printf("Encoding:\n");
    int id_count;
    int* ids = tokenizer_encode(t, cli.prompt, &id_count, cli.add_bos, cli.add_eos);
    if (!ids) {
        fprintf(stderr, "Failed to encode text!\n");
        return EXIT_FAILURE;
    }
    printf("%d ids: ", id_count);
    for (int i = 0; i < id_count; i++) {
        printf("%d ", ids[i]);
    }
    printf("\n\n");

    printf("Decoding:\n");
    char* text = tokenizer_decode(t, ids, id_count);
    if (!text) {
        fprintf(stderr, "Failed to decode ids!\n");
    }
    printf("text: %s\n", text);
    free(text);
    free(ids);

    // Clean up
    tokenizer_free(t);
    cli_free(&cli);

    return EXIT_SUCCESS;
}
