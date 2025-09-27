/**
 * @file      examples/tokenizer/train.c
 * @brief     Train and serialize a BPE tokenizer model
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
#include "core/map.h"

#include "tokenizer/vocab.h"
#include "tokenizer/bpe.h"
#include "tokenizer/model.h"

/**
 * @struct CLIParams
 * @brief Command-line parameters for tokenizer training.
 */
struct CLIParams {
    const char** argv;
    int argc;

    char* input_path;  ///< Input text corpus (plaintext)
    char* output_dir;  ///< Output directory for model files
    int merges;  ///< Number of BPE merges to perform
    bool verbose;  ///< Enable verbose/debug output
};

/**
 * @brief Print usage instructions for the tokenizer trainer.
 */
void cli_usage(const char* prog) {
    printf("Usage: %s --input S --output S [--merges N] [--verbose]\n", prog);
    printf("  --input    -i  Input plaintext corpus file (required)\n");
    printf("  --output   -o  Output directory for tokenizer model (required)\n");
    printf("  --merges   -m  Number of BPE merges (default: 10)\n");
    printf("  --verbose  -v  Enable debug output\n");
    printf("  --help     -h  Show this help message\n");
}

/**
 * @brief Free CLI parameter memory.
 */
void cli_free(struct CLIParams* cli) {
    free(cli->input_path);
    free(cli->output_dir);
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
    cli->input_path = NULL;
    cli->output_dir = NULL;
    cli->merges = 10;
    cli->verbose = false;

    for (int i = 1; i < cli->argc; ++i) {
        if (cli_is_arg(cli->argv[i], "--input", "-i", cli->argc, i)) {
            cli->input_path = strdup(cli->argv[++i]);
        } else if (cli_is_arg(cli->argv[i], "--output", "-o", cli->argc, i)) {
            cli->output_dir = strdup(cli->argv[++i]);
        } else if (cli_is_arg(cli->argv[i], "--merges", "-m", cli->argc, i)) {
            cli->merges = atoi(cli->argv[++i]);
            if (cli->merges < 1) {
                cli->merges = 10;
            }
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

    if (!cli->input_path || !cli->output_dir) {
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
    if (!path_is_file(cli.input_path)) {
        fprintf(stderr, "Error: Input file '%s' does not exist.\n", cli.input_path);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    // Validate output directory
    if (path_is_file(cli.output_dir)) {
        fprintf(stderr, "Error: Output directory can not be a file.\n");
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    // Ensure output directory exists
    if (!path_is_dir(cli.output_dir)) {
        // Create output directory if it doesn't exist
        if (!path_mkdir(cli.output_dir)) {
            fprintf(stderr, "Error: Could not create output directory '%s'.\n", cli.output_dir);
            cli_free(&cli);
            return EXIT_FAILURE;
        }
    }

    // Build vocabulary from input text
    HashMap* vocab = vocab_build(cli.input_path);
    if (!vocab) {
        fprintf(stderr, "Error: Failed to build vocab from '%s'.\n", cli.input_path);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    // Train BPE merges
    BPEModel* model = bpe_train(vocab, (size_t) cli.merges, cli.verbose);
    if (!model) {
        fprintf(stderr, "Error: Failed to train BPE model.\n");
        vocab_map_free(vocab);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    if (cli.verbose) {
        printf("BPEModel:\n");
        for (size_t i = 0; i < model->count; i++) {
            printf("  %s -> %d\n", model->merges[i].pair, model->merges[i].freq);
        }
        printf("\n");
    }

    // Serialize merges to output_dir
    char* out_path = path_join(cli.output_dir, "bpe.model");
    bpe_save(model, out_path);
    printf("Saved merges to %s\n\n", out_path);
    free(out_path);

    // Add default special tokens
    SpecialToken* special = token_special_create(NULL, NULL, NULL, NULL);
    if (!special || !special->bos || !special->eos || !special->pad || !special->unk) {
        fprintf(stderr, "Failed to create special tokens.\n");
        return EXIT_FAILURE;
    }

    Tokenizer* t = tokenizer_create(model, special);
    if (!t) {
        bpe_free(model);
        vocab_map_free(vocab);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    // Print debug info if enabled
    if (cli.verbose) {
        printf("vocab size: %d\n", t->vocab_size);
        printf("model:\n");
        for (int i = 0; i < t->vocab_size; i++) {
            printf("  %03d -> %s\n", i, t->id_to_token[i]);
        }
        printf("\n");
    }

    // Serialize tokenizer to output_dir
    out_path = path_join(cli.output_dir, "tokenizer.model");
    tokenizer_save(t, out_path);
    printf("Saved tokenizer to %s\n\n", out_path);
    free(out_path);

    printf("Encoding:\n");
    int id_count;
    int* ids = tokenizer_encode(t, "Hello, world!", &id_count, false, false);
    if (!ids) {
        fprintf(stderr, "Failed to encode text!\n");
        return EXIT_FAILURE;
    }
    printf("%d ids: ", id_count);
    for (int i = 0; i < id_count; i++) {
        printf("%d ", ids[i]);
    }
    printf("\n");

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
    bpe_free(model);
    vocab_map_free(vocab);
    cli_free(&cli);

    return EXIT_SUCCESS;
}
