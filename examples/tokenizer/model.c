/**
 * @file   examples/tokenizer/model.c
 * @brief  Driver for training and serializing a BPE tokenizer model (Valerie, C)
 * @copyright Copyright © 2025 Austin Berrio
 * @author Austin Berrio
 *
 * @todo Implement tokenizer serialization once model format is decided.
 * @ref arXiv:1508.07909v5 [cs.CL] 10 Jun 2016
 * @ref arXiv:2505.24689 [cs.CL] 30 May 2025
 * @ref https://aclanthology.org/2025.coling-main.400/
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "core/path.h"
#include "core/strext.h"
#include "core/map.h"

#include "tokenizer/vocab.h"
#include "tokenizer/bpe.h"

#define TOKENIZER_MAGIC 0x766F7870
#define TOKENIZER_VERSION 1

typedef struct SpecialToken {
    char* bos;  // <|bos|>
    char* eos;  // <|eos|>
    char* pad;  // <|pad|>
    char* unk;  // <|unk|>
} SpecialToken;

typedef struct Tokenizer {
    int magic;
    int version;
    BPEModel* model;
    SpecialToken* special;
    HashMap* ascii;
    HashMap* token_to_id;
    HashMap* id_to_token;
} Tokenizer;

// exact bijection: 0..255
HashMap* ascii_generate(void) {
    HashMap* latin1 = hash_map_create(256, HASH_MAP_KEY_TYPE_STRING);
    if (!latin1) {
        return NULL;
    }

    for (size_t i = 0; i < 256; i++) {
        char* k = calloc(2, sizeof(char));
        *k = i;
        k[1] = '\0';

        int* v = malloc(sizeof(int));
        *v = i;

        hash_map_insert(latin1, k, v);
    }

    return latin1;
}

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
    printf("  --input   S     Input plaintext corpus file (required)\n");
    printf("  --output  S     Output directory for tokenizer model (required)\n");
    printf("  --merges  N     Number of BPE merges (default: 10)\n");
    printf("  --verbose | -v  Enable debug/verbose output\n");
    printf("  --help    | -h  Show this help message\n");
}

/**
 * @brief Free CLI parameter memory.
 */
void cli_free(struct CLIParams* cli) {
    free(cli->input_path);
    free(cli->output_dir);
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
        if (strcmp(cli->argv[i], "--input") == 0 && i + 1 < cli->argc) {
            cli->input_path = strdup(cli->argv[++i]);
        } else if (strcmp(cli->argv[i], "--output") == 0 && i + 1 < cli->argc) {
            cli->output_dir = strdup(cli->argv[++i]);
        } else if (strcmp(cli->argv[i], "--merges") == 0 && i + 1 < cli->argc) {
            cli->merges = atoi(cli->argv[++i]);
            if (cli->merges < 1) {
                cli->merges = 10;
            }
        } else if (strcmp(cli->argv[i], "--verbose") == 0 || strcmp(cli->argv[i], "-v") == 0) {
            cli->verbose = true;
        } else if (strcmp(cli->argv[i], "--help") == 0 || strcmp(cli->argv[i], "-h") == 0) {
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

    // Ensure output directory exists (or create it)
    if (!path_is_dir(cli.output_dir)) {
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
    if (cli.verbose) {
        vocab_map_print(vocab);
    }

    // Train BPE merges
    BPEModel* model = bpe_train(vocab, (size_t) cli.merges, cli.verbose);
    if (!model) {
        fprintf(stderr, "Error: Failed to train BPE model.\n");
        vocab_map_free(vocab);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    // @todo: Serialize the model to output_dir/bpe.model or similar
    // Example:
    // char* out_path = path_join(cli.output_dir, "bpe.model");
    // bpe_save(model, out_path);
    // free(out_path);

    // Clean up
    bpe_free(model);
    vocab_map_free(vocab);
    cli_free(&cli);

    return EXIT_SUCCESS;
}
