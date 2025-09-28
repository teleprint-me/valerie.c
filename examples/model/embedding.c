/**
 * @file examples/model/embedding.c
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <math.h>

#include "core/logger.h"
#include "core/path.h"
#include "core/lehmer.h"

#include "tokenizer/model.h"

/**
 * @section Embeddings
 * @{
 */

/**
 * X ∈ ℝ^(D × N)
 * ℝ: Set of Real Numbers
 * X: Input embedding matrix
 * N: Number of embeddings
 * D: Vector of length d
 * Ω_e: Vocab embedding matrix (maps id to one-hot vecs)
 *      Ω is a learned parameter.
 */
float* embeddings_create(size_t vocab_size, size_t vector_len) {
    size_t embed_dim = vocab_size * vector_len;
    float* embeddings = calloc(embed_dim, sizeof(float));
    if (!embeddings) {
        return NULL;
    }

    // initialize the embedding table
    for (size_t i = 0; i < embed_dim; i++) {
        // xavier(fan_in, fan_out)
        embeddings[i] = lehmer_xavier(vocab_size, vector_len);
    }

    return embeddings;
}

/** @} */

/**
 * @section CLI
 * @{
 */

struct CLIParams {
    const char** argv;
    int argc;

    char* model_path;
    char* prompt;
    int64_t seed;
    bool add_bos;
    bool add_eos;
    bool verbose;
};

void cli_usage(const char* prog) {
    printf("Usage: %s --model S [--verbose] [--help]\n", prog);
    printf("  --model    -m  Path to tokenizer model file (required)\n");
    printf("  --prompt   -p  Input text to encode and decode (required)\n");
    printf("  --seed     -s  Linear congruential generator seed\n");
    printf("  --add-bos  -b  Enable bos marker\n");
    printf("  --add-eos  -e  Enable eos marker\n");
    printf("  --verbose  -v  Enable debug output\n");
    printf("  --help     -h  Show this help message\n");
}

void cli_free(struct CLIParams* cli) {
    free(cli->model_path);
    free(cli->prompt);
}

bool cli_is_arg(const char* argv, const char* l, const char* s, int argc, int i) {
    return (strcmp(argv, l) == 0 || strcmp(argv, s) == 0) && i + 1 < argc;
}

bool cli_is_flag(const char* argv, const char* l, const char* s) {
    return strcmp(argv, l) == 0 || strcmp(argv, s) == 0;
}

void cli_parse(struct CLIParams* cli) {
    // Set defaults
    cli->model_path = NULL;
    cli->prompt = NULL;
    cli->seed = 1337;
    cli->add_bos = false;
    cli->add_eos = false;
    cli->verbose = false;

    for (int i = 1; i < cli->argc; ++i) {
        if (cli_is_arg(cli->argv[i], "--model", "-m", cli->argc, i)) {
            cli->model_path = strdup(cli->argv[++i]);
        } else if (cli_is_arg(cli->argv[i], "--prompt", "-p", cli->argc, i)) {
            cli->prompt = strdup(cli->argv[++i]);
        } else if (cli_is_arg(cli->argv[i], "--seed", "-s", cli->argc, i)) {
            cli->seed = atol(cli->argv[++i]);
        } else if (cli_is_flag(cli->argv[i], "--add-bos", "-b")) {
            cli->add_bos = true;
        } else if (cli_is_flag(cli->argv[i], "--add-bos", "-b")) {
            cli->add_eos = true;
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

    if (!cli->model_path || !cli->prompt) {
        fprintf(stderr, "Error: --model and --prompt are required.\n");
        cli_usage(cli->argv[0]);
        cli_free(cli);
        exit(EXIT_FAILURE);
    }
}

/** @} */

int main(int argc, const char* argv[]) {
    struct CLIParams cli = {.argc = argc, .argv = argv};
    cli_parse(&cli);

    // Validate input file
    if (!path_is_file(cli.model_path)) {
        LOG_ERROR("Model file does not exist: %s", cli.model_path);
        goto fail_cli;
    }

    // Load the tokenizer model
    Tokenizer* t = tokenizer_load(cli.model_path);
    if (!t) {
        LOG_ERROR("Failed to load tokenizer model.");
        goto fail_cli;
    }

    // Text to ids
    int id_count;
    int* ids = tokenizer_encode(t, cli.prompt, &id_count, cli.add_bos, cli.add_eos);
    if (!ids) {
        LOG_ERROR("Failed to encode text: %s", cli.prompt);
        goto fail_tokenizer;
    }

    // initialize the linear congruential generator
    lehmer_init(cli.seed);

    // create a toy embedding table
    size_t embed_dim = 16;
    float* embeddings = embeddings_create(t->vocab_size, embed_dim);
    if (!embeddings) {
        goto fail_encoder;
    }

    // Ids to text
    char* text = tokenizer_decode(t, ids, id_count);
    if (!text) {
        fprintf(stderr, "Failed to decode ids!\n");
    }
    free(text);
    free(ids);

    // Clean up
    free(embeddings);
    tokenizer_free(t);
    cli_free(&cli);

    return EXIT_SUCCESS;

fail_encoder:
    free(ids);
fail_tokenizer:
    tokenizer_free(t);
fail_cli:
    cli_free(&cli);
    return EXIT_FAILURE;
}
