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

#include "model/activation.h"

/**
 * @section One-hot encoder
 * @{
 */

float* one_hot_encode(size_t label, size_t n_classes) {
    float* vector = calloc(n_classes, sizeof(float));
    if (label < n_classes) {
        vector[label] = 1.0f;
    }
    return vector;
}

/** @} */

/**
 * @section Cross-entropy
 * @{
 */

// y_pred: predicted probabilities (softmax output), shape (n,)
// y_true: target one-hot vector, shape (n,)
// n: number of classes
float cross_entropy(const float* y_pred, const float* y_true, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        // Add epsilon for numerical stability
        float p = fmaxf(y_pred[i], 1e-8f);
        loss -= y_true[i] * logf(p);
    }
    return loss / n;  // Equivalent to log(softmax(x, n))
}

/** @} */

/**
 * @section Softmax
 * @{
 */

void softmax(float* x, int n) {
    float max_score = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_score);
        sum += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/** @} */

/**
 * @section Matrix ops
 * @note inputs are columns. outputs are rows.
 * @{
 */

// Create a row-major matrix
float* mat_new(size_t out, size_t in) {
    float* mat = calloc(out * in, sizeof(float));
    if (!mat) {
        return NULL;
    }
    return mat;
}

void mat_xavier(float* x, size_t n, size_t out, size_t in) {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        // xavier(fan_in, fan_out)
        x[i] = lehmer_xavier(in, out);  // thread is local
    }
}

// Row-major matrix transposition (rows x cols) into (cols x rows)
void mat_T(const float* X, float* X_T, size_t out, size_t in) {
#pragma omp parallel for
    for (size_t i = 0; i < out; i++) {
        for (size_t j = 0; j < in; j++) {
            // W_T[j * rows + i] = W[i * cols + j];
            X_T[j * out + i] = X[i * in + j];
        }
    }
}

// Row-major matrix multiplication (y = Wx + b)
// bias is omitted because it's always 0
void mat_mul(float* y, float* W, float* x, size_t out, size_t in) {
#pragma omp parallel for
    for (size_t i = 0; i < out; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < in; j++) {
            sum += W[i * in + j] * x[j];
        }
        y[i] = sum;
    }
}

// dW = δ_next ⊗ xᵀ (outer product)
void mat_dW(float* dW, float* d_next, float* x, size_t out, size_t in) {
#pragma omp parallel for
    for (size_t i = 0; i < out; i++) {
        for (size_t j = 0; j < in; j++) {
            dW[i * in + j] = d_next[i] * x[j];
        }
    }
}

// Backprop: dy = (W_next^T * d_next) ⊙ f'(x) (chain rule)
void mat_chain(float* dy, float* W_next, float* d_next, float* z, size_t out, size_t out_next) {
#pragma omp parallel for
    for (size_t i = 0; i < out; i++) {
        float sum = 0.0f;

        // Iterate over next-layer neurons
        for (size_t j = 0; j < out_next; j++) {
            sum += W_next[j * out + i] * d_next[j];
        }

        dy[i] = sum * silu_prime(z[i]);
    }
}

// Apply SGD update to weights
void mat_sgd(
    float* W,
    float* vW,
    const float* dW,
    size_t out,
    size_t in,
    float lr,
    float mu,
    float tau,
    int nesterov,
    float lambda
) {
#pragma omp parallel for
    for (size_t i = 0; i < out; i++) {
        for (size_t j = 0; j < in; j++) {
            size_t idx = i * in + j;
            float g = dW[idx];

            // L2 regularization
            if (lambda > 0.0f) {
                g += lambda * W[idx];
            }

            // Momentum
            if (mu > 0.0f) {
                vW[idx] = mu * vW[idx] + (1.0f - tau) * g;

                if (nesterov) {
                    g += mu * vW[idx];  // Lookahead
                } else {
                    g = vW[idx];
                }
            }

            // Weight update
            W[idx] -= lr * g;
        }
    }
}

/** @} */

/**
 * @section Embeddings
 * @{
 */

/**
 * X ∈ ℝ^(D × N)
 * ℝ: Set of Real Numbers
 * X: Input embedding matrix
 * N: Number of embeddings (vocab size)
 * D: Vector length (embed dim)
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

/**
 * Ω_e ∈ ℝ^(D × |V|)
 * Ω_e: Vocab embedding matrix (maps id to one-hot vecs)
 *      Ω is a learned parameter.
 * D: Vector length
 * V: Mapped word embedding
 * |V|: Magnitude of the word embedding
 */
void embeddings_lookup(
    float* out, const float* e, const int* ids, size_t seq_len, size_t embed_dim
) {
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < embed_dim; d++) {
            out[i * embed_dim + d] = e[ids[i] * embed_dim + d];
        }
    }
}

/**
 *
 * T ∈ ℝ^(|V| × N)
 * |V|: Vocab size
 * N: Number of input tokens (aka seq len)
 *    where nth column corresponds to nth token and is a |V| × 1 one-hot vector
 */
/// @todo

/**
 * Log embeddings table to stdout
 */
void embeddings_print(
    const float* e, int* ids, size_t seq_len, size_t embed_dim, char** id_to_token
) {
    for (size_t i = 0; i < seq_len; ++i) {
        printf("id %3d (%-8s):", ids[i], id_to_token ? id_to_token[ids[i]] : "");
        for (size_t d = 0; d < embed_dim; ++d) {
            printf(" % .4f", (double) e[ids[i] * embed_dim + d]);
        }
        printf("\n");
    }
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
    int seq_len;
    int* ids = tokenizer_encode(t, cli.prompt, &seq_len, cli.add_bos, cli.add_eos);
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

    embeddings_print(embeddings, ids, seq_len, embed_dim, t->id_to_token);

    // Ids to text
    char* text = tokenizer_decode(t, ids, seq_len);
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
