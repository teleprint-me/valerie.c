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
#include "core/set.h"
#include "core/sort.h"

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
    SpecialToken* special;  // guidance markers
    HashMap* ascii;  // out of vocab table
    HashMap* token_to_id;  // v : tok -> id is O(1), worst is O(n)
    char** id_to_token;  // char** is more efficient and is also O(1)
    int vocab_size;  // number of ids to tokens
} Tokenizer;

void ascii_free(HashMap* ascii) {
    if (ascii) {
        hash_iter_free_all(ascii, free);  // free key-value pairs
    }
}

void token_set_free(HashSet* tokens) {
    if (tokens) {
        hash_iter_free_all(tokens, NULL);  // only free keys!
    }
}

void id_to_token_free(char** tokens, size_t token_count) {
    if (tokens && token_count > 0) {
        string_split_free(tokens, token_count);  // free everything!
    }
}

void token_to_id_free(HashMap* tokens) {
    if (tokens) {
        hash_map_free(tokens);  // do not free keys or values!
    }
}

void special_token_free(SpecialToken* special) {
    if (special) {
        if (special->bos) {
            free(special->bos);
        }
        if (special->eos) {
            free(special->eos);
        }
        if (special->pad) {
            free(special->pad);
        }
        if (special->unk) {
            free(special->unk);
        }
        free(special);
    }
}

HashMap* ascii_create(void) {
    HashMap* latin1 = hash_map_create(256, HASH_STR);
    if (!latin1) {
        return NULL;
    }

    // exact bijection: 0..255
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

HashSet* token_set_create(BPEModel* model) {
    // create the core token set
    HashSet* set = hash_set_create(model->capacity, HASH_STR);
    if (!set) {
        return NULL;
    }

    // generate base tokens for OOV
    HashMap* ascii = ascii_create();
    if (!ascii) {
        hash_set_free(set);
    }

    // Add base tokens for OOV
    HashEntry* entry;
    HashIt it = hash_iter(ascii);
    while ((entry = hash_iter_next(&it))) {
        // returns true on success
        hash_set_add(set, strdup(entry->key));
    }

    // parse out merges from model
    for (size_t i = 0; i < model->count; i++) {
        // get the current merge
        BPEMerge merge = model->merges[i];

        // parse out the tuple from the current merge pair
        size_t tuple_count;
        char** tuple = string_split_delim(merge.pair, " ", &tuple_count);
        if (tuple_count != 2) {
            ascii_free(ascii);
            token_set_free(set);
            string_split_free(tuple, tuple_count);
            return NULL;
        }
        const char* a = tuple[0];
        const char* b = tuple[1];

        // merge pair: t : a + b
        char* token = string_concat(a, b);

        // add the token to the set
        hash_set_add(set, token);
    }

    // return the token set
    return set;
}

char** id_to_token_create(HashSet* set, SpecialToken* special, size_t* out_count) {
    // create core token list
    size_t core_count = 0;
    // create a shallow copy
    char** core = calloc(1, sizeof(char*));

    // add core token set to list
    HashEntry* entry = NULL;
    HashIt it = hash_iter(set);
    while ((entry = hash_iter_next(&it))) {
        /// @note does **not** duplicate keys
        core = string_append(entry->key, core, &core_count);
        if (!core) {
            free(core);
            return NULL;
        }
    }

    // Sort the core token array
    heap_sort_str(core, core_count);

    // Create the output token list
    size_t token_count = 0;
    char** tokens = calloc(1, sizeof(char*));

    // add special tokens to start of array
    tokens = string_append(strdup(special->bos), tokens, &token_count);
    tokens = string_append(strdup(special->eos), tokens, &token_count);
    tokens = string_append(strdup(special->pad), tokens, &token_count);
    tokens = string_append(strdup(special->unk), tokens, &token_count);

    for (size_t i = 0; i < core_count; i++) {
        tokens = string_append(strdup(core[i]), tokens, &token_count);
    }

    // clean up pre-tokens
    free(core);

    // set the output token count
    *out_count = token_count;

    // return the token list
    return tokens;
}

HashMap* token_to_id_create(char** id_to_tokens, size_t token_count) {
    HashMap* tokens = hash_map_create(1, HASH_STR);  // str -> id
    if (!tokens) {
        return NULL;
    }

    for (size_t i = 0; i < token_count; i++) {
        // shared reference is not to be freed!
        if (HASH_SUCCESS != hash_map_insert(tokens, id_to_tokens[i], &i)) {
            hash_map_free(tokens);
            return NULL;
        }
    }

    return tokens;
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
