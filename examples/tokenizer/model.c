/**
 * @file   examples/tokenizer/model.c
 * @brief  Driver for training and serializing a BPE tokenizer model (Valerie, C)
 * @copyright Copyright © 2025 Austin Berrio
 *
 * @todo Implement tokenizer serialization once model format is decided.
 * @ref arXiv:1508.07909v5 [cs.CL] 10 Jun 2016
 * @ref arXiv:2505.24689 [cs.CL] 30 May 2025
 * @ref https://aclanthology.org/2025.coling-main.400/
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
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
    int count;  // -1 if none
    int capacity;  // total capacity for all token strlens
} SpecialToken;

typedef struct Tokenizer {
    SpecialToken* special;  // guidance markers
    HashMap* scores;  // greedy merges using scores
    HashMap* token_to_id;  // v : tok -> id is O(1), worst is O(n)
    char** id_to_token;  // char** is more efficient and is also O(1)
    int vocab_size;  // number of ids to tokens
} Tokenizer;

/**
 * @section Tokenizer clean up
 * @{
 */

void token_map_free(HashMap* m) {
    if (m) {
        hash_iter_free_all(m, free);  // free everything!
    }
}

void ascii_free(HashMap* ascii) {
    token_map_free(ascii);  // free kv-pairs
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

void token_rank_free(HashMap* ranks) {
    token_map_free(ranks);
}

void token_score_free(HashMap* scores) {
    token_map_free(scores);
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

void tokenizer_free(Tokenizer* t) {
    if (t) {
        special_token_free(t->special);
        token_score_free(t->scores);
        token_to_id_free(t->token_to_id);
        id_to_token_free(t->id_to_token, t->vocab_size);
        free(t);
    }
}

/** @} */

/**
 * @section Tokenizer pipeline
 * @{
 */

SpecialToken* special_default_create(void) {
    SpecialToken* special = malloc(sizeof(SpecialToken));
    if (!special) {
        return NULL;
    }

    char* bos = strdup("<|bos|>");
    char* eos = strdup("<|bos|>");
    char* pad = strdup("<|eos|>");
    char* unk = strdup("<|unk|>");

    special->bos = bos;
    special->eos = eos;
    special->pad = pad;
    special->unk = unk;

    special->count = 4;
    special->capacity = strlen(bos) + strlen(eos) + strlen(pad) + strlen(unk) + 1;

    return special;
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

HashSet* token_set_create(BPEModel* model, HashMap* ascii) {
    // create the core token set
    HashSet* set = hash_set_create(model->capacity, HASH_STR);
    if (!set) {
        return NULL;
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

        // free the tuple
        string_split_free(tuple, tuple_count);
    }

    // return the token set
    return set;
}

char** id_to_token_create(HashSet* set, SpecialToken* special, int* out_count) {
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
    // primary owner for allocated vocab mappings
    char** tokens = calloc(1, sizeof(char*));

    // add special tokens to start of array
    if (special) {
        tokens = string_append(strdup(special->bos), tokens, &token_count);
        tokens = string_append(strdup(special->eos), tokens, &token_count);
        tokens = string_append(strdup(special->pad), tokens, &token_count);
        tokens = string_append(strdup(special->unk), tokens, &token_count);
    }

    for (size_t i = 0; i < core_count; i++) {
        tokens = string_append(strdup(core[i]), tokens, &token_count);
    }

    // clean up pre-tokens
    free(core);

    // set the output token count
    *out_count = token_count;

    // return the token list
    return tokens;  // v : i -> t
}

HashMap* token_to_id_create(char** id_to_token, int token_count) {
    HashMap* tokens = hash_map_create(1, HASH_STR);  // str -> id
    if (!tokens) {
        return NULL;
    }

    for (size_t i = 0; i < (size_t) token_count; i++) {
        // shared reference is not to be freed!
        if (HASH_SUCCESS != hash_map_insert(tokens, id_to_token[i], &i)) {
            hash_map_free(tokens);
            return NULL;
        }
    }

    return tokens;  // v : t -> i
}

HashMap* token_rank_create(BPEModel* model) {
    HashMap* ranks = hash_map_create(1, HASH_STR);  // str -> int
    if (!ranks) {
        return NULL;
    }

    // scores require cononical ordering and requires merges as the src
    for (size_t i = 0; i < model->count; i++) {
        BPEMerge merge = model->merges[i];

        // parse out the tuple from the current merge pair
        size_t tuple_count;
        char** tuple = string_split_delim(merge.pair, " ", &tuple_count);
        if (tuple_count != 2) {
            string_split_free(tuple, tuple_count);
            token_score_free(ranks);
            return NULL;
        }
        const char* a = tuple[0];
        const char* b = tuple[1];

        // merge pair: t : a + b
        char* token = string_concat(a, b);

        // copy id
        int* id = malloc(sizeof(int));
        *id = i;  // order matters!

        // map token to id
        hash_map_insert(ranks, token, id);

        // free the tuple
        string_split_free(tuple, tuple_count);
    }

    return ranks;  // v : t -> i
}

HashMap* token_score_create(HashMap* token_to_id, HashMap* ranks) {
    HashMap* scores = hash_map_create(1, HASH_STR);  // str -> float
    if (!scores) {
        return NULL;
    }

    HashEntry* entry;
    HashIt it = hash_iter(token_to_id);
    while ((entry = hash_iter_next(&it))) {
        // get the id for the current score
        int* id = hash_map_search(ranks, entry->key);
        // allocate memory to score
        float* score = malloc(sizeof(float));

        // calc score
        if (!id) {
            *score = -INFINITY;  // no rank
        } else {
            *score = -logf(*id + 1);
        }

        // do not share refs!
        hash_map_insert(scores, strdup(entry->key), score);
    }

    return scores;  // v : t -> f
}

Tokenizer* tokenizer_create(BPEModel* model, SpecialToken* special) {
    if (!model) {
        return NULL;
    }

    Tokenizer* t = calloc(1, sizeof(Tokenizer));
    if (!t) {
        return NULL;
    }

    // Owns special tokens
    t->special = special;  // Optional (can be NULL)

    // Build ASCII table
    HashMap* ascii = ascii_create();
    if (!ascii) {
        goto fail;
    }

    // Create vocab token set
    HashSet* vocab = token_set_create(model, ascii);
    ascii_free(ascii);
    if (!vocab) {
        goto fail;
    }

    // id_to_token (array) and vocab_size
    t->id_to_token = id_to_token_create(vocab, special, &t->vocab_size);

    // Clean up vocab token set
    token_set_free(vocab);
    if (!t->id_to_token) {
        goto fail;
    }

    // token_to_id (map)
    t->token_to_id = token_to_id_create(t->id_to_token, t->vocab_size);
    if (!t->token_to_id) {
        goto fail;
    }

    // ranks (for BPE merges)
    HashMap* ranks = token_rank_create(model);
    if (!ranks) {
        goto fail;
    }

    // scores (for greedy BPE merges)
    t->scores = token_score_create(t->token_to_id, ranks);

    // Clean up rank map
    token_rank_free(ranks);
    if (!t->scores) {
        goto fail;
    }

    return t;

fail:
    // Free all partially allocated fields (handles NULLs fine)
    tokenizer_free(t);
    return NULL;
}

/** @} */

/**
 * @section Tokenizer encoder and decoder
 * @{
 */

int* tokenizer_encode(Tokenizer* t, char* text, bool add_bos, bool add_eos) {
    if (!t || !text) {
        return NULL;  // invalid input
    }

    // create base ids
    size_t id_count = strlen(text);
    if (t->special && t->special->count != -1 && t->special->capacity > 0) {
        id_count += t->special->capacity;  // +n bytes for special tokens
    }

    int* ids = calloc(id_count, sizeof(int));
    if (!ids) {
        return NULL;
    }

    // pre-init ids to discover end-of-seq and resize to fit
    for (size_t i = 0; i < id_count; i++) {
        ids[i] = INT_MIN;
    }

    // pre-process input text to id
    for (size_t i = 0; i < id_count; i++) {
        char token[2] = {text[i], 0};

        int* id = hash_map_search(t->token_to_id, token);
        if (id) {
            ids[i] = *id;
        } else if (t->special && t->special->unk) {
            id = hash_map_search(t->token_to_id, t->special->unk);
            ids[i] = (id) ? *id : -1;  // if -1, unk is not mapped!
        } else {
            ids[i] = -1;  // no unk, just use -1
        }
    }

    // greed merges using scores
    while (true) {
        float best_score = -INFINITY;
        int best_id = -1;

        // scan for best merge pair
        for (size_t i = 0; i < id_count - 1; i++) {
            // get ids
            int id_a = ids[i];  // current
            int id_b = ids[i + 1];  // next

            // token is unknown and there is no sub
            if (id_a == -1 || id_b == -1) {
                continue;
            }

            // probe for a valid merge pair
            char* a = t->id_to_token[id_a];
            char* b = t->id_to_token[id_b];
            char* merge = string_concat(a, b);

            // probe for a valid score
            float* score = hash_map_search(t->scores, merge);
            free(merge);  // clean up
            if (score && *score > best_score) {
                best_score = *score;
                best_id = i;
            }
        }

        if (best_id == -1) {
            break;  // no merges left
        }

        // merge tokens
        char* a = t->id_to_token[ids[best_id]];  // current
        char* b = t->id_to_token[ids[best_id + 1]];  // next
        char* merge = string_concat(a, b);  // a + b

        // get best merge id
        int* merge_id = hash_map_search(t->token_to_id, merge);
        free(merge);  // clean up
        if (!merge_id) {
            break;
        }

        // update current best merge id
        ids[best_id] = *merge_id;
        // remove next best merge id
        memmove(&ids[best_id + 1], &ids[best_id + 2], (id_count - best_id - 2) * sizeof(int));
        // update count for removing next merge id
        id_count--;
    }

    // recalc max base count
    for (size_t i = 0; i < id_count; i++) {
        if (ids[i] == INT_MIN) {
            id_count = i;
            break;  // end of seq
        }
    }

    // insert special bos if enabled and present
    if (add_bos && t->special && t->special->bos) {
        int* id = hash_map_search(t->token_to_id, t->special->bos);
        for (size_t i = id_count; i > 0; --i) {
            ids[i] = ids[i - 1];  // shift everything upward
        }
        ids[0] = id ? *id : -1;  // add bos
    }

    // append special eos if enabled and present
    if (add_eos && t->special && t->special->eos) {
        int* id = hash_map_search(t->token_to_id, t->special->eos);
        ids[id_count] = id ? *id : -1;
    }

    // return predicted tokens
    return ids;
}

/** @} */

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

    // @todo: Serialize the model to output_dir/model.bpe or similar
    // Example:
    // char* out_path = path_join(cli.output_dir, "model.bpe");
    // bpe_save(model, out_path);
    // free(out_path);

    // Add default special tokens
    SpecialToken* special = special_default_create();
    Tokenizer* t = tokenizer_create(model, special);
    if (!t) {
        bpe_free(model);
        vocab_map_free(vocab);
        cli_free(&cli);
        return EXIT_FAILURE;
    }

    // Clean up
    tokenizer_free(t);
    bpe_free(model);
    vocab_map_free(vocab);
    cli_free(&cli);

    return EXIT_SUCCESS;
}
