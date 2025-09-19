/**
 * @file      tokenizer/bpe.c
 * @brief     Byte-Pair Encoding (BPE) merges and model API.
 * @copyright Copyright © 2025 Austin Berrio
 *
 * This module implements the core BPE model and merge operations,
 * including training, serialization, and manipulation of BPE merges.
 *
 * @note All APIs are designed for use with space-delimited symbol strings
 *       (e.g., "l o o k e d"). Ownership of returned pointers is specified.
 */

#include <stdio.h>

#include "core/path.h"
#include "core/strext.h"
#include "core/map.h"

#include "tokenizer/vocab.h"

/**
 * @def BPE_MAGIC
 * @brief Magic number for merges file format identification ("pair", little-endian).
 */
#define BPE_MAGIC 0x70616972

/**
 * @def BPE_VERSION
 * @brief Current version of the merges file format.
 */
#define BPE_VERSION 1

typedef struct BPEMerge {
    char* pair;
    int freq;
} BPEMerge;

typedef struct BPEModel {
    BPEMerge* merges;
    size_t count;
    size_t capacity;
} BPEModel;

/**
 * BPE clean up
 * @{
 */

void bpe_free_model(BPEModel* model) {
    if (model) {
        if (model->merges) {
            for (size_t i = 0; i < model->count; i++) {
                free(model->merges[i].pair);
            }
            free(model->merges);
        }
        free(model);
    }
}

/** @} */

/**
 * BPE serialization
 * @{
 */

bool bpe_model_save(BPEModel* model, const char* path) {
    char* dirname = path_dirname(path);
    path_mkdir(dirname);  // returns true on success
    free(dirname);

    FILE* file = fopen(path, "wb");
    if (!file) {
        return false;
    }

    int magic = BPE_MAGIC;
    fwrite(&magic, sizeof(int), 1, file);

    int version = BPE_VERSION;
    fwrite(&version, sizeof(int), 1, file);

    fwrite(&model->count, sizeof(size_t), 1, file);
    fwrite(&model->capacity, sizeof(size_t), 1, file);

    for (size_t i = 0; i < model->count; i++) {
        int pair_len = strlen(model->merges[i].pair);
        fwrite(&pair_len, sizeof(int), 1, file);
        fwrite(model->merges[i].pair, sizeof(char), pair_len, file);
        fwrite(&model->merges[i].freq, sizeof(int), 1, file);
    }

    fclose(file);
    return true;
}

BPEModel* bpe_model_load(const char* path) {
    if (!path_is_file(path)) {
        return NULL;
    }

    FILE* file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }

    int magic;
    fread(&magic, sizeof(int), 1, file);
    if (magic != BPE_MAGIC) {
        fclose(file);
        return NULL;
    }

    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != BPE_VERSION) {
        fclose(file);
        return NULL;
    }

    BPEModel* model = malloc(sizeof(BPEModel));
    if (!model) {
        fclose(file);
        return NULL;
    }

    fread(&model->count, sizeof(size_t), 1, file);
    fread(&model->capacity, sizeof(size_t), 1, file);

    model->merges = calloc(model->capacity, sizeof(BPEMerge));
    for (size_t i = 0; i < model->count; i++) {
        int pair_len;
        fread(&pair_len, sizeof(int), 1, file);

        char* pair = calloc(pair_len + 1, sizeof(char));
        fread(pair, sizeof(char), pair_len, file);
        pair[pair_len] = '\0';
        model->merges[i].pair = pair;

        fread(&model->merges[i].freq, sizeof(int), 1, file);
    }

    fclose(file);
    return model;
}

/** @} */

// collect vocab pairs
// once all pairs have been exhausted,
// the pairs function must return NULL to indicate the end of operation
HashMap* bpe_pairs(HashMap* vocab) {
    HashMap* new_pairs = hash_map_create(hash_map_size(vocab), HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        size_t sym_count = 0;
        char** syms = string_split_delim(entry->key, " ", &sym_count);

        for (size_t i = 0; i < sym_count - 1; i++) {
            // Create the symbol tuple
            char* tuple[] = {syms[i], syms[i + 1]};

            // Create the symbol pair
            char* pair = string_join(tuple, 2, " ");

            // Check for pair existence
            int* freq = hash_map_search(new_pairs, pair);
            if (!freq) {  // pair does not exist
                int* new_freq = malloc(sizeof(int));
                *new_freq = *(int*) entry->value;
                hash_map_insert(new_pairs, pair, new_freq);
            } else {  // pair already exists
                *freq += *(int*) entry->value;
                free(pair);
            }
        }

        string_split_free(syms, sym_count);
    }

    return new_pairs;
}

char* bpe_best(HashMap* pairs, int* out_freq) {
    char* best_pair = NULL;
    int best_freq = -1;

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(pairs);
    while ((entry = hash_map_next(&it))) {
        char* pair = entry->key;
        int* freq = entry->value;

        if (*freq > best_freq) {
            best_pair = pair;
            best_freq = *freq;
        } else if (*freq == best_freq && best_pair && strcmp(pair, best_pair) < 0) {
            best_pair = pair;  // lexicographic tie-break
        }
    }

    if (out_freq) {
        *out_freq = best_freq;
    }

    // Caller must validate and free best pair
    return best_pair ? strdup(best_pair) : NULL;
}

HashMap* bpe_merges(HashMap* vocab, const char* best_pair) {
    if (!vocab || !best_pair) {
        return NULL;  // exhausted all pairs
    }

    // Parse tuple: (a, b)
    size_t tuple_count = 0;
    char** tuple = string_split_delim(best_pair, " ", &tuple_count);
    if (tuple_count != 2) {
        string_split_free(tuple, tuple_count);
        return NULL;
    }
    const char* a = tuple[0];
    const char* b = tuple[1];

    // New vocab map
    HashMap* new_vocab = hash_map_create(hash_map_size(vocab), HASH_MAP_KEY_TYPE_STRING);

    HashMapEntry* entry;
    HashMapIterator it = hash_map_iter(vocab);
    while ((entry = hash_map_next(&it))) {
        // Split word into symbols
        size_t sym_count = 0;
        char** syms = string_split_delim(entry->key, " ", &sym_count);

        // Prepare output buffer (never longer than input)
        size_t out_count = 0;
        char** out = calloc(sym_count, sizeof(char*));

        size_t i = 0;
        while (i < sym_count) {
            if (i + 1 < sym_count && strcmp(syms[i], a) == 0 && strcmp(syms[i + 1], b) == 0) {
                // Merge: concat a + b
                size_t merge_count = strlen(a) + strlen(b) + 1;
                char* merge = malloc(merge_count);
                strcpy(merge, a);
                strcat(merge, b);

                out[out_count++] = merge;
                i += 2;  // skip b
            } else {
                out[out_count++] = strdup(syms[i]);
                i += 1;
            }
        }

        // Join tokens into new word (space-delimited)
        char* new_word = string_join(out, out_count, " ");

        // Insert new word into new vocab
        int* freq = hash_map_search(new_vocab, new_word);
        if (!freq) {
            int* new_freq = malloc(sizeof(int));
            *new_freq = *(int*) entry->value;
            hash_map_insert(new_vocab, new_word, new_freq);
        } else {
            *freq += *(int*) entry->value;
            free(new_word);
        }

        // Free intermediate arrays
        string_split_free(out, out_count);
        string_split_free(syms, sym_count);
    }

    string_split_free(tuple, tuple_count);
    return new_vocab;
}

BPEModel* bpe_train(HashMap* vocab, size_t n_merges, bool verbose) {
    // Create a shallow copy of the input vocab
    HashMap* internal_vocab = vocab_map_copy(vocab);  // always copy!
    if (!internal_vocab) {
        return NULL;
    }

    // Create a new BPE model
    BPEModel* model = malloc(sizeof(BPEModel));
    if (!model) {
        vocab_map_free(internal_vocab);
        return NULL;
    }

    // Collect the best merge pairs (used to build the model)
    model->count = 0;
    model->capacity = 8;
    model->merges = malloc(model->capacity * sizeof(BPEMerge));
    if (!model->merges) {
        vocab_map_free(internal_vocab);
        bpe_free_model(model);
        return NULL;
    }

    // Execute BPE merge steps
    for (size_t i = 0; i < n_merges; i++) {
        // Build symbol pairs from vocab
        HashMap* pairs = bpe_pairs(internal_vocab);
        if (verbose) {
            vocab_map_print(pairs);  // debug
        }

        // Calculate the best pairs
        int best_freq;
        char* best_pair = bpe_best(pairs, &best_freq);
        if (!best_pair) {
            printf("[bpe] Exhausted all possible merge pairs at step %zu.\n", i);
            vocab_map_free(pairs);
            break;
        }

        // Observe the best merge pair
        printf("[bpe] step=%zu, best_freq=%d, best_pair=%s\n", i, best_freq, best_pair);

        // Grow array if needed
        if (model->count == model->capacity) {
            size_t new_cap = model->capacity * 2;
            BPEMerge* temp = realloc(model->merges, new_cap * sizeof(BPEMerge));
            if (!temp) {
                // Free everything up to now
                free(best_pair);
                vocab_map_free(pairs);
                vocab_map_free(internal_vocab);
                bpe_free_model(model);
                return NULL;
            }
            model->merges = temp;
            model->capacity = new_cap;
        }

        // Append the best merge pair
        model->merges[model->count++] = (BPEMerge) {strdup(best_pair), best_freq};

        // Merge all matching pairs
        HashMap* new_vocab = bpe_merges(internal_vocab, best_pair);
        if (verbose) {
            vocab_map_print(new_vocab);
        }

        // Clean up
        free(best_pair);
        vocab_map_free(pairs);
        vocab_map_free(internal_vocab);

        // Update
        internal_vocab = new_vocab;
    }

    /// @note ASAN doesn't catch this.
    vocab_map_free(internal_vocab);  // Always free before exiting.
    return model;
}

/**
 * Command-line interface
 * @{
 */

struct CLIParams {
    const char** argv;
    int argc;

    char* vocab_path;
    int merges;
    bool debug;
};

void cli_usage(struct CLIParams* cli) {
    printf("Usage: %s [--vocab S] ...\n", cli->argv[0]);
    printf("--vocab   S Plain text input file (default: samples/simple.txt)\n");
    printf("--merges  N Number of merges (default: 10)\n");
    printf("--verbose B Enables debug log (default: false)\n");
}

void cli_free(struct CLIParams* cli) {
    if (cli->vocab_path) {
        free(cli->vocab_path);
    }
}

void cli_parse(struct CLIParams* cli) {
    if (cli->argc < 2) {
        cli_usage(cli);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < cli->argc; i++) {
        if (strcmp(cli->argv[i], "--vocab") == 0 && i + 1 < cli->argc) {
            cli->vocab_path = strdup(cli->argv[++i]);
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
        .vocab_path = NULL,
        .merges = 1,
        .debug = false,
    };
    cli_parse(&cli);

    // Ensure vocab path is not null
    if (!cli.vocab_path) {
        cli.vocab_path = strdup("samples/simple.txt");
    }

    // Build word frequencies from text file
    HashMap* vocab = vocab_build(cli.vocab_path);
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
    bpe_free_model(model);
    vocab_map_free(vocab);
    free(cli.vocab_path);
    return EXIT_SUCCESS;
}
