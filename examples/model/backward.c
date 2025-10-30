/**
 * @file examples/model/train.c
 * @brief Train Valerie as a decoder-only generative model.
 * @copyright Copyright © 2025 Austin Berrio
 * @ref https://arxiv.org/abs/1207.0580
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "core/logger.h"
#include "linear/lehmer.h"
#include "linear/type.h"
#include "tokenizer/model.h"
#include "model/valerie.h"
#include "model/blocks.h"

void one_hot(float* x, size_t label, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (label == i) {
            x[i] = 1.0f;
        } else {
            x[i] = 0.0f;
        }
    }
}

float cross_entropy(const float* y_pred, const float* y_true, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (y_true[i] == 1.0f) {
            return -logf(fmaxf(y_pred[i], 1e-8f));
        }
    }
    return 0.0f;  // fallback if not one-hot
}

void print_top_n(float* logits, int vocab_size, int n) {
    printf("Logits (first %d values):\n", n);
    for (int i = 0; i < n && i < vocab_size; i++) {
        printf("  [%4d]: % .5f\n", i, (double) logits[i]);
    }
}

void logit_max_id(float* logits, int vocab_size, float* max_val, int* max_id) {
    *max_id = 0;
    *max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > *max_val) {
            *max_val = logits[i];
            *max_id = i;
        }
    }
    printf("Predicted next token: %d (logit=%.5f)\n", *max_id, (double) *max_val);
}

void print_token_ids(char** id_to_token, int* ids, int len) {
    printf("Token ids (%d):\n", len);
    for (int i = 0; i < len; i++) {
        printf("  [%4d] -> '%s'\n", ids[i], id_to_token[ids[i]]);
    }
    printf("\n");
}

int main(void) {
    lehmer_init(1337);

    Tokenizer t = tokenizer_load("models/tokenizer.model");
    Params p = v_params_new(t.vocab_size);
    Valerie v = v_model_new(t, p, TYPE_Q8);

    LOG_INFO("Model initialized.");
    v_dim_log(v.dim);

    // source ids
    int src_len;
    // ["H", "e", "l", "lo", ",", " "]
    char src[] = "Hello, ";
    // [44, 87, 106, 110, 16, 4]
    int* src_ids = tokenizer_encode(&v.t, src, &src_len, false, false);
    print_token_ids(t.id_to_token, src_ids, src_len);

    // target ids
    int tgt_len;
    // ["H", "e", "l", "lo", ",", " ", "wor", "ld", "!"]
    char tgt[] = "Hello, world!";
    // [44, 87, 106, 110, 16, 4, 140, 107, 5]
    int* tgt_ids = tokenizer_encode(&v.t, tgt, &tgt_len, false, false);
    print_token_ids(t.id_to_token, tgt_ids, tgt_len);

    // do a simple forward pass for now
    int pos = 0;  // increment for each input token id
    int token_id = src_ids[0];  // V : 44 -> "H"
    float* logits = forward(&v, token_id, pos);

    int max_id;
    float max_val;
    print_top_n(logits, v.t.vocab_size, 10);
    logit_max_id(logits, v.t.vocab_size, &max_val, &max_id);

    // calculate probabilities (consider an intemediary buffer if needed)
    softmax(logits, v.t.vocab_size);  // operates in-place

    free(src_ids);
    free(tgt_ids);
    v_model_free(&v);
    LOG_INFO("Model freed cleanly.");
    return 0;
}
