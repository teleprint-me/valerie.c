/**
 * @file examples/model/backward.c
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

void log_token_ids(Tokenizer* t, int* ids, int len) {
    printf("Token ids (%d):\n", len);
    for (int i = 0; i < len; i++) {
        printf("  [%4d] -> '%s'\n", ids[i], t->id_to_token[ids[i]]);
    }
    printf("\n");
}

void log_top_n(char* label, Tokenizer* t, float* logits, int n) {
    printf("%s (first %d values):\n", label, n);
    for (int i = 0; i < n && i < t->vocab_size; i++) {
        printf("  [%4d]: % .5f\n", i, (double) logits[i]);
    }
    printf("\n");
}

void log_max_id(Tokenizer* t, float* logits) {
    int max_id = 0;
    float max_val = logits[0];
    for (int i = 1; i < t->vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_id = i;
        }
    }
    char* max_tok = t->id_to_token[max_id];
    printf("Next token: '%s' -> %d (logit=%.5f)\n\n", max_tok, max_id, (double) max_val);
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
    log_token_ids(&t, src_ids, src_len);

    // target ids
    int tgt_len;
    // ["H", "e", "l", "lo", ",", " ", "wor", "ld", "!"]
    char tgt[] = "Hello, world!";
    // [44, 87, 106, 110, 16, 4, 140, 107, 5]
    int* tgt_ids = tokenizer_encode(&v.t, tgt, &tgt_len, false, false);
    log_token_ids(&t, tgt_ids, tgt_len);

    // do a simple forward pass for now
    int pos = 0;  // increment for each input token id
    int token_id = src_ids[0];  // V : 44 -> "H"
    float* logits = forward(&v, token_id, pos);  // maybe output a tensor instead?
    log_top_n("Logits", &t, logits, 10);
    log_max_id(&t, logits);

    // calculate probabilities (consider an intemediary buffer if needed)
    softmax(logits, t.vocab_size);  // operates in-place
    log_top_n("Softmax", &t, logits, 10);
    log_max_id(&t, logits);

    // numerical stability sanity check
    float sum = 0.0f;
    for (int i = 0; i < t.vocab_size; i++) {
        sum += logits[i];  // should sum to 1.0f
    }
    printf("Sum of softmaxed values is %.5f\n", (double) sum);

    // current output mask
    float* target = calloc(t.vocab_size, sizeof(float));
    one_hot(target, tgt_ids[pos + 1], t.vocab_size);

    // current probable loss (model fit or confidence)
    float loss = cross_entropy(logits, target, t.vocab_size);
    printf("Loss: %.6f\n\n", (double) loss);

    // derivative of the logistic activation (logits)
    float* dlogits = calloc(t.vocab_size, sizeof(float));
    for (int i = 0; i < t.vocab_size; i++) {
        dlogits[i] = logits[i] - target[i];
    }
    log_top_n("Derivatives", &t, dlogits, 10);  // sample top derivatives

    // numerical stability sanity check
    float grad_sum = 0.0f;
    for (int i = 0; i < t.vocab_size; i++) {
        grad_sum += dlogits[i];
    }
    printf("Sum of gradients: %.5f\n", (double) grad_sum);  // Should be near 0.0f

    /**
     * Still not sure how I want to handle this.
     * Just sketching out some rough ideas at the moment.
     */

    // embed shape is (vocab_size, d_model)
    size_t vocab_size = tensor_rows(&v.embed.token);  // (out features,)
    size_t d_model = tensor_cols(&v.embed.token);  // (in features,)
    // Tensor dtoken = tensor_new((Shape) {{vocab_size, d_model}, SHAPE_MAT})
    float* dtoken = malloc(vocab_size * d_model * sizeof(float));  // (rows = out, cols = in)

    // gradient w.r.t. embedding weights
    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < d_model; j++) {
            // uint8_t* x_norm = (uint8_t*) v.state.x_norm + j * stride
            float* x_norm = (float*) tensor_view(&v.state.x_norm, j);  // TYPE_F32
            dtoken[i * v.dim.d_model + j] += dlogits[i] * (*x_norm);
        }
    }

    // gradient w.r.t. hidden state
    float* dx_norm = malloc(d_model * sizeof(float));
    for (size_t j = 0; j < d_model; j++) {
        dx_norm[j] = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            float* token = (float*) tensor_view(&v.embed.token, i * d_model + j);
            dx_norm[j] = dlogits[i] * (*token);
        }
    }

    free(dx_norm);
    free(dtoken);
    free(dlogits);
    free(target);
    free(src_ids);
    free(tgt_ids);
    v_model_free(&v);
    LOG_INFO("Model freed cleanly.");
    return 0;
}
