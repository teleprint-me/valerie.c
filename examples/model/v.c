/**
 * @file examples/model/v.c
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @copyright Copyright © 2025 Austin Berrio
 * @ref https://github.com/adriancable/qwen3.c
 * @ref https://arxiv.org/abs/1207.0580
 * ┌──────────────────────────────┐
 * │          Valerie             │  (struct Valerie)
 * │ ┌──────────────────────────┐ │
 * │ │        Dim (Config)      │ │  (d_model, n_heads, hidden, vocab_size, seq_len, etc.)
 * │ └──────────────────────────┘ │
 * │ ┌──────────────────────────┐ │
 * │ │   Embedding Matrix E     │ │  (vocab_size × d_model) ← learned params
 * │ └──────────────────────────┘ │
 * │ ┌──────────────────────────┐ │
 * │ │       Blocks [N]         │ │  (struct Block × d.layers)
 * │ │  ┌────────────────────┐  │ │
 * │ │  │    Block (layer)   │  │ │
 * │ │  │  ┌───────────────┐ │  │ │
 * │ │  │  │ RMSNorm_att   │ │  │ │  (d_model,)
 * │ │  │  └───────────────┘ │  │ │
 * │ │  │  ┌───────────────┐ │  │ │
 * │ │  │  │ Multi-Head    │ │  │ │
 * │ │  │  │ Attention     │ │  │ │  (matrices: d_model ↔ heads × d_head)
 * │ │  │  │ Wq, Wk, Wv, Wo│ │  │ │
 * │ │  │  └───────────────┘ │  │ │
 * │ │  │  ┌───────────────┐ │  │ │
 * │ │  │  │ Residual Add  │ │  │ │
 * │ │  │  └───────────────┘ │  │ │
 * │ │  │  ┌───────────────┐ │  │ │
 * │ │  │  │ RMSNorm_ffn   │ │  │ │  (d_model,)
 * │ │  │  └───────────────┘ │  │ │
 * │ │  │  ┌───────────────┐ │  │ │
 * │ │  │  │ MLP (FFN)     │ │  │ │
 * │ │  │  │ W1, W2, W3    │ │  │ │  (d_model → hidden → d_model)
 * │ │  │  └───────────────┘ │  │ │
 * │ │  │  ┌───────────────┐ │  │ │
 * │ │  │  │ Residual Add  │ │  │ │
 * │ │  │  └───────────────┘ │  │ │
 * │ │  └────────────────────┘  │ │
 * │ └──────────────────────────┘ │
 * │ ┌──────────────────────────┐ │
 * │ │   Final RMSNorm          │ │  (d_model,) applied after last block
 * │ └──────────────────────────┘ │
 * │ ┌──────────────────────────┐ │
 * │ │   Output Projection      │ │  (d_model × vocab_size), weight-tied with E
 * │ └──────────────────────────┘ │
 * │ ┌──────────────────────────┐ │
 * │ │   State (activations)    │ │  (struct State: x, q, k, v, att, logits, kv-cache)
 * │ └──────────────────────────┘ │
 * └──────────────────────────────┘
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


int main(void) {
    lehmer_init(1337);

    Tokenizer t = tokenizer_load("models/tokenizer.model");
    Params p = v_params_new(t.vocab_size);
    Valerie v = v_model_new(t, p, TYPE_F32);

    LOG_INFO("Model initialized.");
    v_dim_log(v.dim);

    int token_id = 0;
    int pos = 0;
    float* logits = forward(&v, token_id, pos);

    int top_n = 10;
    printf("Logits (first %d values):\n", top_n);
    for (int i = 0; i < top_n && i < v.dim.vocab_size; i++) {
        printf("  [%4d]: % .5f\n", i, (double) logits[i]);
    }

    int max_id = 0;
    float max_val = logits[0];
    for (int i = 1; i < v.dim.vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_id = i;
        }
    }
    printf("Predicted next token: %d (logit=%.5f)\n", max_id, (double) max_val);

    v_model_free(&v);
    LOG_INFO("Model freed cleanly.");
    return 0;
}
