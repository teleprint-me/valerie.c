/**
 * @file examples/model/v.c
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @copyright Copyright © 2025 Austin Berrio
 * @note Valerie is not a replica of Qwen3 and is incompatible as a result.
 * @ref https://github.com/adriancable/qwen3.c
 *
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

#include <assert.h>
#include <math.h>

#include "core/type.h"
#include "core/lehmer.h"

#include "tokenizer/model.h"

#include "model/activation.h"
#include "model/matrix.h"
#include "model/blocks.h"

/**
 * @section Dimensions
 */

typedef struct Dim {
    int d_model;  // width of the model (hidden size)
    int hidden;  // FFN hidden dimension (usually 4 * d_model)
    int layers;  // number of transformer blocks (depth)
    int heads;  // number of attention heads
    int head_dim;  // per-head dimension (d_model / heads)
    int proj_dim;  // heads * head_dim
    int kv_dim;  // kv_heads * head_dim
    int kv_mul;  // multi-query attention (heads / kv_heads)
    int kv_heads;  // number of key/value heads (multiquery attention)
    int vocab_size;  // vocabulary size
    int seq_len;  // maximum context length
} Dim;

/**
 * @section Forward
 */

typedef struct Attention {
    quant8_t* Wq;  // (d_model, n_heads * head_dim)
    quant8_t* Wk;  // (d_model, n_kv_heads * head_dim)
    quant8_t* Wv;  // (d_model, n_kv_heads * head_dim)
    quant8_t* Wo;  // (n_heads * head_dim, d_model)

    float* rms;  // (d_model,) RMSNorm params
} Attention;

typedef struct FeedForward {
    quant8_t* W1;  // (hidden, d_model)
    quant8_t* W2;  // (d_model, hidden)
    quant8_t* W3;  // (hidden, d_model)

    float* rms;  // (d_model,) RMSNorm params
} FeedForward;

typedef struct Layer {
    Attention attn;  // multi-headed self attention
    FeedForward ffn;  // feed-forward network
    float* rms_attn;  // (d_model,) RMSNorm params
    float* rms_ffn;  // (d_model,) RMSNorm params
} Layer;

typedef struct State {
    /// @note Output weights are tied to embeddings
    /// @ref https://arxiv.org/abs/1608.05859
    float* embeddings;  // (vocab_size, d_model)

    float* x;  // (d_model,)

    float* rms;  // (d_model,)

    float* q;  // (d_model,)
    float* k;  // (d_model,)
    float* v;  // (d_model,)

    float* att;  // (heads, seq_len)
    float* logits;  // (vocab_size,)

    float* k_cache;  // (seq_len, d_model)
    float* v_cache;  // (seq_len, d_model)

    quant8_t* qattn;
    quant8_t* qffn;
} State;

typedef struct Valerie {
    Tokenizer* t;
    Layer* layers;
    State state;
    Dim dim;
} Valerie;

/**
 * @section Backward
 */

typedef struct AttentionOpt {
    // Derivatives
    bfloat16_t* dWk;
    bfloat16_t* dWq;
    bfloat16_t* dWv;
    bfloat16_t* dWo;

    // Velocities
    bfloat16_t* vWk;
    bfloat16_t* vWq;
    bfloat16_t* vWv;
    bfloat16_t* vWo;
} AttentionOpt;

typedef struct FeedForwardOpt {
    // Derivatives
    bfloat16_t* dW1;
    bfloat16_t* dW2;
    bfloat16_t* dW3;

    // Velocities
    bfloat16_t* vW1;
    bfloat16_t* vW2;
    bfloat16_t* vW3;
} FeedForwardOpt;

typedef struct LayerOpt {
    AttentionOpt attn;
    FeedForwardOpt ffn;
    float* drms_attn;  // (d_model,) RMSNorm params
    float* drms_ffn;  // (d_model,) RMSNorm params
} LayerOpt;

/**
 * Model pipeline
 * @{
 */

float* embed_new(unsigned vocab_size, unsigned embed_dim) {
    float* E = mat_new(vocab_size, embed_dim, TYPE_FLOAT32);
    if (!E) {
        return NULL;
    }

    mat_xavier(E, vocab_size * embed_dim, vocab_size, embed_dim);
    return E;
}

/** @} */

int main(void) {
    lehmer_init(1337);
    return 0;
}
