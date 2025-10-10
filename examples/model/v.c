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
} Attention;

typedef struct FeedForward {
    quant8_t* W1;  // (hidden, d_model)
    quant8_t* W2;  // (d_model, hidden)
    quant8_t* W3;  // (hidden, d_model)
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
    float* E;  // (vocab_size, d_model)

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

// Default micro configuration (~8–10M params)
// Reference: https://arxiv.org/pdf/2305.13245
Dim v_dim_new(void) {
    const int d_model = 320;  // Model width
    const int heads = 32;  // Query heads
    const int kv_heads = 4;  // Shared key/value heads (GQA/MQA)
    const int head_dim = d_model / heads;  // 10 dims per head

    return (Dim) {
        .d_model = d_model,
        .hidden = 4 * d_model,  // FFN inner dimension
        .layers = 6,  // Transformer depth
        .heads = heads,
        .head_dim = head_dim,
        .proj_dim = heads * head_dim,  // == d_model
        .kv_heads = kv_heads,
        .kv_mul = heads / kv_heads,  // 8 → 8 Q per K/V
        .kv_dim = kv_heads * head_dim,  // total shared KV width
        .vocab_size = 149,  // tiny test vocab
        .seq_len = 128,  // context length
    };
}

quant8_t* mat_new_q8(size_t rows, size_t cols, size_t block_size) {
    quant8_t* q8 = malloc(sizeof(quant8_t));
    if (!q8) {
        return NULL;
    }

    size_t len = rows * cols;
    size_t blocks = (len + block_size - 1) / block_size;

    q8->q = calloc(len, sizeof(int8_t));
    q8->s = calloc(blocks, sizeof(uint8_t));

    return q8;
}

void mat_init_q8(quant8_t* q8, size_t rows, size_t cols, size_t block_size) {
    float* src = malloc(rows * cols * sizeof(float));

    size_t len = rows * cols;
    for (size_t i = 0; i < len; i++) {
        src[i] = lehmer_xavier(rows, cols);  // (fan_out, fan_in)
    }

    q8_encode(q8, src, len, block_size);
    free(src);
}

Attention v_attn_new(Dim* d) {
    Attention attn = {0};

    attn.Wq = mat_new_q8(d->d_model, d->heads * d->head_dim, Q8_BLOCK_SIZE);
    attn.Wk = mat_new_q8(d->d_model, d->kv_heads * d->head_dim, Q8_BLOCK_SIZE);
    attn.Wv = mat_new_q8(d->d_model, d->kv_heads * d->head_dim, Q8_BLOCK_SIZE);
    attn.Wo = mat_new_q8(d->heads * d->head_dim, d->d_model, Q8_BLOCK_SIZE);

    mat_init_q8(attn.Wq, d->d_model, d->heads * d->head_dim, Q8_BLOCK_SIZE);
    mat_init_q8(attn.Wk, d->d_model, d->kv_heads * d->head_dim, Q8_BLOCK_SIZE);
    mat_init_q8(attn.Wv, d->d_model, d->kv_heads * d->head_dim, Q8_BLOCK_SIZE);
    mat_init_q8(attn.Wo, d->heads * d->head_dim, d->d_model, Q8_BLOCK_SIZE);

    return attn;
}

float* v_embed_new(unsigned vocab_size, unsigned embed_dim) {
    float* E = mat_new(vocab_size, embed_dim, TYPE_F32);
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
