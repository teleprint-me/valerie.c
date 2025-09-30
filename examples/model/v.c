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

#include "core/path.h"
#include "core/type.h"
#include "tokenizer/model.h"

typedef struct Dim {
    int d_model;  // width of the model (hidden size)
    int hidden;  // FFN hidden dimension (usually 4 * d_model)
    int layers;  // number of transformer blocks (depth)
    int heads;  // number of attention heads
    int kv_heads;  // number of key/value heads (multiquery attention)
    int vocab_size;  // vocabulary size
    int seq_len;  // maximum context length
    int head_dim;  // per-head dimension (d_model / heads)
} Dim;

typedef struct Attention {
    float* Wq;  // (d_model, n_heads * head_dim)
    float* Wk;  // (d_model, n_kv_heads * head_dim)
    float* Wv;  // (d_model, n_kv_heads * head_dim)
    float* Wo;  // (n_heads * head_dim, d_model)
    float* rms;  // (d_model,) RMSNorm params
} Attention;

typedef struct FeedForward {
    float* W1;  // (hidden, d_model)
    float* W2;  // (d_model, hidden)
    float* W3;  // (hidden, d_model)
    float* rms;  // (d_model,) RMSNorm params
} FeedForward;

typedef struct Block {
    Attention* attn;  // multi-headed self attention
    FeedForward* ffn;  // feed-forward network
    float* rms_attn;  // (d_model,) RMSNorm params
    float* rms_ffn;  // (d_model,) RMSNorm params
} Block;

typedef struct State {
    float* x;  // (d_model,)

    float* q;  // (d_model,)
    float* k;  // (d_model,)
    float* v;  // (d_model,)

    float* att;  // (heads, seq_len)
    float* logits;  // (vocab_size,)

    float* k_cache;  // (seq_len, d_model)
    float* v_cache;  // (seq_len, d_model)
} State;

typedef struct Valerie {
    /// @note Output weights are tied to embeddings
    /// @ref https://arxiv.org/abs/1608.05859
    float* embeddings;  // (vocab_size, d_model)
    float* rms;  // (d_model,)

    Block* blocks;
    State* state;
    Tokenizer* t;
    Dim dim;
} Valerie;

int main(void) {
    return 0;
}
