/**
 * @file examples/model/v.c
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @ref https://github.com/adriancable/qwen3.c
 * @note Valerie is not a replica of Qwen3 and is incompatible as a result.
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

typedef struct Block {
    float* embeddings;  // (vocab_size, d_model)

    float* Wq;  // (d_model, heads * head_dim)
    float* Wk;  // (d_model, kv_heads * head_dim)
    float* Wv;  // (d_model, kv_heads * head_dim)
    float* Wo;  // (heads * head_dim, d_model)

    float* W1;  // (hidden_dim, d_model)
    float* W2;  // (d_model, hidden_dim)
    float* W3;  // (hidden_dim, d_model)

    float* rms_att;  // (d_model,)
    float* rms_ffn;  // (d_model,)
    float* rms_final;  // (d_model,)
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
    Block* b;
    State* s;
    Tokenizer* t;
    Dim d;
} Valerie;

int main(void) {
    return 0;
}
