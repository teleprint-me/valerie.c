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
    Q8* Wq;  // (d_model, n_heads * head_dim)
    Q8* Wk;  // (d_model, n_kv_heads * head_dim)
    Q8* Wv;  // (d_model, n_kv_heads * head_dim)
    Q8* Wo;  // (n_heads * head_dim, d_model)

    // Derivatives
    uint16_t* dWk;
    uint16_t* dWq;
    uint16_t* dWv;
    uint16_t* dWo;

    // Velocities
    uint16_t* vWk;
    uint16_t* vWq;
    uint16_t* vWv;
    uint16_t* vWo;

    float* rms;  // (d_model,) RMSNorm params
} Attention;

typedef struct FeedForward {
    Q8* W1;  // (hidden, d_model)
    Q8* W2;  // (d_model, hidden)
    Q8* W3;  // (hidden, d_model)

    uint16_t* dW1;
    uint16_t* dW2;
    uint16_t* dW3;

    uint16_t* vW1;
    uint16_t* vW2;
    uint16_t* vW3;

    float* rms;  // (d_model,) RMSNorm params
} FeedForward;

typedef struct Block {
    Attention attn;  // multi-headed self attention
    FeedForward ffn;  // feed-forward network
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

    Q8* qattn;
    Q8* qffn;
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

/**
 * @section Linear ops
 * @{
 */

void* vec_new(unsigned n, DataTypeId id) {
    void* vec = NULL;

    switch (id) {
        case TYPE_QUANT8:
            vec = calloc(n, sizeof(Q8));
            break;
        case TYPE_BFLOAT16:
            vec = calloc(n, sizeof(uint16_t));
            break;
        case TYPE_FLOAT32:
        default:
            vec = calloc(n, sizeof(float));
            break;
    }

    return vec;
}

void vec_xavier(float* x, unsigned n, unsigned out, unsigned in) {
#pragma omp parallel for
    for (unsigned i = 0; i < n; i++) {
        // xavier(fan_in, fan_out)
        x[i] = lehmer_xavier(in, out);  // thread is local
    }
}

// Create a row-major matrix
void* mat_new(unsigned out, unsigned in, DataTypeId id) {
    unsigned dim = out * in;
    void* mat = NULL;

    switch (id) {
        case TYPE_QUANT8:
            mat = calloc(dim, sizeof(Q8));
            break;
        case TYPE_BFLOAT16:
            mat = calloc(dim, sizeof(uint16_t));
            break;
        case TYPE_FLOAT32:
        default:
            mat = calloc(dim, sizeof(float));
            break;
    }

    return mat;
}

void mat_xavier(float* x, unsigned n, unsigned out, unsigned in) {
#pragma omp parallel for
    for (unsigned i = 0; i < n; i++) {
        // xavier(fan_in, fan_out)
        x[i] = lehmer_xavier(in, out);  // thread is local
    }
}

// Row-major matrix multiplication (y = Wx + b)
// bias is omitted because it's always 0
void mat_mul(float* y, Q8* W, Q8* x, unsigned out, unsigned in) {
#pragma omp parallel for
    for (unsigned i = 0; i < out; i++) {
        float sum = 0.0f;
        for (unsigned j = 0; j < in; j++) {
            sum += dequantize_scalar_q8(W[i * in + j]) * dequantize_scalar_q8(x[j]);
        }
        y[i] = sum;
    }
}

/** @} */

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

/**
 * Model ops
 * @{
 */

/// @todo Add precomputed rotary cache.
void rotary(float* x, int pos, unsigned head_dim) {
    unsigned half_dim = head_dim / 2;

    for (unsigned i = 0; i < half_dim; i++) {
        float angle = pos * powf(1e6f, -(float) i / half_dim);
        float cos_a = cosf(angle), sin_a = sinf(angle);

        float real = x[i];
        float imag = x[i + half_dim];

        x[i] = real * cos_a - imag * sin_a;
        x[i + half_dim] = real * sin_a + imag * cos_a;
    }
}

void rmsnorm(float* y, float* w, float* x, unsigned n) {
    // Avoid division by 0
    assert(n > 0 && "Division by zero!");

    // calculate sum of squares
    float sos = 0.0f;
    for (unsigned i = 0; i < n; i++) {
        sos += x[i] * x[i];
    }
    sos = 1.0f / sqrtf((sos / n) + 1e-6f);

    // normalize and scale
    for (unsigned i = 0; i < n; i++) {
        y[i] = w[i] * (sos * x[i]);
    }
}

// Applied to multi-head self-attention
void softmax(float* x, unsigned n) {
    float max_score = x[0];
    for (unsigned i = 1; i < n; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (unsigned i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_score);
        sum += x[i];
    }

    for (unsigned i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/** @} */

int main(void) {
    lehmer_init(1337);
    return 0;
}
