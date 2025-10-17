/**
 * @file examples/model/v.c
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @copyright Copyright © 2025 Austin Berrio
 * @ref https://github.com/adriancable/qwen3.c
 * @ref https://arxiv.org/abs/1207.0580
 * @ref https://arxiv.org/abs/1608.05859
 * @ref https://arxiv.org/pdf/2305.13245
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
#include "core/type.h"
#include "core/lehmer.h"

#include "tokenizer/model.h"

#include "model/activation.h"
#include "model/matrix.h"

/**
 * @struct Params
 * @brief User-configurable settings for initializing a model.
 *
 * @details
 * The Params object defines tunable hyperparameters such as model width,
 * depth, and context length. It is intended for CLI or programmatic interfaces
 * where users can override default values before computing derived dimensions.
 *
 * The Params struct is transient: it exists only long enough to populate
 * a Dim struct via `v_dim_new()`. The tokenizer determines the fixed vocabulary
 * size (`vocab_size`) prior to model initialization.
 */
typedef struct Params {
    int d_model;  // model width (hidden size)
    int heads;  // number of attention heads
    int kv_heads;  // number of key/value heads (for GQA/MQA)
    int hidden_mul;  // FFN hidden multiplier
    int layers;  // number of transformer blocks
    int seq_len;  // maximum context length
    int vocab_size;  // tokenizer map size
} Params;

/**
 * @struct Dim
 * @brief Compute internal model dimensions from user-specified parameters.
 *
 * @details
 * Converts user-configurable Params into a fully expanded Dim struct
 * used throughout the model. Performs internal consistency checks to
 * ensure that head dimensions and grouping ratios divide evenly.
 *
 * @usage:
 * ```
 * Params p = v_params_new(tokenizer->vocab_size);
 * Dim d = v_dim_new(p);
 * ```
 */
typedef struct Dim {
    int d_model;  // hidden size (width of the model)
    int hidden;  // feed-forward hidden dimension (≈ 4 * d_model)
    int layers;  // number of transformer blocks (depth)
    int heads;  // number of attention heads
    int head_dim;  // per-head dimension (d_model / heads)
    int proj_dim;  // projection dimension (heads * head_dim)
    int kv_dim;  // key/value dimension (kv_heads * head_dim)
    int kv_mul;  // multi-query ratio (heads / kv_heads)
    int kv_heads;  // number of key/value heads (multiquery attention)
    int vocab_size;  // vocabulary size
    int seq_len;  // maximum context length
} Dim;

/**
 * @struct Attention
 * Trainable model-level parameters.
 */
typedef struct Attention {
    void* Wq;  // (d_model, heads * head_dim)
    void* Wk;  // (d_model, kv_heads * head_dim)
    void* Wv;  // (d_model, kv_heads * head_dim)
    void* Wo;  // (heads * head_dim, d_model)
} Attention;

/**
 * @struct FeedForward
 * Trainable model-level parameters.
 */
typedef struct FeedForward {
    void* W1;  // (hidden, d_model)
    void* W2;  // (d_model, hidden)
    void* W3;  // (hidden, d_model)
} FeedForward;

/**
 * @struct Cache
 * Layer-wise key/value caches for autoregressive attention.
 */
typedef struct Cache {
    float* k;  // key buffer (seq_len, kv_dim)
    float* v;  // value buffer (seq_len, kv_dim)
} Cache;

/**
 * @struct Layer
 * Transformer Block.
 */
typedef struct Layer {
    Attention attn;  // multi-head self-attention
    FeedForward ffn;  // feed-forward network
    Cache cache;  // key/value cache
    float* rms_attn;  // (d_model,) RMSNorm weights
    float* rms_ffn;  // (d_model,) RMSNorm weights
} Layer;

/**
 * @struct Embedding
 * Trainable model-level parameters.
 */
typedef struct Embedding {
    float* token;  // token embeddings (vocab_size, d_model)
    float* output;  // tied output weights (vocab_size, d_model)
    float* norm;  // final norm weights (d_model,)
} Embedding;

/**
 * @struct Rotary
 * Precomputed, non-trainable rotary frequencies.
 */
typedef struct Rotary {
    float* cos;  // (seq_len, head_dim / 2)
    float* sin;  // (seq_len, head_dim / 2)
} Rotary;

/**
 * @struct State
 * @brief Transient forward-pass buffers (not trainable).
 *
 * All buffers are allocated once per model instantiation and reused across tokens.
 * The `k` and `v` pointers are transient views into each layer's cache.
 */
typedef struct State {
    // Core stream
    float* x;  // (d_model,) residual stream
    float* x_norm;  // (d_model,) normalized stream

    // Attention intermediates
    float* q;  // (proj_dim,)
    float* k;  // view into cache (kv_dim,)
    float* v;  // view into cache (kv_dim,)
    float* attn_scores;  // (heads, seq_len) attention weights
    float* attn_out;  // (d_model,) attention output accumulator

    // Feed-forward intermediates
    float* mlp_in;  // (hidden,)
    float* mlp_gate;  // (hidden,)

    // Output
    float* logits;  // (vocab_size,)

    // Quantization scratch
    void* xq_dmodel;  // Embedding column width (d_model,)
    void* xq_hidden;  // MLP hidden width (hidden,)
} State;

/**
 * @struct Transformer Model
 */
typedef struct Valerie {
    Tokenizer t;  // tokenizer reference
    Dim dim;  // model dimensions and hyperparameters
    Rotary rope;  // precomputed rotary frequencies (non-learned)
    Embedding embed;  // embedding and output weights
    State state;  // forward-pass working state
    Layer* layers;  // array of transformer layers
    TypeId dtype;
} Valerie;

/**
 * Model pipeline
 * @{
 */

/**
 * @brief Create default hyperparameters for the model.
 *
 * Defaults to micro configuration (~8–10M params)
 *
 * @param vocab_size Vocabulary size (determined by tokenizer)
 * @return Params initialized with default settings
 */
Params v_params_new(int vocab_size) {
    Params params = {0};

    params.d_model = 320;  // model width (hidden size)
    params.heads = 32;  // number of attention heads
    params.kv_heads = 4;  // number of key/value heads (for GQA/MQA)
    params.hidden_mul = 4;  // FFN hidden multiplier
    params.layers = 6;  // number of transformer blocks
    params.seq_len = 128;  // maximum context length
    params.vocab_size = vocab_size;

    return params;
}

/**
 * @brief Compute derived transformer dimensions from hyperparameters.
 *
 * @param params User-specified hyperparameters
 * @return Fully computed Dim struct
 */
Dim v_dim_new(Params params) {
    assert(params.d_model % params.heads == 0);
    assert(params.heads % params.kv_heads == 0);

    // pre-compute model dimensions
    const int head_dim = params.d_model / params.heads;
    const int kv_mul = params.heads / params.kv_heads;
    const int kv_dim = params.kv_heads * head_dim;
    const int proj_dim = params.heads * head_dim;
    const int hidden = params.hidden_mul * params.d_model;

    return (Dim) {
        .d_model = params.d_model,  // model width
        .hidden = hidden,  // FFN inner dimension
        .layers = params.layers,  // transformer depth
        .heads = params.heads,  // attention heads
        .head_dim = head_dim,  // per-head dimension
        .proj_dim = proj_dim,  // == d_model
        .kv_heads = params.kv_heads,  // number of key/value heads
        .kv_mul = kv_mul,  // Q-per-K/V ratio (for GQA/MQA)
        .kv_dim = kv_dim,  // total KV width
        .vocab_size = params.vocab_size,  // tokenizer vocabulary size
        .seq_len = params.seq_len,  // context length
    };
}

void v_dim_log(Dim dim) {
    LOG_INFO("d_model: %d", dim.d_model);
    LOG_INFO("hidden: %d", dim.hidden);
    LOG_INFO("layers: %d", dim.layers);
    LOG_INFO("heads: %d", dim.heads);
    LOG_INFO("head_dim: %d", dim.head_dim);
    LOG_INFO("proj_dim: %d", dim.proj_dim);
    LOG_INFO("kv_dim: %d", dim.kv_dim);
    LOG_INFO("kv_mul: %d", dim.kv_mul);
    LOG_INFO("kv_heads: %d", dim.kv_heads);
    LOG_INFO("vocab_size: %d", dim.vocab_size);
    LOG_INFO("seq_len: %d", dim.seq_len);
}

Attention v_attn_new(Dim* d, TypeId dtype) {
    Attention attn = {0};

    attn.Wq = mat_new(d->d_model, d->heads * d->head_dim, dtype);
    attn.Wk = mat_new(d->d_model, d->kv_heads * d->head_dim, dtype);
    attn.Wv = mat_new(d->d_model, d->kv_heads * d->head_dim, dtype);
    attn.Wo = mat_new(d->heads * d->head_dim, d->d_model, dtype);

    mat_xavier(attn.Wq, d->d_model, d->heads * d->head_dim, dtype);
    mat_xavier(attn.Wk, d->d_model, d->kv_heads * d->head_dim, dtype);
    mat_xavier(attn.Wv, d->d_model, d->kv_heads * d->head_dim, dtype);
    mat_xavier(attn.Wo, d->heads * d->head_dim, d->d_model, dtype);

    return attn;
}

void v_attn_free(Attention* attn, TypeId dtype) {
    if (attn) {
        mat_free(attn->Wq, dtype);
        mat_free(attn->Wk, dtype);
        mat_free(attn->Wv, dtype);
        mat_free(attn->Wo, dtype);
    }
}

FeedForward v_ffn_new(Dim* d, TypeId dtype) {
    FeedForward ffn = {0};

    ffn.W1 = mat_new(d->hidden, d->d_model, dtype);
    ffn.W2 = mat_new(d->d_model, d->hidden, dtype);
    ffn.W3 = mat_new(d->hidden, d->d_model, dtype);

    mat_xavier(ffn.W1, d->hidden, d->d_model, dtype);
    mat_xavier(ffn.W2, d->d_model, d->hidden, dtype);
    mat_xavier(ffn.W3, d->hidden, d->d_model, dtype);

    return ffn;
}

void v_ffn_free(FeedForward* ffn, TypeId dtype) {
    if (ffn) {
        mat_free(ffn->W1, dtype);
        mat_free(ffn->W2, dtype);
        mat_free(ffn->W3, dtype);
    }
}

Cache v_cache_new(Dim* d) {
    return (Cache) {
        .k = mat_new(d->seq_len, d->kv_dim, TYPE_F32),
        .v = mat_new(d->seq_len, d->kv_dim, TYPE_F32),
    };
}

void v_cache_free(Cache* cache) {
    if (cache) {
        free(cache->k);
        free(cache->v);
    }
}

Layer* v_layers_new(Dim* d, TypeId dtype) {
    assert(d && d->layers > 0 && dtype < TYPE_COUNT);

    Layer* layers = calloc(d->layers, sizeof(Layer));
    if (!layers) {
        return NULL;
    }

    for (int i = 0; i < d->layers; i++) {
        layers[i].attn = v_attn_new(d, dtype);
        layers[i].ffn = v_ffn_new(d, dtype);
        layers[i].cache = v_cache_new(d);

        // RMSNorm parameters (γ)
        layers[i].rms_attn = calloc(d->d_model, sizeof(float));
        layers[i].rms_ffn = calloc(d->d_model, sizeof(float));

        // Initialize to 1.0 (identity scaling)
        for (int j = 0; j < d->d_model; j++) {
            layers[i].rms_attn[j] = 1.0f;
            layers[i].rms_ffn[j] = 1.0f;
        }
    }

    return layers;
}

void v_layers_free(Layer* layers, int n, TypeId dtype) {
    if (layers) {
        for (int i = 0; i < n; i++) {
            v_attn_free(&layers[i].attn, dtype);
            v_ffn_free(&layers[i].ffn, dtype);
            v_cache_free(&layers[i].cache);
            free(layers[i].rms_attn);
            free(layers[i].rms_ffn);
        }
        free(layers);
    }
}

State v_state_new(Dim* d, TypeId dtype) {
    State s = {0};
    s.x = calloc(d->d_model, sizeof(float));
    s.x_norm = calloc(d->d_model, sizeof(float));
    s.q = calloc(d->d_model, sizeof(float));
    s.k = NULL;  // Alias for key cache
    s.v = NULL;  // Alias for value cache
    s.attn_scores = calloc(d->heads * d->seq_len, sizeof(float));
    s.attn_out = calloc(d->d_model, sizeof(float));
    s.mlp_in = calloc(d->hidden, sizeof(float));
    s.mlp_gate = calloc(d->hidden, sizeof(float));
    s.logits = calloc(d->vocab_size, sizeof(float));
    s.xq_dmodel = vec_new(d->d_model, dtype);
    s.xq_hidden = vec_new(d->hidden, dtype);
    return s;
}

void v_state_free(State* s, TypeId dtype) {
    if (s) {
        free(s->x);
        free(s->x_norm);
        free(s->q);
        // Do not free key alias
        // Do not free value alias
        free(s->attn_scores);
        free(s->attn_out);
        free(s->mlp_in);
        free(s->mlp_gate);
        free(s->logits);
        vec_free(s->xq_dmodel, dtype);
        vec_free(s->xq_hidden, dtype);
    }
}

Embedding v_embed_new(Dim* d) {
    Embedding embed = {0};

    embed.token = mat_new(d->vocab_size, d->d_model, TYPE_F32);
    mat_xavier(embed.token, d->vocab_size, d->d_model, TYPE_F32);

    // Tie input to output weights
    embed.output = embed.token;

    // Final RMSNorm (same shape as d_model)
    embed.norm = calloc(d->d_model, sizeof(float));
    for (int i = 0; i < d->d_model; i++) {
        embed.norm[i] = 1.0f;  // γ initialized to 1
    }

    return embed;
}

void v_embed_free(Embedding* embed) {
    if (embed) {
        mat_free(embed->token, TYPE_F32);
        // output points to token, no need to free again
        free(embed->norm);
    }
}

Rotary v_rotary_new(Dim* d) {
    Rotary rope = {0};

    float theta = 10000.0f;  // @todo temp hard-coded value

    int dim = d->head_dim;  // per-head dimension
    int rows = d->seq_len;
    int cols = dim / 2;

    // base frequencies
    float* freqs = malloc(cols * sizeof(float));
    for (int j = 0; j < cols; j++) {
        // freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)] / dim))
        freqs[j] = 1.0f / powf(theta, (float) j / (float) dim);
    }

    // outer product
    rope.cos = mat_new(rows, cols, TYPE_F32);
    rope.sin = mat_new(rows, cols, TYPE_F32);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float angle = (float) i * freqs[j];
            rope.cos[i * cols + j] = cosf(angle);
            rope.sin[i * cols + j] = sinf(angle);
        }
    }

    free(freqs);

    return rope;
}

void v_rotary_free(Rotary* rope) {
    if (rope) {
        mat_free(rope->cos, TYPE_F32);
        mat_free(rope->sin, TYPE_F32);
    }
}

Valerie v_model_new(Tokenizer t, Params p, TypeId dtype) {
    Valerie v = {0};

    v.t = t;
    v.dtype = dtype;

    v.dim = v_dim_new(p);
    v.rope = v_rotary_new(&v.dim);
    v.embed = v_embed_new(&v.dim);
    v.state = v_state_new(&v.dim, v.dtype);
    v.layers = v_layers_new(&v.dim, v.dtype);

    return v;
}

void v_model_free(Valerie* v) {
    if (v) {
        tokenizer_free(&v->t);
        v_rotary_free(&v->rope);
        v_embed_free(&v->embed);
        v_state_free(&v->state, v->dtype);
        v_layers_free(v->layers, v->dim.layers, v->dtype);
    }
}

/** @} */

/**
 * Model Blocks
 */

// @ref https://arxiv.org/abs/1910.07467
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

// @ref https://arxiv.org/abs/2104.09864
void rotary(float* x, int pos, unsigned head_dim, const float* cos, const float* sin) {
    unsigned half_dim = head_dim / 2;

    const float* cos_t = cos + pos * half_dim;
    const float* sin_t = sin + pos * half_dim;

    for (unsigned i = 0; i < half_dim; i++) {
        float c = cos_t[i];
        float s = sin_t[i];

        float real = x[i];
        float imag = x[i + half_dim];

        x[i] = real * c - imag * s;
        x[i + half_dim] = real * s + imag * c;
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

// @ref https://arxiv.org/abs/1512.03385
void residual(float* y, float* x, int n) {
    assert(n > 0);

    for (int i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

// @ref https://arxiv.org/abs/1706.03762
void v_forward_attn(Valerie* v, Layer* L, int pos) {
    Dim* d = &v->dim;
    State* s = &v->state;
    TypeId dtype = v->dtype;

    // Tie current KV cache slot to state buffer (seq_len, kv_dim)
    s->k = L->cache.k + pos * d->kv_dim;  // share cache owned ref with k
    s->v = L->cache.v + pos * d->kv_dim;  // share cache owned ref with v

    // Normalize input
    rmsnorm(s->x_norm, L->rms_attn, s->x, d->d_model);

    // Quantize normed input
    quant_vec(s->xq_dmodel, s->x_norm, d->d_model, dtype);

    // Compute Q, K, V projections
    mat_mul(s->q, L->attn.Wq, s->xq_dmodel, d->proj_dim, d->d_model, dtype);
    mat_mul(s->k, L->attn.Wk, s->xq_dmodel, d->kv_dim, d->d_model, dtype);
    mat_mul(s->v, L->attn.Wv, s->xq_dmodel, d->kv_dim, d->d_model, dtype);

    // Apply rotary embeddings per head/group
    for (int h = 0; h < d->heads; h++) {
        int group = h / d->kv_mul;
        float* qh = s->q + h * d->head_dim;
        float* kh = s->k + group * d->head_dim;
        rotary(qh, pos, d->head_dim, v->rope.cos, v->rope.sin);
        rotary(kh, pos, d->head_dim, v->rope.cos, v->rope.sin);
    }

    // Compute attention scores (Q * K^T / sqrt(d_k))
    for (int h = 0; h < d->heads; h++) {
        float* qh = s->q + h * d->head_dim;
        float* scores = s->attn_scores + h * d->seq_len;

        for (int t = 0; t <= pos; t++) {
            // each K_t per head group
            float* kt = L->cache.k + t * d->kv_dim + (h / d->kv_mul) * d->head_dim;

            float dot = 0.0f;
            for (int k = 0; k < d->head_dim; k++) {
                dot += qh[k] * kt[k];
            }

            scores[t] = dot / sqrtf((float) d->head_dim);
        }

        // Softmax attention scores
        softmax(scores, pos + 1);

        // Weighted sum of scores (context vector)
        float* out_h = s->attn_out + h * d->head_dim;
        memset(out_h, 0, d->head_dim * sizeof(float));

        for (int t = 0; t <= pos; t++) {
            float w = scores[t];
            float* vt = L->cache.v + t * d->kv_dim + (h / d->kv_mul) * d->head_dim;
            for (int k = 0; k < d->head_dim; k++) {
                out_h[k] += w * vt[k];
            }
        }
    }

    // Quantize attention output
    quant_vec(s->xq_dmodel, s->attn_out, d->d_model, dtype);

    // Project concatenated heads back to model dimension (Wo)
    mat_mul(s->x_norm, L->attn.Wo, s->xq_dmodel, d->d_model, d->proj_dim, dtype);

    // Attention residual connection
    residual(s->x, s->x_norm, d->d_model);
}

void v_forward_ffn(Valerie* v, Layer* L) {
    Dim* d = &v->dim;
    State* s = &v->state;
    TypeId dtype = v->dtype;

    // Normalize input
    rmsnorm(s->x_norm, L->rms_ffn, s->x, d->d_model);

    // Quantize normed input
    quant_vec(s->xq_dmodel, s->x_norm, d->d_model, dtype);

    // Up-projection (W1)
    mat_mul(s->mlp_in, L->ffn.W1, s->xq_dmodel, d->hidden, d->d_model, dtype);
    // Gating path (W2)
    mat_mul(s->mlp_gate, L->ffn.W3, s->xq_dmodel, d->hidden, d->d_model, dtype);

    // SwiGLU (SiLU activation)
    for (int i = 0; i < d->hidden; i++) {
        s->mlp_in[i] *= silu(s->mlp_gate[i]);
    }

    // Quanitze up-projection (W1)
    quant_vec(s->xq_hidden, s->mlp_in, d->hidden, dtype);

    // Down projection (W2)
    mat_mul(s->x_norm, L->ffn.W2, s->xq_hidden, d->d_model, d->hidden, dtype);

    // FFN residual connection
    residual(s->x, s->x_norm, d->d_model);
}

// Single-token forward pass (autoregressive)
// @param id  current token id
// @param pos current position (0..n)
// @returns updated logit stream
float* v_forward(Valerie* v, int id, int pos) {
    Dim* d = &v->dim;
    State* s = &v->state;
    Embedding* e = &v->embed;

    // Token embedding lookup
    memcpy(s->x, e->token + id * d->d_model, d->d_model * sizeof(float));

    // Iterate over model layers
    for (int l = 0; l < d->layers; l++) {
        Layer* L = &v->layers[l];
        v_forward_attn(v, L, pos);
        v_forward_ffn(v, L);
    }

    // Final layer normalization
    rmsnorm(s->x_norm, e->norm, s->x, d->d_model);

    // Output projection (is always F32)
    mat_mul(s->logits, e->output, s->x_norm, d->vocab_size, d->d_model, TYPE_F32);
    return s->logits;
}

/** @} */

int main(void) {
    lehmer_init(1337);

    Tokenizer t = tokenizer_load("models/tokenizer.model");
    Params p = v_params_new(t.vocab_size);
    Valerie v = v_model_new(t, p, TYPE_Q8);

    LOG_INFO("Model initialized.");
    v_dim_log(v.dim);

    int token_id = 0;
    int pos = 0;
    float* logits = v_forward(&v, token_id, pos);

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
