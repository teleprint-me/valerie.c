/**
 * @file valerie.h
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @copyright Copyright © 2025 Austin Berrio
 * @ref https://github.com/adriancable/qwen3.c
 */

#ifndef VALERIE_H
#define VALERIE_H

#include "linear/type.h"
#include "linear/tensor.h"
#include "tokenizer/model.h"

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
 * @note norm must always be TYPE_F32.
 */
typedef struct Attention {
    Tensor Wq;  // ANY (d_model, heads * head_dim)
    Tensor Wk;  // ANY (d_model, kv_heads * head_dim)
    Tensor Wv;  // ANY (d_model, kv_heads * head_dim)
    Tensor Wo;  // ANY (heads * head_dim, d_model)
    Tensor norm;  // F32 (d_model,) RMSNorm weights
} Attention;

/**
 * @struct FeedForward
 * Trainable model-level parameters.
 * @note norm must always be TYPE_F32.
 */
typedef struct FeedForward {
    Tensor W1;  // ANY (hidden, d_model)
    Tensor W2;  // ANY (d_model, hidden)
    Tensor W3;  // ANY (hidden, d_model)
    Tensor norm;  // F32 (d_model,) RMSNorm weights
} FeedForward;

/**
 * @struct Cache
 * Layer-wise key/value caches for autoregressive attention.
 * Tensor must be TYPE_F32.
 */
typedef struct Cache {
    Tensor K;  // key buffer (seq_len, kv_dim)
    Tensor V;  // value buffer (seq_len, kv_dim)
} Cache;

/**
 * @struct Layer
 * Transformer Block.
 * Tensor must be TYPE_F32.
 */
typedef struct Layer {
    Attention attn;  // multi-head self-attention
    FeedForward ffn;  // feed-forward network
    Cache cache;  // key/value cache
} Layer;

/**
 * @struct Embedding
 * Trainable model-level parameters.
 * Tensor must be TYPE_F32.
 * @ref https://arxiv.org/abs/1608.05859
 */
typedef struct Embedding {
    Tensor token;  // token embeddings (vocab_size, d_model)
    Tensor norm;  // final norm weights (d_model,)
} Embedding;

/**
 * @struct Rotary
 * Precomputed, non-trainable rotary frequencies.
 * Tensor must be TYPE_F32.
 */
typedef struct Rotary {
    Tensor cos;  // (seq_len, head_dim / 2)
    Tensor sin;  // (seq_len, head_dim / 2)
} Rotary;

/**
 * @struct State
 * @brief Transient forward-pass buffers (not trainable).
 *
 * All buffers are allocated once per model instantiation and reused across tokens.
 * The `k` and `v` pointers are transient views into each layer's cache.
 *
 * Tensor must be TYPE_F32.
 */
typedef struct State {
    // Core stream
    Tensor x;  // (d_model,) residual stream
    Tensor x_norm;  // (d_model,) normalized stream

    // Attention intermediates
    Tensor q;  // (proj_dim,)
    Tensor k;  // view into cache (kv_dim,)
    Tensor v;  // view into cache (kv_dim,)
    Tensor attn_scores;  // (heads, seq_len) attention weights
    Tensor attn_out;  // (d_model,) attention output accumulator

    // Feed-forward intermediates
    Tensor mlp_in;  // (hidden,)
    Tensor mlp_gate;  // (hidden,)

    // Output
    Tensor logits;  // (vocab_size,)
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
 * @brief Create default hyperparameters for the model.
 *
 * Defaults to micro configuration (~8–10M params)
 *
 * @param vocab_size Vocabulary size (determined by tokenizer)
 * @return Params initialized with default settings
 */
Params v_params_new(int vocab_size);

/**
 * @brief Compute derived transformer dimensions from hyperparameters.
 *
 * @param params User-specified hyperparameters
 * @return Fully computed Dim struct
 */
Dim v_dim_new(Params params);

void v_dim_log(Dim dim);

Attention v_attn_new(const Dim* d, TypeId dtype);
void v_attn_free(Attention* attn);

FeedForward v_ffn_new(const Dim* d, TypeId dtype);
void v_ffn_free(FeedForward* ffn);

Cache v_cache_new(const Dim* d);
void v_cache_free(Cache* cache);

Layer* v_layers_new(const Dim* d, TypeId dtype);
void v_layers_free(Layer* layers, size_t n);

Embedding v_embed_new(const Dim* d);
void v_embed_free(Embedding* embed);

Rotary v_rotary_new(const Dim* d);
void v_rotary_free(Rotary* rope);

State v_state_new(const Dim* d);
void v_state_free(State* s);

Valerie v_model_new(Tokenizer t, Params p, TypeId dtype);
void v_model_free(Valerie* v);

#endif  // VALERIE_H
