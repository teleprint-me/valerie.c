/**
 * @file valerie.c
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @copyright Copyright © 2025 Austin Berrio
 */

#include <assert.h>
#include <math.h>
#include "core/logger.h"
#include "model/valerie.h"

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

Attention v_attn_new(const Dim* d, TypeId dtype) {
    Attention attn = {0};

    attn.Wq = tensor_new(shape_mat(d->d_model, d->heads * d->head_dim), dtype);
    attn.Wk = tensor_new(shape_mat(d->d_model, d->kv_heads * d->head_dim), dtype);
    attn.Wv = tensor_new(shape_mat(d->d_model, d->kv_heads * d->head_dim), dtype);
    attn.Wo = tensor_new(shape_mat(d->heads * d->head_dim, d->d_model), dtype);

    tensor_xavier(&attn.Wq);
    tensor_xavier(&attn.Wk);
    tensor_xavier(&attn.Wv);
    tensor_xavier(&attn.Wo);

    return attn;
}

void v_attn_free(Attention* attn) {
    if (attn) {
        tensor_free(&attn->Wq);
        tensor_free(&attn->Wk);
        tensor_free(&attn->Wv);
        tensor_free(&attn->Wo);
    }
}

FeedForward v_ffn_new(const Dim* d, TypeId dtype) {
    FeedForward ffn = {0};

    ffn.W1 = tensor_new(shape_mat(d->hidden, d->d_model), dtype);
    ffn.W2 = tensor_new(shape_mat(d->d_model, d->hidden), dtype);
    ffn.W3 = tensor_new(shape_mat(d->hidden, d->d_model), dtype);

    tensor_xavier(&ffn.W1);
    tensor_xavier(&ffn.W2);
    tensor_xavier(&ffn.W3);

    return ffn;
}

void v_ffn_free(FeedForward* ffn) {
    if (ffn) {
        tensor_free(&ffn->W1);
        tensor_free(&ffn->W2);
        tensor_free(&ffn->W3);
    }
}

Cache v_cache_new(const Dim* d) {
    Cache cache = {0};
    cache.k = tensor_new(shape_mat(d->seq_len, d->kv_dim), TYPE_F32);
    cache.v = tensor_new(shape_mat(d->seq_len, d->kv_dim), TYPE_F32);
    return cache;
}

void v_cache_free(Cache* cache) {
    if (cache) {
        tensor_free(&cache->k);
        tensor_free(&cache->v);
    }
}

Layer* v_layers_new(const Dim* d, TypeId dtype) {
    assert(d && d->layers > 0 && dtype < TYPE_COUNT);

    Layer* layers = calloc(d->layers, sizeof(Layer));
    if (!layers) {
        LOG_ERROR("v_layers_new: failed to allocate %d layers", d->layers);
        return NULL;
    }

    for (int i = 0; i < d->layers; i++) {
        Layer* L = &layers[i];

        // Core trainable submodules
        L[i].attn = v_attn_new(d, dtype);
        L[i].ffn = v_ffn_new(d, dtype);
        L[i].cache = v_cache_new(d);

        // RMSNorm weights (γ parameters)
        L[i].rms_attn = tensor_new(shape_vec(d->d_model), TYPE_F32);
        L[i].rms_ffn = tensor_new(shape_vec(d->d_model), TYPE_F32);

        // Initialize RMSNorm weights to identity scaling (γ = 1.0)
        tensor_ones(&L[i].rms_attn);
        tensor_ones(&L[i].rms_ffn);
    }

    return layers;
}

void v_layers_free(Layer* layers, size_t n) {
    if (layers) {
        for (size_t i = 0; i < n; i++) {
            v_attn_free(&layers[i].attn);
            v_ffn_free(&layers[i].ffn);
            v_cache_free(&layers[i].cache);
            tensor_free(&layers[i].rms_attn);
            tensor_free(&layers[i].rms_ffn);
        }
        free(layers);
    }
}

Embedding v_embed_new(const Dim* d) {
    Embedding embed = {0};

    embed.token = tensor_new(shape_mat(d->vocab_size, d->d_model), TYPE_F32);
    tensor_xavier(&embed.token);

    // Tie input to output weights
    embed.output = embed.token;

    // Final RMSNorm (same shape as d_model)
    embed.norm = tensor_new(shape_vec(d->d_model), TYPE_F32);
    tensor_ones(&embed.norm);

    return embed;
}

void v_embed_free(Embedding* embed) {
    if (embed) {
        tensor_free(&embed->token);
        // output points to token, no need to free again
        tensor_free(&embed->norm);
    }
}

Rotary v_rotary_new(const Dim* d) {
    Rotary rope = {0};

    float theta = 10000.0f;  // @todo temp hard-coded value

    int rows = d->seq_len;
    int cols = d->head_dim / 2;

    // base frequencies
    float* freqs = malloc(cols * sizeof(float));
    for (int j = 0; j < cols; j++) {
        // freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)] / dim))
        freqs[j] = 1.0f / powf(theta, (float) j / (float) d->head_dim);
    }

    // outer product
    rope.cos = tensor_new(shape_mat(rows, cols), TYPE_F32);
    rope.sin = tensor_new(shape_mat(rows, cols), TYPE_F32);

    // type cast tensors to float
    float* cos = (float*) rope.cos.data;
    float* sin = (float*) rope.sin.data;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float angle = (float) i * freqs[j];
            cos[i * cols + j] = cosf(angle);
            sin[i * cols + j] = sinf(angle);
        }
    }

    free(freqs);
    return rope;
}

void v_rotary_free(Rotary* rope) {
    if (rope) {
        tensor_free(&rope->cos);
        tensor_free(&rope->sin);
    }
}

State v_state_new(const Dim* d, TypeId dtype) {
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
    s.xq_dmodel = tensor_new(shape_vec(d->d_model), dtype);
    s.xq_hidden = tensor_new(shape_vec(d->hidden), dtype);
    return s;
}

void v_state_free(State* s) {
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
        tensor_free(&s->xq_dmodel);
        tensor_free(&s->xq_hidden);
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
        v_state_free(&v->state);
        v_layers_free(v->layers, v->dim.layers);
    }
}
