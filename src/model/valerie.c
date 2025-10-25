/**
 * @file valerie.c
 */

#include <assert.h>
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
