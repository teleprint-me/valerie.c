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
#include "linear/quant.h"
#include "linear/tensor.h"
#include "linear/activation.h"
#include "tokenizer/model.h"
#include "model/valerie.h"

/**
 * Model Blocks
 */

/**
 * @brief Root-mean-square normalization (RMSNorm) for vectors.
 *
 * y = w * (x / sqrt(mean(x^2) + epsilon))
 *
 * @param y   Output vector (normalized and scaled)
 * @param w   Weight vector (per-feature scaling)
 * @param x   Input vector (features)
 * @note All tensors must be of type TYPE_F32 and 1D, with shape (d-model,).
 * @ref https://arxiv.org/abs/1910.07467
 */
void rmsnorm(Tensor* y, Tensor* w, Tensor* x) {
    // Assert valid tensors
    assert(y && w && x);
    // Assert tensors have single precision.
    assert(y->id == TYPE_F32);
    assert(w->id == TYPE_F32);
    assert(x->id == TYPE_F32);
    // Assert tensors are vectors
    assert(tensor_is_vec(y));
    assert(tensor_is_vec(w));
    assert(tensor_is_vec(x));
    // Assert shapes match
    assert(tensor_cols_match(y, w));
    assert(tensor_cols_match(w, x));  // chain matches
    // Extract vector length
    size_t len = tensor_cols(y);
    // Cast void* to float*
    float* yf = (float*) y->data;
    const float* wf = (float*) w->data;
    const float* xf = (float*) x->data;

    // Compute sum of squares
    float sos = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sos += xf[i] * xf[i];
    }

    // Compute scalar
    float scale = 1.0f / sqrtf((sos / (float) len) + 1e-6f);

    // Normalize and scale
    for (size_t i = 0; i < len; i++) {
        yf[i] = wf[i] * (xf[i] * scale);
    }
}

/**
 * @brief Matrix-vector multiply with quantization-aware dequantization.
 *
 * y = W @ x
 *
 * W: (rows, cols) matrix (any type)
 * x: (cols,) vector (any type)
 * y: (rows,) output vector (float only)
 * @ref https://understandinglinearalgebra.org/sec-matrices-lin-combs.html
 * @note The only way to sanely resolve the compute buffers is a graph.
 */
void matmul(Tensor* y, Tensor* W, Tensor* x) {
    // Assert valid tensors
    assert(y && W && x);
    // Assert output type (W and x may be any type)
    assert(y->id == TYPE_F32);  // output must be float
    // Assert shape types
    assert(tensor_is_vec(y));
    assert(tensor_is_mat(W));
    assert(tensor_is_vec(x));
    // Assert dims match y (r,) = W (r, c) @ x (c,)
    assert(tensor_cols_match(x, W));  // match input
    assert(tensor_cols_match_rows(y, W));  // match output
    // Extract input dimensions
    const size_t W_rows = tensor_rows(W);
    const size_t W_cols = tensor_cols(W);
    const size_t x_cols = tensor_cols(x);  // in dim

    // Alias output buffer
    float* yf = (float*) y->data;

    // Convert input to float
    float* xf = calloc(x_cols, sizeof(float));  // scratch buffer
    tensor_dequant_vec(xf, x, x_cols);

#pragma omp parallel for
    for (size_t r = 0; r < W_rows; r++) {
        // Compute source row pointer
        float* wdst = calloc(W_cols, sizeof(float));  // scratch buffer
        const void* wsrc = tensor_view_row(W, r);
        tensor_dequant_vec(wdst, wsrc, W_cols);

        // Compute dot product
        float sum = 0.0f;
        for (size_t c = 0; c < W_cols; c++) {
            sum += wdst[c] * xf[c];
        }

        yf[r] = sum;
        free(wdst);
    }

    free(xf);
}

/**
 * @brief Applies in-place rotary position embedding to a 2*half_dim vector.
 *
 * x: [real_0, ..., real_{half_dim-1}, imag_0, ..., imag_{half_dim-1}]
 * rope: pointer to rotary embedding tensors (cos/sin), always F32.
 * pos: position (row) to use in rope tensors.
 * len: total length of x (must be even, half_dim = len / 2).
 * @ref https://arxiv.org/abs/2104.09864
 */
void rotary(float* x, Rotary* rope, size_t pos, size_t len) {
    assert(x && rope);
    assert(len % 2 == 0);
    // Pre-computed rope frequencies
    const Tensor* cos = &rope->cos;
    const Tensor* sin = &rope->sin;
    // Rotary must always be TYPE_F32
    assert(cos->id == TYPE_F32);
    assert(sin->id == TYPE_F32);
    // Rotary must be shape (seq_len, head_dim / 2)
    assert(tensor_cols_match(cos, sin));
    assert(tensor_rows_match(cos, sin));
    // Column space is always half-dim
    size_t half_dim = tensor_cols(cos);
    assert(half_dim == len / 2);  // half_dim == head_dim / 2

    const float* cos_t = (float*) cos->data + pos * half_dim;
    const float* sin_t = (float*) sin->data + pos * half_dim;

    for (size_t i = 0; i < half_dim; i++) {
        float c = cos_t[i];
        float s = sin_t[i];

        float real = x[i];
        float imag = x[i + half_dim];

        x[i] = real * c - imag * s;
        x[i + half_dim] = real * s + imag * c;
    }
}

/**
 * @brief In-place numerically stable softmax on a float buffer.
 *
 * x: vector of length `len`. Output overwrites input.
 * @ref https://deeplearningbook.org/contents/mlp.html#pf11
 */
void softmax(float* x, size_t len) {
    float max_score = x[0];
    for (size_t i = 1; i < len; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        x[i] = expf(x[i] - max_score);
        sum += x[i];
    }

    for (size_t i = 0; i < len; i++) {
        x[i] /= sum;
    }
}

/**
 * @brief In-place residual connection: dst += src (elementwise).
 *
 * Both tensors must be 1D float vectors of size d-model.
 * @ref https://arxiv.org/abs/1512.03385
 */
void residual(Tensor* dst, Tensor* src) {
    assert(dst && src);
    assert(dst->id == TYPE_F32);
    assert(src->id == TYPE_F32);
    assert(tensor_is_vec(dst));
    assert(tensor_is_vec(src));
    assert(tensor_cols_match(dst, src));

    size_t len = tensor_cols(dst);
    float* yf = (float*) dst->data;
    const float* xf = (float*) src->data;

    for (size_t i = 0; i < len; i++) {
        yf[i] += xf[i];
    }
}

// @ref https://arxiv.org/abs/1706.03762
void v_forward_attn(Valerie* v, Layer* L, int pos) {
    Dim* d = &v->dim;
    State* s = &v->state;

    // Tie current KV cache slot to state buffer (seq_len, kv_dim)
    s->k.data = tensor_view(&L->cache.K, pos * d->kv_dim);  // cache owned ref (kv_dim,)
    s->v.data = tensor_view(&L->cache.V, pos * d->kv_dim);  // cache owned ref (kv_dim,)

    // Normalize input
    rmsnorm(&s->x_norm, &L->attn.norm, &s->x);

    // Compute Q, K, V projections
    matmul(&s->q, &L->attn.Wq, &s->x_norm);  // (proj_dim,)
    matmul(&s->k, &L->attn.Wk, &s->x_norm);  // (kv_dim,)
    matmul(&s->v, &L->attn.Wv, &s->x_norm);  // (kv_dim,)

    // Apply rotary embeddings per head/group
    // @ref https://arxiv.org/pdf/2305.13245
#pragma omp parallel for
    for (int h = 0; h < d->heads; h++) {
        int group = h / d->kv_mul;
        float* qh = tensor_view(&s->q, h * d->head_dim);
        float* kh = tensor_view(&s->k, group * d->head_dim);
        rotary(qh, &v->rope, pos, d->head_dim);
        rotary(kh, &v->rope, pos, d->head_dim);
    }

    // Compute attention scores (Q * K^T / sqrt(d_k))
#pragma omp parallel for
    for (int h = 0; h < d->heads; h++) {
        float* qh = tensor_view(&s->q, h * d->head_dim);
        float* scores = tensor_view(&s->attn_scores, h * d->seq_len);

        for (int t = 0; t <= pos; t++) {
            // each K_t per head group
            size_t offset = t * d->kv_dim + (h / d->kv_mul) * d->head_dim;
            float* kt = tensor_view(&L->cache.K, offset);

            float dot = 0.0f;
            for (int k = 0; k < d->head_dim; k++) {
                dot += qh[k] * kt[k];
            }

            scores[t] = dot / sqrtf((float) d->head_dim);
        }

        // Softmax attention scores
        softmax(scores, pos + 1);

        // Weighted sum of scores (context vector)
        float* out_h = tensor_view(&s->attn_out, h * d->head_dim);
        memset(out_h, 0, d->head_dim * sizeof(float));

        for (int t = 0; t <= pos; t++) {
            float w = scores[t];
            size_t offset = t * d->kv_dim + (h / d->kv_mul) * d->head_dim;
            float* vt = tensor_view(&L->cache.V, offset);
            for (int k = 0; k < d->head_dim; k++) {
                out_h[k] += w * vt[k];
            }
        }
    }

    // Project concatenated heads back to model dimension (Wo)
    matmul(&s->x_norm, &L->attn.Wo, &s->attn_out);

    // Attention residual connection
    residual(&s->x, &s->x_norm);
}

// @ref https://deeplearningbook.org/contents/mlp.html#pf1
void v_forward_ffn(Valerie* v, Layer* L) {
    Dim* d = &v->dim;
    State* s = &v->state;

    // Normalize input
    rmsnorm(&s->x_norm, &L->ffn.norm, &s->x);

    // Up-projection (W1)
    matmul(&s->mlp_in, &L->ffn.W1, &s->x_norm);
    // Gating path (W3)
    matmul(&s->mlp_gate, &L->ffn.W3, &s->x_norm);

    // SwiGLU (SiLU activation)
    float* mlp_in = (float*) s->mlp_in.data;
    float* mlp_gate = (float*) s->mlp_gate.data;
    for (int i = 0; i < d->hidden; i++) {
        mlp_in[i] *= silu(mlp_gate[i]);
    }

    // Down projection (W2)
    matmul(&s->x_norm, &L->ffn.W2, &s->mlp_in);

    // FFN residual connection
    residual(&s->x, &s->x_norm);
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
    float* dst = (float*) s->x.data;  // (d_model,)
    float* src = (float*) tensor_view_row(&e->token, id);  // (d_model,)
    memcpy(dst, src, d->d_model * sizeof(float));

    // Iterate over model layers
    for (int l = 0; l < d->layers; l++) {
        Layer* L = &v->layers[l];
        v_forward_attn(v, L, pos);
        v_forward_ffn(v, L);
    }

    // Final layer normalization
    rmsnorm(&s->x_norm, &e->norm, &s->x);

    // Output projection (is always F32)
    matmul(&s->logits, &e->token, &s->x_norm);
    return s->logits.data;
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
