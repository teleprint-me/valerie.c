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

// @ref https://deeplearningbook.org/contents/mlp.html#pf11
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

// https://understandinglinearalgebra.org/sec-matrices-lin-combs.html
void matmul(Tensor* y, Tensor* W, Tensor* x) {
    assert(y && W && x);
    assert(y->shape.id == SHAPE_VEC);
    assert(W->shape.id == SHAPE_MAT);
    assert(x->shape.id == SHAPE_VEC);

    size_t y_cols = y->shape.dims[0];  // out dim
    size_t W_rows = W->shape.dims[0];
    size_t W_cols = W->shape.dims[1];
    size_t x_cols = x->shape.dims[0];  // in dim
    size_t W_stride = type_size(W->id);
    assert(W_rows == y_cols);  // match out
    assert(W_cols == x_cols);  // match in
    assert(W_stride > 0);  // at least 1 byte

    // Convert input to float
    float* xf = calloc(x_cols, type_size(x->id));
    dequant_vec(xf, x->data, x_cols, x->id);

    // Temporary buffer for each row of W
    float* wf = malloc(W_cols * sizeof(float));
    float* yf = malloc(y_cols * sizeof(float));

    for (size_t r = 0; r < W_rows; r++) {
        // Compute source row pointer
        const void* wsrc = tensor_mat_row(W, r);
        dequant_vec(wf, wsrc, W_cols, W->id);

        // Compute dot product
        float sum = 0.0f;
        for (size_t c = 0; c < W_cols; c++) {
            sum += wf[c] * xf[c];
        }

        yf[r] = sum;
    }

    // Write result
    quant_vec(y->data, yf, y_cols, y->id);

    // Clean up
    free(wf);
    free(xf);
    free(yf);
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
    matmul(s->q, L->attn.Wq, s->xq_dmodel, d->proj_dim, d->d_model, dtype);
    matmul(s->k, L->attn.Wk, s->xq_dmodel, d->kv_dim, d->d_model, dtype);
    matmul(s->v, L->attn.Wv, s->xq_dmodel, d->kv_dim, d->d_model, dtype);

    // Apply rotary embeddings per head/group
    // @ref https://arxiv.org/pdf/2305.13245
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
    matmul(s->x_norm, L->attn.Wo, s->xq_dmodel, d->d_model, d->proj_dim, dtype);

    // Attention residual connection
    residual(s->x, s->x_norm, d->d_model);
}

// @ref https://deeplearningbook.org/contents/mlp.html#pf1
void v_forward_ffn(Valerie* v, Layer* L) {
    Dim* d = &v->dim;
    State* s = &v->state;
    TypeId dtype = v->dtype;

    // Normalize input
    rmsnorm(s->x_norm, L->rms_ffn, s->x, d->d_model);

    // Quantize normed input
    quant_vec(s->xq_dmodel, s->x_norm, d->d_model, dtype);

    // Up-projection (W1)
    matmul(s->mlp_in, L->ffn.W1, s->xq_dmodel, d->hidden, d->d_model, dtype);
    // Gating path (W2)
    matmul(s->mlp_gate, L->ffn.W3, s->xq_dmodel, d->hidden, d->d_model, dtype);

    // SwiGLU (SiLU activation)
    for (int i = 0; i < d->hidden; i++) {
        s->mlp_in[i] *= silu(s->mlp_gate[i]);
    }

    // Quanitze up-projection (W1)
    quant_vec(s->xq_hidden, s->mlp_in, d->hidden, dtype);

    // Down projection (W2)
    matmul(s->x_norm, L->ffn.W2, s->xq_hidden, d->d_model, d->hidden, dtype);

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
    memcpy(s->x, (float*) e->token.data + id * d->d_model, d->d_model * sizeof(float));

    // Iterate over model layers
    for (int l = 0; l < d->layers; l++) {
        Layer* L = &v->layers[l];
        v_forward_attn(v, L, pos);
        v_forward_ffn(v, L);
    }

    // Final layer normalization
    rmsnorm(s->x_norm, e->norm.data, s->x, d->d_model);

    // Output projection (is always F32)
    matmul(s->logits, e->output, s->x_norm, d->vocab_size, d->d_model, TYPE_F32);
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
