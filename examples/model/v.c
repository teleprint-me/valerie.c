/**
 * @file examples/model/v.c
 * @brief Minimal driver for Valerie's forward and backward passes.
 * @copyright Copyright © 2025 Austin Berrio
 *
 * Valerie is a dense, decoder-only transformer inspired by Llama, Mistral, Qwen, and GPT.
 * This file implements a from-scratch training loop with explicit forward and backward logic.
 *
 * Goals:
 *   - Demonstrate and test the full backward pass for model differentiation.
 *   - Identify pain points and refine Valerie's interface for real-world usage.
 */

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "core/logger.h"
#include "tokenizer/model.h"

/** psuedo-random number generation */

float random_normal(void) {
    return (float) rand() / (float) RAND_MAX;
}

float random_xavier(size_t rows, size_t cols) {
    float a = sqrtf(6.0f / (rows + cols));
    float ud = 2.0f * random_normal() - 1.0f;
    return ud * a;
}

/** tensors */

typedef enum Rank {
    RANK_SCALAR = 0, /**< 0D tensor (scalar): dims[0] is 1 */
    RANK_VECTOR = 1, /**< 1D tensor (vector): dims[0] elements */
    RANK_MATRIX = 2, /**< 2D tensor (matrix): dims[0] rows, dims[1] cols */
} Rank;

typedef struct Shape {
    size_t dims[2]; /**< Dimensions: scalar[1, 0], vector[len, 0], matrix[rows, cols] */
    Rank rank; /**< RANK_SCALAR, RANK_VECTOR, or RANK_MATRIX */
} Shape;

typedef struct Tensor {
    float* d;  // data (buffer)
    float* g;  // derivative (gradient)
    float* v;  // velocity (momentum)
    Shape shape;
} Tensor;

Shape shape_scalar(void) {
    return (Shape) {{1, 0}, RANK_SCALAR};
}

Shape shape_vector(size_t len) {
    return (Shape) {{len, 0}, RANK_VECTOR};
}

Shape shape_matrix(size_t rows, size_t cols) {
    return (Shape) {{rows, cols}, RANK_MATRIX};
}

bool tensor_is_null(const Tensor* t) {
    return !t || !t->d;
}

bool tensor_is_scalar(const Tensor* t) {
    return !tensor_is_null(t) && t->shape.rank == RANK_SCALAR;
}

bool tensor_is_vector(const Tensor* t) {
    return !tensor_is_null(t) && t->shape.rank == RANK_VECTOR;
}

bool tensor_is_matrix(const Tensor* t) {
    return !tensor_is_null(t) && t->shape.rank == RANK_MATRIX;
}

size_t tensor_cols(const Tensor* t) {
    switch (t->shape.rank) {
        case RANK_SCALAR:
        case RANK_VECTOR:
            return t->shape.dims[0];
        case RANK_MATRIX:
            return t->shape.dims[1];
        default:
            abort();
    }
}

size_t tensor_rows(const Tensor* t) {
    switch (t->shape.rank) {
        case RANK_SCALAR:
        case RANK_VECTOR:
            return 1;
        case RANK_MATRIX:
            return t->shape.dims[0];
        default:
            abort();
    }
}

size_t tensor_count(const Tensor* t) {
    switch (t->shape.rank) {
        case RANK_SCALAR:
        case RANK_VECTOR:
        case RANK_MATRIX:
            return tensor_rows(t) * tensor_cols(t);
        default:
            abort();
    }
}

bool tensor_cols_match(const Tensor* a, const Tensor* b) {
    return tensor_cols(a) == tensor_cols(b);
}

bool tensor_rows_match(const Tensor* a, const Tensor* b) {
    return tensor_rows(a) == tensor_rows(b);
}

Tensor tensor_null(Shape shape) {
    Tensor t = {0};
    t.d = NULL;
    t.g = NULL;
    t.v = NULL;
    t.shape = shape;
    return t;
}

Tensor tensor_new(Shape shape, bool use_grad) {
    Tensor t = {0};
    t.shape = shape;
    size_t len = tensor_count(&t);
    t.d = malloc(len * sizeof(float));
    if (use_grad) {
        t.g = calloc(len, sizeof(float));
        t.v = calloc(len, sizeof(float));
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        if (t->d) {
            free(t->d);
            t->d = NULL;
        }
        if (t->g) {
            free(t->g);
            t->g = NULL;
        }
        if (t->v) {
            free(t->v);
            t->v = NULL;
        }
    }
}

void tensor_fill(Tensor* t, float value) {
    for (size_t i = 0; i < tensor_count(t); i++) {
        t->d[i] = value;
    }
}

void tensor_zeros(Tensor* t) {
    tensor_fill(t, 0.0f);
}

void tensor_ones(Tensor* t) {
    tensor_fill(t, 1.0f);
}

void tensor_random(Tensor* t) {
    size_t rows = tensor_rows(t);
    size_t cols = tensor_cols(t);
    for (size_t i = 0; i < tensor_count(t); i++) {
        t->d[i] = (rows > 1 && cols > 1) ? random_xavier(rows, cols) : random_normal();
    }
}

void shape_print(const Shape* s) {
    switch (s->rank) {
        case RANK_SCALAR:
        case RANK_VECTOR:
            printf("(%zu)\n", s->dims[0]);
            break;
        case RANK_MATRIX:
            printf("(%zu, %zu)\n", s->dims[0], s->dims[1]);
            break;
    }
}

void tensor_print(const char* name, const Tensor* t) {
    printf("%s ", name);
    shape_print(&t->shape);

    size_t cols = tensor_cols(t);
    for (size_t r = 0; r < tensor_rows(t); ++r) {
        printf("[");
        float* row = t->d + r * cols;
        for (size_t c = 0; c < cols; ++c) {
            printf(" % .5f", (double) row[c]);
        }
        printf(" ]\n");
    }
}

/** Model */

// User-configurable settings for initializing a model.
typedef struct Param {
    int model;  // model width (hidden size)
    int heads;  // number of attention heads
    int kv_heads;  // number of key/value heads (for GQA/MQA)
    int hid_mul;  // FFN hidden multiplier
    int layers;  // number of transformer blocks
    int seq_len;  // maximum context length
    int vocab_size;  // tokenizer map size
    float theta;
} Param;

// Compute internal model dimensions from user-specified parameters.
typedef struct Dim {
    int model;  // hidden size (width of the model)
    int hidden;  // feed-forward hidden dimension (≈ 4 * d_model)
    int heads;  // number of attention heads
    int head_dim;  // per-head dimension (d_model / heads)
    int half_dim;  // number of dimensions per frequency (head_dim / 2)
    int proj_dim;  // projection dimension (heads * head_dim)
    int kv_dim;  // key/value dimension (kv_heads * head_dim)
    int kv_mul;  // multi-query ratio (heads / kv_heads)
    int kv_heads;  // number of key/value heads (multiquery attention)
    int layers;  // number of transformer blocks (depth)
    int seq_len;  // maximum context length
    int vocab_size;  // vocabulary size
    float theta;
} Dim;

// @note mat -> (rows, cols) -> (out, in)
typedef struct Attention {
    Tensor Wq;  // (proj_dim, d_model)
    Tensor Wk;  // (kv_dim, d_model)
    Tensor Wv;  // (kv_dim, d_model)
    Tensor Wo;  // (d_model, proj_dim)
    Tensor norm;  // (d_model,)
} Attention;

// @note mat -> (rows, cols) -> (out, in)
typedef struct FeedForward {
    Tensor W1;  // (hidden, d_model)
    Tensor W2;  // (d_model, hidden)
    Tensor W3;  // (hidden, d_model)
    Tensor norm;  // (d_model,) RMSNorm weights
} FeedForward;

// Layer-wise key/value caches for autoregressive attention.
typedef struct Cache {
    Tensor Wk;  // key buffer (seq_len, kv_dim)
    Tensor Wv;  // value buffer (seq_len, kv_dim)
} Cache;

// Transformer block
typedef struct Layer {
    Attention attn;  // multi-head self-attention
    FeedForward ffn;  // feed-forward network
    Cache cache;  // key/value cache
} Layer;

// Word to vector mapping
typedef struct Embedding {
    Tensor token;  // token embeddings (vocab_size, d_model)
    Tensor norm;  // final norm weights (d_model,)
} Embedding;

// Precomputed RoPE frequencies (not trainable).
typedef struct Rotary {
    Tensor cos;  // (seq_len, half_dim)
    Tensor sin;  // (seq_len, half_dim)
} Rotary;

// Transient forward-pass buffers (not trainable).
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

// Transformer model
typedef struct Valerie {
    Tokenizer t;  // tokenizer reference
    Dim d;  // model dimensions and hyperparameters
    Rotary r;  // pre-computed RoPE frequencies (non-learned)
    Embedding e;  // embedding and output weights
    State s;  // forward-pass working state
    Layer* l;  // array of transformer layers
} Valerie;

/** Model life-cycle */

// Create a micro-model (intentionally small)
Param param_new(int vocab_size) {
    Param param = {0};
    param.model = 256;  // model width (hidden size)
    param.heads = 16;  // number of attention heads
    param.kv_heads = 4;  // number of key/value heads (for GQA/MQA)
    param.hid_mul = 4;  // FFN hidden multiplier
    param.layers = 3;  // number of transformer blocks
    param.seq_len = 128;  // maximum context length
    param.vocab_size = vocab_size;
    param.theta = 10000.0f;
    return param;
}

Dim dim_new(Param param) {
    assert(param.model % param.heads == 0);
    assert(param.heads % param.kv_heads == 0);

    // pre-compute model dimensions
    const int head_dim = param.model / param.heads;
    const int half_dim = head_dim / 2;
    const int kv_mul = param.heads / param.kv_heads;
    const int kv_dim = param.kv_heads * head_dim;
    const int proj_dim = param.heads * head_dim;
    const int hidden = param.hid_mul * param.model;

    return (Dim) {
        .model = param.model,  // model width
        .hidden = hidden,  // FFN inner dimension
        .heads = param.heads,  // attention heads
        .head_dim = head_dim,  // per-head dimension
        .half_dim = half_dim,  // per-frequency dimension
        .proj_dim = proj_dim,  // == d_model
        .kv_heads = param.kv_heads,  // number of key/value heads
        .kv_mul = kv_mul,  // Q-per-K/V ratio (for GQA/MQA)
        .kv_dim = kv_dim,  // total KV width
        .layers = param.layers,  // transformer depth
        .seq_len = param.seq_len,  // context length
        .vocab_size = param.vocab_size,  // tokenizer vocabulary size
        .theta = param.theta,
    };
}

Rotary rotary_new(const Dim* d) {
    Rotary rope = {0};

    // base frequencies
    float* freqs = malloc(d->half_dim * sizeof(float));
    for (int j = 0; j < d->half_dim; j++) {
        // freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)] / dim))
        freqs[j] = 1.0f / powf(d->theta, (float) j / (float) d->head_dim);
    }

    // outer product
    rope.cos = tensor_new(shape_matrix(d->seq_len, d->half_dim), false);
    rope.sin = tensor_new(shape_matrix(d->seq_len, d->half_dim), false);

    // pre-compute frequencies
    for (int i = 0; i < d->seq_len; i++) {
        for (int j = 0; j < d->half_dim; j++) {
            float angle = (float) i * freqs[j];
            rope.cos.d[i * d->half_dim + j] = cosf(angle);
            rope.sin.d[i * d->half_dim + j] = sinf(angle);
        }
    }

    free(freqs);
    return rope;
}

void rotary_free(Rotary* rope) {
    if (rope) {
        tensor_free(&rope->cos);
        tensor_free(&rope->sin);
    }
}

Embedding embed_new(const Dim* d) {
    Embedding embed = {0};

    // Input is tied to output
    embed.token = tensor_new(shape_matrix(d->vocab_size, d->model), true);
    embed.norm = tensor_new(shape_vector(d->model), true);

    tensor_random(&embed.token);
    tensor_ones(&embed.norm);

    return embed;
}

void embed_free(Embedding* embed) {
    if (embed) {
        tensor_free(&embed->token);
        tensor_free(&embed->norm);
    }
}

// mat -> (rows, cols) -> (out, in)
Attention attn_new(const Dim* d) {
    Attention attn = {0};

    attn.Wq = tensor_new(shape_matrix(d->proj_dim, d->model), true);
    attn.Wk = tensor_new(shape_matrix(d->kv_dim, d->model), true);
    attn.Wv = tensor_new(shape_matrix(d->kv_dim, d->model), true);
    attn.Wo = tensor_new(shape_matrix(d->model, d->proj_dim), true);
    attn.norm = tensor_new(shape_vector(d->model), true);

    tensor_random(&attn.Wq);
    tensor_random(&attn.Wk);
    tensor_random(&attn.Wv);
    tensor_random(&attn.Wo);
    tensor_ones(&attn.norm);

    return attn;
}

void attn_free(Attention* attn) {
    if (attn) {
        tensor_free(&attn->Wq);
        tensor_free(&attn->Wk);
        tensor_free(&attn->Wv);
        tensor_free(&attn->Wo);
        tensor_free(&attn->norm);
    }
}

FeedForward ffn_new(const Dim* d) {
    FeedForward ffn = {0};

    ffn.W1 = tensor_new(shape_matrix(d->hidden, d->model), true);
    ffn.W2 = tensor_new(shape_matrix(d->model, d->hidden), true);
    ffn.W3 = tensor_new(shape_matrix(d->hidden, d->model), true);
    ffn.norm = tensor_new(shape_vector(d->model), true);

    tensor_random(&ffn.W1);
    tensor_random(&ffn.W2);
    tensor_random(&ffn.W3);
    tensor_ones(&ffn.norm);

    return ffn;
}

void ffn_free(FeedForward* ffn) {
    if (ffn) {
        tensor_free(&ffn->W1);
        tensor_free(&ffn->W2);
        tensor_free(&ffn->W3);
        tensor_free(&ffn->norm);
    }
}

Cache cache_new(const Dim* d) {
    Cache cache = {0};
    cache.Wk = tensor_new(shape_matrix(d->seq_len, d->kv_dim), true);
    cache.Wv = tensor_new(shape_matrix(d->seq_len, d->kv_dim), true);
    return cache;
}

void cache_free(Cache* cache) {
    if (cache) {
        tensor_free(&cache->Wk);
        tensor_free(&cache->Wv);
    }
}

Layer* layers_new(const Dim* d) {
    Layer* layers = calloc(d->layers, sizeof(Layer));

    for (int i = 0; i < d->layers; i++) {
        Layer* L = &layers[i];
        L->attn = attn_new(d);
        L->ffn = ffn_new(d);
        L->cache = cache_new(d);
    }

    return layers;
}

void layers_free(Layer* layers, size_t len) {
    if (layers) {
        for (size_t i = 0; i < len; i++) {
            Layer* L = &layers[i];
            attn_free(&L->attn);
            ffn_free(&L->ffn);
            cache_free(&L->cache);
        }
        free(layers);
        layers = NULL;
    }
}

State state_new(const Dim* d) {
    State s = {0};
    s.x = tensor_new(shape_vector(d->model), true);
    s.x_norm = tensor_new(shape_vector(d->model), true);
    s.q = tensor_new(shape_vector(d->proj_dim), true);
    s.k = tensor_null(shape_vector(d->kv_dim));  // Alias for key cache
    s.v = tensor_null(shape_vector(d->kv_dim));  // Alias for value cache
    s.attn_scores = tensor_new(shape_matrix(d->heads, d->seq_len), true);
    s.attn_out = tensor_new(shape_vector(d->model), true);
    s.mlp_in = tensor_new(shape_vector(d->hidden), true);
    s.mlp_gate = tensor_new(shape_vector(d->hidden), true);
    s.logits = tensor_new(shape_vector(d->vocab_size), true);
    return s;
}

void state_free(State* s) {
    if (s) {
        tensor_free(&s->x);
        tensor_free(&s->x_norm);
        tensor_free(&s->q);
        // Do not free key alias
        // Do not free value alias
        tensor_free(&s->attn_scores);
        tensor_free(&s->attn_out);
        tensor_free(&s->mlp_in);
        tensor_free(&s->mlp_gate);
        tensor_free(&s->logits);
    }
}

Valerie valerie_new(Tokenizer t, Param p) {
    Valerie v = {0};
    v.t = t;
    v.d = dim_new(p);
    v.r = rotary_new(&v.d);
    v.e = embed_new(&v.d);
    v.s = state_new(&v.d);
    v.l = layers_new(&v.d);
    return v;
}

void valerie_free(Valerie* v) {
    if (v) {
        tokenizer_free(&v->t);
        rotary_free(&v->r);
        embed_free(&v->e);
        state_free(&v->s);
        layers_free(v->l, v->d.layers);
    }
}

/** activations */

/**
 * @brief Logistic sigmoid function.
 *        σ(x) = 1 / (1 + exp(-x))
 * @see https://en.wikipedia.org/wiki/Sigmoid_function
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief SiLU (Swish-1) activation.
 *        silu(x) = x * sigmoid(x)
 * Forward pass: y = x * σ(x)
 * @see https://arxiv.org/abs/1710.05941
 */
void silu_forward(Tensor* y, const Tensor* x) {
    assert(tensor_is_vector(y));
    assert(tensor_is_vector(x));
    assert(tensor_cols_match(y, x));

    for (size_t i = 0; i < tensor_cols(y); i++) {
        y->d[i] = x->d[i] * sigmoid(x->d[i]);
    }
}

/**
 * @brief Derivative of SiLU activation.
 *        silu'(x) = σ(x) + x * σ(x) * (1 - σ(x))
 * Backward pass: out->g contains ∂L/∂y, accumulate ∂L/∂x
 */
void silu_backward(Tensor* y, const Tensor* x) {
    assert(tensor_is_vector(y));
    assert(tensor_is_vector(x));
    assert(tensor_cols_match(y, x));

    for (size_t i = 0; i < tensor_cols(y); i++) {
        float s = sigmoid(x->d[i]);
        float dx = s + x->d[i] * s * (1.0f - s);  // derivative
        x->g[i] += y->g[i] * dx;
    }
}

// Softmax: in-place and does not require a backward pass
// https://www.deeplearningbook.org/contents/mlp.html#pf11
void softmax_forward(float* x, size_t len) {
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

void softmax_backward(float* dx, float* y, size_t len) {
    float dot = 0.0f;
    for (size_t t = 0; t < len; t++) {
        dot += dx[t] * y[t];  // S = sum_j dx[j] * y[j]
    }

    for (size_t t = 0; t < len; t++) {
        dx[t] = (dx[t] - dot) * y[t];  // dx[i] = (dx[i] - S) * y[i]
    }
}

/**
 * root mean square normalization
 * https://arxiv.org/abs/1910.07467
 */

// forward
void rmsnorm_forward(Tensor* y, Tensor* w, Tensor* x) {
    assert(tensor_is_vector(y));
    assert(tensor_is_vector(w));
    assert(tensor_is_vector(x));
    assert(tensor_cols_match(y, w));
    assert(tensor_cols_match(w, x));

    size_t len = tensor_cols(y);

    // Compute sum of squares
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += x->d[i] * x->d[i];
    }

    // Normalize and scale
    float norm = sqrtf((sum / (float) len) + 1e-6f);
    float inv = 1.0f / norm;
    for (size_t i = 0; i < len; i++) {
        y->d[i] = w->d[i] * (x->d[i] * inv);
    }

    // Optionally store inv/norm for backward
    // (e.g., x->v[0] = inv;)
}

// backward
void rmsnorm_backward(Tensor* y, Tensor* w, Tensor* x) {
    assert(tensor_is_vector(y));
    assert(tensor_is_vector(w));
    assert(tensor_is_vector(x));
    assert(tensor_cols_match(y, w));
    assert(tensor_cols_match(w, x));

    size_t len = tensor_cols(y);

    // recompute norm
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += x->d[i] * x->d[i];
    }

    float norm = sqrtf((sum / (float) len) + 1e-6f);
    float inv = 1.0f / norm;
    float denom = len * norm * norm * norm;  // d * norm^3

    // precompute dot = Σ_j dy[j] * w[j] * x[j]
    float dot = 0.0f;
    for (size_t j = 0; j < len; j++) {
        dot += y->g[j] * w->d[j] * x->d[j];
    }

    // gradient accumulation
    for (size_t i = 0; i < len; i++) {
        // ∂L/∂x_i
        x->g[i] = (y->g[i] * w->d[i]) * inv - (x->d[i] * dot) / denom;
        // ∂L/∂w_i
        w->g[i] = y->g[i] * x->d[i] * inv;
    }
}

/**
 * row-major matrix multiplication
 * https://understandinglinearalgebra.org/sec-matrices-lin-combs.html
 */

// forward
void matmul_forward(Tensor* y, Tensor* W, Tensor* x) {
    assert(tensor_is_matrix(W));
    assert(tensor_is_vector(x));
    assert(tensor_is_vector(y));
    assert(tensor_cols(W) == tensor_cols(x));
    assert(tensor_rows(W) == tensor_cols(y));

    size_t rows = tensor_rows(W);
    size_t cols = tensor_cols(W);

    tensor_zeros(y);  // zero output

    // y_i = Σ_j W_ij * x_j
#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            sum += W->d[i * cols + j] * x->d[j];  // dot product
        }
        y->d[i] = sum;  // omit bias
    }
}

// backward
void matmul_backward(Tensor* y, Tensor* W, Tensor* x) {
    assert(tensor_is_matrix(W));
    assert(tensor_is_vector(x));
    assert(tensor_is_vector(y));
    assert(tensor_cols(W) == tensor_cols(x));
    assert(tensor_rows(W) == tensor_cols(y));

    size_t rows = tensor_rows(W);
    size_t cols = tensor_cols(W);

    // accumulate into W->g
    // dW: ∂L/∂W[i,j] = dy[i] * x[j]
#pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            W->g[i * cols + j] += y->g[i] * x->d[j];
        }
    }

    // accumulate into x->g
    // dx: ∂L/∂x[j] = sum_i (dy[i] * W[i,j])
#pragma omp parallel for
    for (size_t j = 0; j < cols; ++j) {
        float grad = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            grad += W->d[i * cols + j] * y->g[i];
        }
        x->g[j] += grad;  // accumulate
    }
}

/**
 * residual adder
 * https://arxiv.org/abs/1512.03385
 */

void residual_forward(Tensor* y, Tensor* x) {
    assert(tensor_is_vector(y));
    assert(tensor_is_vector(x));
    assert(tensor_cols_match(y, x));

    for (size_t i = 0; i < tensor_cols(y); i++) {
        y->d[i] += x->d[i];
    }
}

void residual_backward(Tensor* y, Tensor* x) {
    assert(tensor_is_vector(y));
    assert(tensor_is_vector(x));
    assert(tensor_cols_match(y, x));

    // // ∂L/∂x_i += ∂L/∂y_i
    for (size_t i = 0; i < tensor_cols(x); ++i) {
        x->g[i] += y->g[i];
    }
}

/** causal self-attention */

// Rotary position embedding
// https://arxiv.org/abs/2104.09864
void rotary_forward(float* x, Dim* d, Rotary* r, int pos) {
    // Column space is always half-dim
    assert(d->half_dim == d->head_dim / 2);

    const float* cos_t = r->cos.d + pos * d->half_dim;
    const float* sin_t = r->sin.d + pos * d->half_dim;

    for (int i = 0; i < d->half_dim; i++) {
        float c = cos_t[i];
        float s = sin_t[i];

        float real = x[i];
        float imag = x[i + d->half_dim];

        x[i] = real * c - imag * s;
        x[i + d->half_dim] = real * s + imag * c;
    }
}

void rotary_backward(float* x, Dim* d, Rotary* r, int pos) {
    const float* cos_t = r->cos.d + pos * d->half_dim;
    const float* sin_t = r->sin.d + pos * d->half_dim;

    for (int i = 0; i < d->half_dim; i++) {
        float c = cos_t[i];
        float s = sin_t[i];

        float real = x[i];
        float imag = x[i + d->half_dim];

        x[i] = real * c + imag * s;
        x[i + d->half_dim] = -real * s + imag * c;
    }
}

// Grouped query attention
// @ref https://arxiv.org/pdf/2305.13245
void gqa_forward(Tensor* q, Tensor* k, Dim* d, Rotary* r, int pos) {
#pragma omp parallel for
    for (int h = 0; h < d->heads; h++) {
        int group = h / d->kv_mul;
        float* qh = q->d + h * d->head_dim;
        float* kh = k->d + group * d->head_dim;
        rotary_forward(qh, d, r, pos);
        rotary_forward(kh, d, r, pos);
    }
}

void gqa_backward(Tensor* q, Tensor* k, Dim* d, Rotary* r, int pos) {
#pragma omp parallel for
    for (int h = 0; h < d->heads; h++) {
        int group = h / d->kv_mul;
        float* qh = q->g + h * d->head_dim;
        float* kh = k->g + group * d->head_dim;
        rotary_backward(qh, d, r, pos);
        rotary_backward(kh, d, r, pos);
    }
}

// Causal self attention
// https://arxiv.org/abs/1706.03762
void attn_forward(Valerie* v, Layer* L, int pos) {
    Dim* d = &v->d;
    Rotary* r = &v->r;
    State* s = &v->s;

    // Tie current KV cache slot to state buffer (seq_len, kv_dim)
    s->k.d = L->cache.Wk.d + pos * d->kv_dim;  // cache owned ref (kv_dim,)
    s->v.d = L->cache.Wv.d + pos * d->kv_dim;  // cache owned ref (kv_dim,)

    // Normalize input
    rmsnorm_forward(&s->x_norm, &L->attn.norm, &s->x);  // (d_model,)

    // Compute Q, K, V projections
    matmul_forward(&s->q, &L->attn.Wq, &s->x_norm);  // (proj_dim, d_model)
    matmul_forward(&s->k, &L->attn.Wk, &s->x_norm);  // (kv_dim, d_model)
    matmul_forward(&s->v, &L->attn.Wv, &s->x_norm);  // (kv_dim, d_model)

    // Grouped query attention
    gqa_forward(&s->q, &s->k, d, r, pos);

    // Compute attention scores (Q * K^T / sqrt(d_k))
#pragma omp parallel for
    for (int h = 0; h < d->heads; h++) {
        int group = h / d->kv_mul;
        int kv_group = group * d->head_dim;
        float* qh = s->q.d + h * d->head_dim;
        float* scores = s->attn_scores.d + h * d->seq_len;
        float* attn_out = s->attn_out.d + h * d->head_dim;

        // Dot product
        for (int t = 0; t <= pos; t++) {
            float* kt = L->cache.Wk.d + t * d->kv_dim + kv_group;

            float dot = 0.0f;
            for (int k = 0; k < d->head_dim; k++) {
                dot += qh[k] * kt[k];
            }

            scores[t] = dot / sqrtf((float) d->head_dim);
        }

        // Logarithmic probability
        softmax_forward(scores, pos + 1);

        // Context vector (weighted sum of scores)
        memset(attn_out, 0, d->head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float* vt = L->cache.Wv.d + t * d->kv_dim + kv_group;

            for (int k = 0; k < d->head_dim; k++) {
                attn_out[k] += scores[t] * vt[k];
            }
        }
    }

    // Project concatenated heads back to model dimension (Wo)
    matmul_forward(&s->x_norm, &L->attn.Wo, &s->attn_out);  // (d_model, proj_dim)

    // Attention residual connection
    residual_forward(&s->x, &s->x_norm);  // (d_model,)
}

// https://incml.github.io/2023/03/05/Transformer-GPT.html
void attn_backward(Valerie* v, Layer* L, int pos) {
    Dim* d = &v->d;
    Rotary* r = &v->r;
    State* s = &v->s;

    // Tie key cache slot (seq_len, kv_dim)
    s->k.d = L->cache.Wk.d + pos * d->kv_dim;  // cache owned ref (kv_dim,)
    s->k.g = L->cache.Wk.g + pos * d->kv_dim;  // cache owned ref (kv_dim,)

    // Tie value cache slot (seq_len, kv_dim)
    s->v.d = L->cache.Wv.d + pos * d->kv_dim;  // cache owned ref (kv_dim,)
    s->v.g = L->cache.Wv.g + pos * d->kv_dim;  // cache owned ref (kv_dim,)

    // Output residual
    residual_backward(&s->x, &s->x_norm);

    // Output projection
    matmul_backward(&s->x_norm, &L->attn.Wo, &s->attn_out);

#pragma omp parallel for
    for (int h = 0; h < d->heads; h++) {
        int group = h / d->kv_mul;
        int kv_group = group * d->head_dim;
        // float* attn_out = s->attn_out.d + h * d->head_dim;
        float* grad_attn_out = s->attn_out.g + h * d->head_dim;
        float* scores = s->attn_scores.d + h * d->seq_len;
        float* grad_scores = s->attn_scores.g + h * d->seq_len;
        float* qhd = s->q.d + h * d->head_dim;
        float* qhg = s->q.g + h * d->head_dim;

        // Context vector: attn_out.g (incoming gradient), scores, vt, and their .g buffers
        for (int t = 0; t <= pos; t++) {
            float sum = 0.0f;
            float* vtg = L->cache.Wv.g + t * d->kv_dim + kv_group;
            float* vtd = L->cache.Wv.d + t * d->kv_dim + kv_group;

            for (int k = 0; k < d->head_dim; k++) {
                vtg[k] += grad_attn_out[k] * scores[t];
                sum += grad_attn_out[k] * vtd[k];
            }

            grad_scores[t] += sum;  // accumulate gradient for scores[t]
        }

        // Softmax (in-place)
        softmax_backward(grad_scores, scores, pos + 1);

        // Dot product
        for (int t = 0; t <= pos; t++) {
            float* ktd = L->cache.Wk.d + t * d->kv_dim + kv_group;
            float* ktg = L->cache.Wk.g + t * d->kv_dim + kv_group;

            for (int k = 0; k < d->head_dim; k++) {
                qhg[k] += grad_scores[t] * ktd[k] / sqrtf((float) d->head_dim);
                ktg[k] += grad_scores[t] * qhd[k] / sqrtf((float) d->head_dim);
            }
        }
    }

    // Grouped query attention
    gqa_backward(&s->q, &s->k, d, r, pos);

    // Projections
    matmul_backward(&s->q, &L->attn.Wq, &s->x_norm);
    matmul_backward(&s->k, &L->attn.Wk, &s->x_norm);
    matmul_backward(&s->v, &L->attn.Wv, &s->x_norm);

    // Norm
    rmsnorm_backward(&s->x_norm, &L->attn.norm, &s->x);
}

/** feed-forward network */

// https://www.deeplearningbook.org/contents/mlp.html#pf1
void ffn_forward(Valerie* v, Layer* L) {
    State* s = &v->s;

    // Normalize input
    rmsnorm_forward(&s->x_norm, &L->ffn.norm, &s->x);
    // Up-projection (W1)
    matmul_forward(&s->mlp_in, &L->ffn.W1, &s->x_norm);
    // Gating path (W3)
    matmul_forward(&s->mlp_gate, &L->ffn.W3, &s->x_norm);
    // SwiGLU (SiLU activation)
    silu_forward(&s->mlp_in, &s->mlp_gate);
    // Down projection (W2)
    matmul_forward(&s->x_norm, &L->ffn.W2, &s->mlp_in);
    // FFN residual connection
    residual_forward(&s->x, &s->x_norm);
}

void ffn_backward(Valerie* v, Layer* L) {
    State* s = &v->s;

    // Residual: ∂L/∂x += ∂L/∂(x + x_norm)
    residual_backward(&s->x, &s->x_norm);
    // Down-projection: dW2, dmlp_in, accumulate x_norm.g
    matmul_backward(&s->x_norm, &L->ffn.W2, &s->mlp_in);
    // SwiGLU backward: propagate through SiLU gate
    silu_backward(&s->mlp_in, &s->mlp_gate);
    // Gating path: W3 backward
    matmul_backward(&s->mlp_gate, &L->ffn.W3, &s->x_norm);
    // Up-projection: W1 backward
    matmul_backward(&s->mlp_in, &L->ffn.W1, &s->x_norm);
    // RMSNorm backward: propagate into s->x and ffn.norm
    rmsnorm_backward(&s->x_norm, &L->ffn.norm, &s->x);
}

/** embedding */

// Token embedding lookup (vocab_size, d_model)
// output buffer = y (d_model,)
// lookup row = w (d_model,) = W + id * d_model
void embed_forward(Tensor* y, Tensor* W, int id) {
    assert(tensor_is_matrix(W));
    assert(tensor_is_vector(y));
    assert(tensor_cols_match(W, y));
    assert(id >= 0 && (size_t) id < tensor_rows(W));

    size_t len = tensor_cols(W);  // (d_model,)
    for (size_t i = 0; i < len; i++) {
        y->d[i] = W->d[id * len + i];
    }
}

// scatter gradient into input row
void embed_backward(Tensor* y, Tensor* W, int id) {
    assert(tensor_is_matrix(W));
    assert(tensor_is_vector(y));
    assert(tensor_cols_match(W, y));
    assert(id >= 0 && (size_t) id < tensor_rows(W));

    size_t len = tensor_cols(W);  // (d_model,)
    for (size_t i = 0; i < len; i++) {
        W->g[id * len + i] += y->g[i];  // ∂L/∂e
    }
}

/** forward pass */

// Single-token forward pass (autoregressive)
// @param id  current token id
// @param pos current position (0..seq_len)
// @returns updated logit stream
Tensor forward(Valerie* v, int id, int pos) {
    Dim* d = &v->d;
    State* s = &v->s;
    Embedding* e = &v->e;

    // Token embedding lookup
    embed_forward(&s->x, &e->token, id);

    // Iterate over model layers
    for (int i = 0; i < d->layers; i++) {
        attn_forward(v, &v->l[i], pos);
        ffn_forward(v, &v->l[i]);
    }

    // Final layer normalization
    rmsnorm_forward(&s->x_norm, &e->norm, &s->x);

    // Output projection (is always F32)
    matmul_forward(&s->logits, &e->token, &s->x_norm);

    // Output logarithmic probability
    return s->logits;
}

/** backward pass */

void backward(Valerie* v, int id, int pos) {
    Dim* d = &v->d;
    State* s = &v->s;
    Embedding* e = &v->e;

    // Backpropagate into final projection and norm
    matmul_backward(&s->logits, &e->token, &s->x_norm);
    rmsnorm_backward(&s->x_norm, &e->norm, &s->x);

    // Backpropagate through layers (reverse order)
    for (int i = d->layers - 1; i >= 0; --i) {
        ffn_backward(v, &v->l[i]);
        attn_backward(v, &v->l[i], pos);
    }

    // Scatter gradient back into source embedding
    embed_backward(&s->x, &e->token, id);
}

/** loss functions */

// https://en.wikipedia.org/wiki/One-hot
void one_hot(Tensor* x, size_t label) {
    assert(tensor_is_vector(x));
    assert(label < tensor_cols(x));

    size_t len = tensor_cols(x);
    for (size_t i = 0; i < len; i++) {
        if (label == i) {
            x->d[i] = 1.0f;
        } else {
            x->d[i] = 0.0f;
        }
    }
}

// https://en.wikipedia.org/wiki/Logistic_regression
float cross_entropy_forward(Tensor* y_pred, Tensor* y_true) {
    assert(tensor_is_vector(y_pred));
    assert(tensor_is_vector(y_true));
    assert(tensor_cols_match(y_pred, y_true));

    size_t len = tensor_cols(y_pred);
    for (size_t i = 0; i < len; i++) {
        if (y_true->d[i] == 1.0f) {
            return -logf(fmaxf(y_pred->d[i], 1e-6f));
        }
    }
    return 0.0f;  // fallback if not one-hot
}

// cross_entropy_backward: computes ∂L/∂y_pred (logits)
void cross_entropy_backward(Tensor* y_pred, const Tensor* y_true) {
    assert(tensor_is_vector(y_pred));
    assert(tensor_is_vector(y_true));
    assert(tensor_cols_match(y_pred, y_true));

    size_t len = tensor_cols(y_pred);
    for (size_t i = 0; i < len; i++) {
        // ∂L/∂z_i = p_i - y_i
        y_pred->g[i] = y_pred->d[i] - y_true->d[i];
    }
}

/** optimizer steps */

// just focus on stochastic gradient descent for now.
// adamw can be implemented later on.
void sgd(Tensor* t, float lr) {
    if (!t || !t->d || !t->g) {
        abort();
    }

    size_t len = tensor_count(t);
    for (size_t i = 0; i < len; i++) {
        // Sanity check
        assert(!isnan(t->g[i]) && "Gradient is NAN");
        assert(!isinf(t->g[i]) && "Gradient is INF");
        assert(t->g[i] > 1e-6f && "Gradient vanished");
        assert(t->g[i] < 1e+6f && "Gradient exploded");
        t->d[i] -= lr * t->g[i];  // update the weight
        t->g[i] = 0.0f;  // zero the gradient
    }
}

void update(Valerie* v, float lr) {
    for (int i = 0; i < v->d.layers; i++) {
        printf("Updating layer %d\n", i);
        Layer* L = &v->l[i];
        sgd(&L->attn.Wq, lr);
        sgd(&L->attn.Wk, lr);
        sgd(&L->attn.Wv, lr);
        sgd(&L->attn.Wo, lr);
        sgd(&L->attn.norm, lr);
        sgd(&L->ffn.W1, lr);
        sgd(&L->ffn.W2, lr);
        sgd(&L->ffn.W3, lr);
        sgd(&L->ffn.norm, lr);
    }
    sgd(&v->e.token, lr);
    sgd(&v->e.norm, lr);
}

// https://cs231n.github.io/linear-classify/#softmax
void train(Valerie* v, int* src_ids, int* tgt_ids, int src_len, int epochs, float lr) {
    Tensor target = tensor_new(shape_vector(v->d.vocab_size), false);
    for (int step = 0; step < epochs; step++) {
        for (int pos = 0; pos < src_len - 1; pos++) {
            Tensor logits = forward(v, src_ids[pos], pos);
            softmax_forward(logits.d, tensor_cols(&logits));  // compute probabilities
            one_hot(&target, tgt_ids[pos + 1]);  // next token prediction
            float loss = cross_entropy_forward(&logits, &target);  // compute current loss
            cross_entropy_backward(&logits, &target);  // initialize gradients
            printf("Loss: %.6f\n\n", (double) loss);
            backward(v, src_ids[pos], pos);
            update(v, lr);
        }
    }
    tensor_free(&target);
}

/** logging */

void log_tokens(Tokenizer* t, int* ids, int len) {
    printf("Token ids (%d):\n", len);
    for (int i = 0; i < len; i++) {
        printf("  [%4d] -> '%s'\n", ids[i], t->id_to_token[ids[i]]);
    }
    printf("\n");
}

void log_dim(Dim dim) {
    LOG_INFO("d_model: %d", dim.model);
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

// do a single pass until the model pipeline is operational and verified.
int main(void) {
    srand(73);  // the best number ever

    // hyperparameters
    float lr = 0.1f;  // learning rate

    // tokenizer model
    Tokenizer t = tokenizer_load("models/tokenizer.model");
    Param p = param_new(t.vocab_size);
    Valerie v = valerie_new(t, p);
    LOG_INFO("Model initialized.");
    log_dim(v.d);

    // source ids
    int src_len;
    // ["H", "e", "l", "lo", ",", " "]
    char src[] = "Hello, ";
    // [44, 87, 106, 110, 16, 4]
    int* src_ids = tokenizer_encode(&t, src, &src_len, false, false);
    log_tokens(&t, src_ids, src_len);

    // target ids
    int tgt_len;
    // ["H", "e", "l", "lo", ",", " ", "wor", "ld", "!"]
    char tgt[] = "Hello, world!";
    // [44, 87, 106, 110, 16, 4, 140, 107, 5]
    int* tgt_ids = tokenizer_encode(&t, tgt, &tgt_len, false, false);
    log_tokens(&t, tgt_ids, tgt_len);

    // do a simple forward pass for now
    int pos = 0;  // increment for each input token id
    int token_id = src_ids[0];  // V : 44 -> "H"
    Tensor logits = forward(&v, token_id, pos);  // compute log-odds
    tensor_print("Forward", &logits);

    // compute probabilities (operates in-place)
    softmax_forward(logits.d, tensor_cols(&logits));
    tensor_print("Softmax forward", &logits);

    // create next token prediction
    Tensor target = tensor_new(shape_vector(t.vocab_size), false);
    one_hot(&target, tgt_ids[pos + 1]);  // target class
    tensor_print("One Hot", &target);

    // compute model confidence
    float loss = cross_entropy_forward(&logits, &target);
    printf("Loss: %.6f\n\n", (double) loss);

    // initialize gradients
    cross_entropy_backward(&logits, &target);
    tensor_print("Softmax backward", &logits);

    // compute gradients
    softmax_backward(logits.g, logits.d, tensor_cols(&logits));
    backward(&v, token_id, pos);

    // update weights
    update(&v, lr);

    // clean up
    free(src_ids);
    free(tgt_ids);
    tensor_free(&target);
    valerie_free(&v);
    LOG_INFO("Model freed cleanly.");

    // exit
    return 0;
}
