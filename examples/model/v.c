/**
 * @file examples/model/v.c
 * @brief Valerie is a transformer model that mirrors the Qwen3 architecture.
 * @ref https://github.com/adriancable/qwen3.c
 * @note Valerie is not a replica of Qwen3 and is incompatible as a result.
 */

#include "core/path.h"
#include "core/type.h"
#include "tokenizer/model.h"

struct Dim {
    int dim;  // d_model (maybe rename for clarity?)
    int hidden;
    int layers;
    int heads;
    int kv_heads;
    int vocab_size;
    int seq_len;
    int head;
};

struct Weights {
    float* embeddings;

    float* Wq;
    float* Wk;
    float* Wv;
    float* Wo;

    float* W1;
    float* W2;
    float* W3;

    float* rms_att;
    float* rms_ffn;
    float* rms_final;
};

struct State {
    float* x;

    float* q;
    float* k;
    float* v;

    float* attn;
    float* logits;

    float* k_cache;
    float* v_cache;
};

int main(void) {
    return 0;
}
