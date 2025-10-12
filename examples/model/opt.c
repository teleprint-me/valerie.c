#include "core/type.h"

/**
 * @section Backward
 */

typedef struct AttentionOpt {
    // Derivatives
    void* dWk;
    void* dWq;
    void* dWv;
    void* dWo;

    // Velocities
    void* vWk;
    void* vWq;
    void* vWv;
    void* vWo;

    TypeId id;
} AttentionOpt;

typedef struct FeedForwardOpt {
    // Derivatives
    void* dW1;
    void* dW2;
    void* dW3;

    // Velocities
    void* vW1;
    void* vW2;
    void* vW3;

    TypeId id;
} FeedForwardOpt;

typedef struct LayerOpt {
    AttentionOpt attn;
    FeedForwardOpt ffn;
    float* drms_attn;  // (d_model,) RMSNorm params
    float* drms_ffn;  // (d_model,) RMSNorm params
} LayerOpt;

int main(void) {
    return 0;
}
