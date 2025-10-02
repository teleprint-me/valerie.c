#ifndef MODEL_BLOCKS
#define MODEL_BLOCKS

void one_hot(float* x, unsigned label, unsigned n);
float cross_entropy(const float* y_pred, const float* y_true, unsigned n);
void rmsnorm(float* y, float* w, float* x, unsigned n);
void rotary(float* x, int pos, unsigned head_dim);
void softmax(float* x, unsigned n);

#endif  // MODEL_BLOCKS
