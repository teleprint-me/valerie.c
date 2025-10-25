/**
 * @file blocks.h
 */

#ifndef MODEL_BLOCKS_H
#define MODEL_BLOCKS_H

#include <stdlib.h>

void one_hot(float* x, size_t label, size_t n);

// y_pred: predicted probabilities (softmax output), shape (n,)
// y_true: target one-hot vector, shape (n,)
// n: number of classes
float cross_entropy(const float* y_pred, const float* y_true, size_t n);

// @ref https://arxiv.org/abs/1910.07467
void rmsnorm(float* y, float* w, float* x, size_t n);

// @ref https://arxiv.org/abs/2104.09864
void rotary(float* x, int pos, size_t head_dim, const float* cos, const float* sin);

// @ref https://deeplearningbook.org/contents/mlp.html#pf11
void softmax(float* x, size_t n);

// @ref https://arxiv.org/abs/1512.03385
void residual(float* y, float* x, size_t n);

#endif  // MODEL_BLOCKS_H
