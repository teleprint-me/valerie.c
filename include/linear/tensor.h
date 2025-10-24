/**
 * @file tensor.h
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include "linear/type.h"

typedef enum ShapeId {
    SHAPE_VEC = 1,  // 1D (len, 0)
    SHAPE_MAT = 2,  // 2D (rows, cols)
} ShapeId;

typedef struct Shape {
    size_t dims[2];  // Up to 2D: [rows, cols] or [len, 0]
    ShapeId id;  // 0 for vector, 1 for matrix
} Shape;

typedef struct Tensor {
    void* data;  // Owned, type per TypeId
    float* workspace;  // Optional, for (de)quant, owned
    Shape shape;
    TypeId id;
} Tensor;

static inline size_t shape_count(const Shape* s);
static inline Shape shape_vec(size_t len);
static inline Shape shape_mat(size_t rows, size_t cols);

/**
 * @brief Create a new tensor with shape (1D or 2D) and type id.
 *        Allocates data and sets workspace to NULL.
 * @param shape Shape object (dims/n must be set)
 * @param id TypeId for storage (e.g. TYPE_F32, TYPE_Q8)
 * @return Tensor struct, .data is allocated, .workspace=NULL.
 */
Tensor tensor_new(Shape shape, TypeId id);

/**
 * @brief Free tensor data and workspace.
 */
void tensor_free(Tensor* t);

void tensor_fill(Tensor* t, float value);
void tensor_zeros(Tensor* t);
void tensor_ones(Tensor* t);

void tensor_lehmer(Tensor* t);
void tensor_xavier(Tensor* t);
void tensor_muller(Tensor* t);

#endif
