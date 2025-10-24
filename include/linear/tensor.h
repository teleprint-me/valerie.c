/**
 * @file tensor.h
 * @brief Minimal tensor abstraction for 1D/2D numeric data.
 *
 * Provides heap-allocated vector/matrix storage with runtime-typed elements,
 * quantization-aware allocation, and utility functions for initialization,
 * filling, and zeroing. Intended as a composable building block for
 * higher-level neural model layers and blocks.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include "linear/type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ShapeId
 * @brief Identifier for tensor dimensionality (vector/matrix).
 */
typedef enum ShapeId {
    SHAPE_VEC = 1, /**< 1D tensor (vector): dims[0] elements */
    SHAPE_MAT = 2, /**< 2D tensor (matrix): dims[0] rows, dims[1] cols */
} ShapeId;

/**
 * @struct Shape
 * @brief Shape descriptor for 1D/2D tensors.
 *
 * Holds logical dimensions and shape type (vector or matrix).
 */
typedef struct Shape {
    size_t dims[2]; /**< Dimensions: [len, 0] for vector, [rows, cols] for matrix */
    ShapeId id; /**< SHAPE_VEC or SHAPE_MAT */
} Shape;

/**
 * @struct Tensor
 * @brief Heap-allocated tensor of arbitrary numeric type and shape.
 *
 * Data is owned and aligned to the type's requirements. Optional workspace
 * for temporary buffers (e.g., dequantization).
 */
typedef struct Tensor {
    void* data; /**< Pointer to storage buffer, type per @ref TypeId */
    float* workspace; /**< Optional temporary buffer for (de)quantization */
    Shape shape; /**< Shape descriptor (vector or matrix) */
    TypeId id; /**< Numeric type identifier (e.g., TYPE_F32, TYPE_Q8) */
} Tensor;

/**
 * @brief Compute number of elements for a given shape.
 * @param s Shape pointer
 * @return Number of elements (product of active dimensions)
 */
static inline size_t shape_count(const Shape* s);

/**
 * @brief Construct 1D vector shape.
 * @param len Number of elements
 * @return Shape struct for a vector of length @p len
 */
static inline Shape shape_vec(size_t len);

/**
 * @brief Construct 2D matrix shape.
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Shape struct for a matrix with @p rows and @p cols
 */
static inline Shape shape_mat(size_t rows, size_t cols);

/**
 * @brief Create a heap-allocated tensor of specified shape and type.
 *
 * Data buffer is allocated and zero-initialized. Workspace is NULL.
 *
 * @param shape Logical shape (dims/id must be set)
 * @param id    Numeric type identifier (@ref TypeId)
 * @return Tensor struct with allocated data; must call @ref tensor_free
 */
Tensor tensor_new(Shape shape, TypeId id);

/**
 * @brief Release all storage owned by a tensor.
 *
 * Frees data and workspace if allocated. Safe to call with NULL or zeroed tensor.
 * @param t Tensor pointer
 */
void tensor_free(Tensor* t);

/**
 * @brief Fill tensor with a constant value.
 *
 * For quantized types, fills each row/block with quantized @p value.
 * @param t Tensor pointer
 * @param value Scalar to fill
 */
void tensor_fill(Tensor* t, float value);

/**
 * @brief Fill tensor with all zeros.
 * @param t Tensor pointer
 */
void tensor_zeros(Tensor* t);

/**
 * @brief Fill tensor with all ones.
 * @param t Tensor pointer
 */
void tensor_ones(Tensor* t);

/**
 * @brief Fill tensor with pseudo-random values using Lehmer RNG.
 *
 * Each element is filled with a random float from the Lehmer generator.
 * @param t Tensor pointer
 */
void tensor_lehmer(Tensor* t);

/**
 * @brief Initialize tensor using Xavier/Glorot initialization.
 *
 * For weights: values are drawn from the Xavier uniform distribution.
 * @param t Tensor pointer
 */
void tensor_xavier(Tensor* t);

/**
 * @brief Initialize tensor using Muller (Gaussian) initialization.
 *
 * For weights: values are drawn from a Gaussian (normal) distribution.
 * @param t Tensor pointer
 */
void tensor_muller(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */
