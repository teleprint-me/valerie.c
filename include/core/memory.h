/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/memory.h
 * @brief Utility functions for memory alignment, padding, and aligned allocation.
 *
 * Provides helper functions to:
 * - Check power-of-two properties
 * - Determine alignment of addresses or sizes
 * - Calculate padding and aligned sizes
 * - Allocate aligned memory blocks with posix_memalign
 *
 * This API explicitly disallows zero-size allocations and invalid alignments. All memory returned
 * is guaranteed to be aligned and non-NULL, or the function fails explicitly with NULL.
 */

#ifndef MEMORY_H
#define MEMORY_H

#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#include "core/posix.h"  // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * @brief Default maximum fallback memory size in bytes (4 GiB).
 */
#ifndef MEMORY_MAX_FALLBACK
    #define MEMORY_MAX_FALLBACK ((size_t) 1 << 32)
#endif

/**
 * @brief Default maximum reserve memory size in bytes (1 GiB).
 */
#ifndef MEMORY_MAX_RESERVE
    #define MEMORY_MAX_RESERVE ((size_t) 1 << 30)
#endif

/**
 * @name RAM Utilities
 * @{
 */

/**
 * @brief Returns the maximum allocatable RAM size in bytes.
 *
 * Takes into account system physical memory and reserves a fixed amount.
 *
 * @return Maximum allocatable RAM size in bytes.
 */
size_t memory_ram_max(void);

/**
 * @brief Returns the total physical RAM size in bytes.
 *
 * @return Total physical RAM size in bytes.
 */
size_t memory_ram_total(void);

/**
 * @brief Returns the amount of free RAM in bytes.
 *
 * @return Amount of free RAM in bytes.
 */
size_t memory_ram_free(void);

/** @} */

/**
 * @name Alignment Utilities
 * @{
 */

/**
 * @brief Checks if a value is a power of two and not zero.
 *
 * @param value Value to check.
 * @return true if value is a non-zero power of two, false otherwise.
 */
bool memory_is_power_of_two(uintptr_t value);

/**
 * @brief Returns the offset of a value within the given alignment boundary.
 *
 * For example, for alignment = 8 and value = 14, returns 6.
 *
 * @param value Value to check.
 * @param alignment Alignment boundary (must be a power of two).
 * @return Offset of value within alignment.
 */
uintptr_t memory_align_offset(uintptr_t value, uintptr_t alignment);

/**
 * @brief Checks if a value is aligned to the given alignment boundary.
 *
 * @param value Value to check.
 * @param alignment Alignment boundary (must be a power of two).
 * @return true if value is aligned, false otherwise.
 */
bool memory_is_aligned(uintptr_t value, uintptr_t alignment);

/**
 * @brief Aligns a value up to the next multiple of the given alignment.
 *
 * If already aligned, returns value unchanged.
 *
 * @param value Value to align.
 * @param alignment Alignment boundary (must be a power of two).
 * @return Aligned value rounded up.
 *
 * @warning All alignment and sizing utilities honor overflow propagation:
 * If `memory_align_up()` returns `UINTPTR_MAX`, downstream functions must treat it
 * as a fatal alignment error. This guards against undefined behavior in arithmetic-heavy
 * memory calculations.
 */
uintptr_t memory_align_up(uintptr_t value, uintptr_t alignment);

/**
 * @brief Aligns a value down to the previous multiple of the given alignment.
 *
 * If already aligned, returns value unchanged.
 *
 * @param value Value to align.
 * @param alignment Alignment boundary (must be a power of two).
 * @return Aligned value rounded down.
 */
uintptr_t memory_align_down(uintptr_t value, uintptr_t alignment);

/**
 * @brief Aligns a byte size up to the nearest system page size.
 *
 * @param value Byte size to align.
 * @return Byte size aligned up to page size.
 */
size_t memory_align_up_pagesize(size_t value);

/**
 * @brief Returns the number of padding bytes needed to align an address up to alignment.
 *
 * Returns zero if the address is already aligned.
 *
 * @param value Address to check.
 * @param alignment Alignment boundary (must be a power of two).
 * @return Number of padding bytes needed.
 */
size_t memory_padding_needed(uintptr_t value, size_t alignment);

/**
 * @brief Returns the minimal count of objects of size 'object_size' required
 *        to cover 'value' bytes, after rounding 'value' up to the given alignment.
 *
 * Useful to compute how many aligned units are needed to contain a buffer.
 *
 * @param value Required size in bytes.
 * @param size Size of each object in bytes.
 * @param alignment Alignment boundary (must be a power of two).
 * @return Minimal number of objects required.
 */
uintptr_t memory_align_unit_count(uintptr_t value, uintptr_t size, uintptr_t alignment);

/** @} */

/**
 * @name Aligned Memory Allocation
 * @{
 */

/**
 * @brief Allocates memory of the given size aligned to the specified boundary.
 *
 * Internally uses posix_memalign. The returned pointer must be freed with free().
 *
 * If @p size or @p alignment is zero, or if the allocation would overflow, the function returns
 * NULL. This avoids undefined or implementation-defined behavior from zero-byte allocations or
 * invalid alignment values.
 *
 * Additionally, checks if the alignment is a non-zero power of two.
 *
 * @param size Number of bytes to allocate.
 * @param alignment Alignment boundary (must be a non-zero power of two and >= sizeof(void*)).
 * @return Pointer to aligned memory on success, or NULL on failure or invalid input.
 */
void* memory_alloc(size_t size, size_t alignment);

/**
 * @brief Allocates zero-initialized memory for an array with specified alignment.
 *
 * Semantically equivalent to calloc, but guarantees the returned pointer is aligned.
 *
 * If any of @p n, @p size, or @p alignment is zero, or if the allocation would overflow, the
 * function returns NULL. Overflow in @p n * @p size is also guarded.
 *
 * @param n Number of elements.
 * @param size Size of each element in bytes.
 * @param alignment Alignment boundary (must be a non-zero power of two and >= sizeof(void*)).
 * @return Pointer to zeroed aligned memory on success, or NULL on failure or invalid input.
 */
void* memory_calloc(size_t n, size_t size, size_t alignment);

/**
 * @brief Reallocates an aligned memory block to a new size with alignment guarantee.
 *
 * - If @p ptr is NULL, behaves like memory_alloc(@p new_size, @p alignment).
 * - If @p new_size is zero, frees @p ptr and returns NULL.
 * - Copies the lesser of @p old_size and @p new_size bytes to the new block.
 *
 * The original @p ptr must have been allocated using memory_alloc or memory_calloc.
 * If @p alignment is not a valid power of two, or if any parameter is invalid,
 * the function returns NULL.
 *
 * @param ptr Pointer to previously allocated memory (or NULL).
 * @param old_size Size of the existing allocation in bytes.
 * @param new_size Desired size in bytes.
 * @param alignment Alignment boundary (must be a non-zero power of two and >= sizeof(void*)).
 * @return Pointer to newly allocated memory on success, or NULL on failure or invalid input.
 */
void* memory_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment);

/**
 * @brief Frees memory allocated by memory_alloc or memory_calloc.
 *
 * @param ptr Pointer to memory block to free.
 */
void memory_free(void* ptr);

/** @} */

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DSA_MEMORY_H
