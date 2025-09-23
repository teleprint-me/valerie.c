/**
 * @file      page.h
 * @brief     Page-based memory allocator with tracked metadata.
 * @copyright Copyright © 2023 Austin Berrio
 *
 * Provides a simple allocator for manual memory management, built atop a linear-probing hash map.
 * Each allocation is tracked via size and alignment metadata, enabling:
 *   - Manual allocation and deallocation
 *   - Safe reallocation with metadata updates
 *   - Bulk deallocation of all tracked memory
 *
 * The allocator tracks each allocation as a (pointer, metadata) pair, supporting retroactive
 * addition of externally-allocated memory and robust cleanup.
 *
 * @note
 * - Uses the generic hash interface for metadata management; see hash.h for details.
 * - The hash API is thread-agnostic: **clients are responsible for locking via hash_lock()
 *   and hash_unlock()** if using in multithreaded contexts.
 * - Linear probing handles collisions internally (see hash.h).
 * - All memory tracked by this allocator must be freed only via this API to avoid leaks.
 *
 * Example use cases:
 *   - Automatic cleanup of multiple allocations in error handling paths.
 *   - Tracking buffer lifetimes in dynamic or plugin systems.
 *   - Simplifying retroactive adoption of manual memory tracking.
 */

#ifndef PAGE_ALLOCATOR_H
#define PAGE_ALLOCATOR_H

#include <stdbool.h>
#include <stddef.h>

#include "core/hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Alias for the internal hash map tracking page allocations.
 */
typedef struct Hash PageAllocator;

/**
 * @name Page Allocation API
 * @{
 */

/**
 * @brief Allocates a memory block with the given size and alignment.
 *
 * Tracks the allocation in the given PageAllocator.
 *
 * @param allocator Pointer to a valid PageAllocator.
 * @param size Number of bytes to allocate.
 * @param alignment Alignment requirement (must be power of two).
 * @return Pointer to allocated memory, or NULL on failure.
 */
void* page_malloc(PageAllocator* allocator, size_t size, size_t alignment);

/**
 * @brief Reallocates a previously allocated memory block.
 *
 * Updates tracking metadata and returns the new pointer.
 * - If @p ptr is NULL, behaves like page_malloc().
 * - If @p size is zero, frees the memory.
 *
 * @param allocator Pointer to a valid PageAllocator.
 * @param ptr Pointer to existing allocation (may be NULL).
 * @param size New size in bytes.
 * @param alignment Alignment requirement.
 * @return Pointer to reallocated memory, or NULL on failure.
 */
void* page_realloc(PageAllocator* allocator, void* ptr, size_t size, size_t alignment);

/**
 * @brief Frees a previously allocated memory block and removes its metadata.
 *
 * Logs an error if the pointer is not tracked.
 *
 * @param allocator Pointer to a valid PageAllocator.
 * @param ptr Pointer to memory block to free.
 */
void page_free(PageAllocator* allocator, void* ptr);

/**
 * @brief Frees all memory blocks tracked by the allocator.
 *
 * Clears the internal hash map but does not destroy the PageAllocator itself.
 *
 * @param allocator Pointer to a valid PageAllocator.
 *
 * @warning This is not thread-safe. Thread-locks must be implemented externally.
 *
 */
void page_free_all(PageAllocator* allocator);

/** @} */

/**
 * @name Allocator Utilities
 */

/**
 * Transfers ownership of a pre-allocated memory region to the PageAllocator.
 * Once added, the allocator is responsible for freeing the memory.
 *
 * The memory should be allocated using a compatible allocation mechanism
 * (e.g., malloc, strdup, or a known aligned allocator).
 *
 * The memory will be freed via `page_free()` or `page_free_all()`.
 *
 * This is useful for retroactive tracking of memory (e.g., UTF-8 strings,
 * externally allocated buffers).
 */
bool page_add(PageAllocator* allocator, void* ptr, size_t size, size_t alignment);

/** @} */

/**
 * @name Allocator Lifecycle
 * @{
 */

/**
 * @brief Creates a new PageAllocator with address-key tracking.
 *
 * @param capacity Initial capacity of the allocator's hash map.
 * @return Pointer to a new PageAllocator, or NULL on failure.
 */
PageAllocator* page_allocator_create(size_t capacity);

/**
 * @brief Frees all memory tracked by the allocator and destroys it.
 *
 * Equivalent to calling page_free_all() followed by hash map destruction.
 *
 * @warning This is not thread-safe.
 * @param allocator Pointer to the PageAllocator to destroy.
 *
 * @warning This is not thread-safe. Thread-locks must be implemented externally.
 */
void page_allocator_free(PageAllocator* allocator);

/** @} */

/**
 * @name Debug Utilities
 * @{
 */

/**
 * @brief Dumps the internal state of the PageAllocator for debugging.
 *
 * @param allocator Pointer to a valid PageAllocator.
 *
 * @warning This is not thread-safe. Thread-locks must be implemented externally.
 */
void page_allocator_dump(PageAllocator* allocator);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // PAGE_ALLOCATOR_H
