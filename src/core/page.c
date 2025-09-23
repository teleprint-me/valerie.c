/**
 * @file      page.c
 * @brief     Page-based memory allocator with tracked metadata.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include "core/memory.h"
#include "core/logger.h"
#include "core/hash.h"
#include "core/map.h"
#include "core/page.h"

/**
 * @section Private
 * @{
 */

/**
 * @brief Internal page metadata for host allocations.
 *
 * `PageEntry` stores metadata associated with an allocation
 * managed by the `PageAllocator`. It is used to track memory size
 * and alignment for each allocation.
 *
 * This struct is stored in a hash map keyed by allocation address.
 */
typedef struct PageEntry {
    size_t size;
    size_t alignment;
} PageEntry;

/**
 * @brief Allocates and initializes a PageEntry.
 *
 * Allocates a `PageEntry` on the heap and fills in the provided metadata.
 * The returned pointer should be freed using `page_free()` when no longer needed.
 *
 * @param size Size of the allocation in bytes.
 * @param alignment Alignment in bytes.
 * @return Pointer to the initialized `PageEntry`, or NULL on failure.
 */
PageEntry* page_entry_create(size_t size, size_t alignment) {
    PageEntry* page = memory_alloc(sizeof(PageEntry), alignof(PageEntry));
    if (NULL == page) {
        return NULL;
    }

    *page = (PageEntry) {
        .size = size,
        .alignment = alignment,
    };

    return page;
}

/**
 * @brief Frees a previously allocated PageEntry.
 *
 * This function safely frees the metadata associated with a Vulkan allocation.
 * Passing NULL is safe and has no effect.
 *
 * @param page Pointer to the `PageEntry` to free.
 */
void page_entry_free(PageEntry* page) {
    if (NULL == page) {
        return;
    }
    memory_free(page);
}

/** @} */

/**
 * @section Page Allocation API
 * @{
 */

void* page_malloc(PageAllocator* allocator, size_t size, size_t alignment) {
    if (NULL == allocator) {
        LOG_ERROR("[PA_MALLOC] Missing allocation context (PageAllocator)");
        return NULL;
    }

    void* address = memory_alloc(size, alignment);
    if (NULL == address) {
        LOG_ERROR("[PA_MALLOC] Allocation failed (size=%zu, align=%zu)", size, alignment);
        return NULL;
    }

    PageEntry* page = page_entry_create(size, alignment);
    if (NULL == page) {
        memory_free(address);
        LOG_ERROR("[PA_MALLOC] Failed to allocate page metadata for %p", address);
        return NULL;
    }

    HashState state = hash_map_insert(allocator, address, page);
    if (HASH_FULL == state) {
        // Attempt to resize
        state = hash_map_resize(allocator, allocator->size * 2);
        if (HASH_SUCCESS != state) {
            memory_free(address);
            page_entry_free(page);
            LOG_ERROR("[PA_MALLOC] Failed to resize page allocator.");
            return NULL;
        }

        // Retry insertion
        state = hash_map_insert(allocator, address, page);
    }

    if (HASH_SUCCESS != state) {
        memory_free(address);
        page_entry_free(page);
        LOG_ERROR(
            "[PA_MALLOC] Failed to insert %p into page allocator (state = %d)", address, state
        );
        return NULL;
    }

    return address;
}

void* page_realloc(PageAllocator* allocator, void* ptr, size_t size, size_t alignment) {
    // Null context
    if (NULL == allocator) {
        LOG_ERROR("[PA_REALLOC] Missing allocation context (page allocator)");
        return NULL;
    }

    // Fresh allocation
    if (NULL == ptr) {
        return page_malloc(allocator, size, alignment);
    }

    // Lookup existing metadata
    PageEntry* page = hash_map_search(allocator, ptr);
    if (NULL == page) {
        LOG_ERROR("[PA_REALLOC] Unknown pointer %p", ptr);
        return NULL;
    }

    // Vulkan signals free via realloc with size == 0
    if (0 == size) {
        if (HASH_SUCCESS != hash_map_delete(allocator, ptr)) {
            LOG_ERROR("[PA_REALLOC] Failed to remove page for %p", ptr);
        }
        page_entry_free(page);
        memory_free(ptr);
        return NULL;
    }

    void* address = memory_realloc(ptr, page->size, size, alignment);
    if (NULL == address) {
        LOG_ERROR("[PA_REALLOC] Failed to realloc %p (%zu → %zu bytes)", ptr, page->size, size);
        return NULL;
    }

    // Update page metadata in-place
    *page = (PageEntry) {
        .size = size,
        .alignment = alignment,
    };

    // Re-allocator page metadata to new address
    if (HASH_SUCCESS != hash_map_delete(allocator, ptr)) {
        LOG_ERROR("[PA_REALLOC] Failed to remove old mapping for %p", ptr);
        return NULL;
    }

    HashState state = hash_map_insert(allocator, address, page);
    if (HASH_FULL == state) {
        // Attempt to resize
        state = hash_map_resize(allocator, allocator->size * 2);
        if (HASH_SUCCESS != state) {
            memory_free(address);
            page_entry_free(page);
            LOG_ERROR("[PA_REALLOC] Failed to resize page allocator.");
            return NULL;
        }

        // Retry insertion
        state = hash_map_insert(allocator, address, page);
    }

    if (HASH_SUCCESS != state) {
        memory_free(address);
        page_entry_free(page);
        LOG_ERROR(
            "[PA_REALLOC] Failed to insert %p into page allocator (state = %d)", address, state
        );
        return NULL;
    }

    return address;
}

void page_free(PageAllocator* allocator, void* ptr) {
    if (NULL == allocator || NULL == ptr) {
        return;
    }

    PageEntry* page = (PageEntry*) hash_map_search(allocator, ptr);
    if (NULL == page) {
        LOG_ERROR("[PA_FREE] Attempted to free untracked memory %p", ptr);
        return;
    }

    if (HASH_SUCCESS != hash_map_delete(allocator, ptr)) {
        LOG_ERROR("[PA_FREE] Failed to remove page for %p", ptr);
        return;
    }

    page_entry_free(page);
    memory_free(ptr);
}

void page_free_all(PageAllocator* allocator) {
    if (NULL == allocator) {
        return;
    }

    HashIt it = hash_iter(allocator);
    HashEntry* entry = NULL;

    while ((entry = hash_iter_next(&it))) {
        void* ptr = entry->key;
        PageEntry* page = (PageEntry*) entry->value;

        if (ptr && page) {
            page_entry_free(page);
            memory_free(ptr);
        }
    }

    hash_map_clear(allocator); // Clears internal map state
}

/** @} */

/**
 * @section Allocator Utilities
 * @{
 */

bool page_add(PageAllocator* allocator, void* ptr, size_t size, size_t alignment) {
    if (NULL == allocator) {
        LOG_ERROR("[PA_ADD] Missing allocation context (PageAllocator)");
        return false;
    }

    if (NULL == ptr) {
        LOG_ERROR("[PA_ADD] Cannot track NULL pointer");
        return false;
    }

    // Optional: guard against double tracking
    if (hash_map_search(allocator, ptr)) {
        LOG_WARN("[PA_ADD] Pointer %p is already tracked", ptr);
        return false;
    }

    PageEntry* page = page_entry_create(size, alignment);
    if (NULL == page) {
        LOG_ERROR("[PA_ADD] Failed to allocate page metadata for %p", ptr);
        return false;
    }

    HashState state = hash_map_insert(allocator, ptr, page);
    if (HASH_FULL == state) {
        state = hash_map_resize(allocator, allocator->size * 2);
        if (HASH_SUCCESS != state) {
            page_entry_free(page);
            LOG_ERROR("[PA_ADD] Failed to resize page allocator");
            return false;
        }

        // Retry insertion after resize
        state = hash_map_insert(allocator, ptr, page);
    }

    if (HASH_SUCCESS != state) {
        page_entry_free(page);
        LOG_ERROR("[PA_ADD] Failed to insert %p into page allocator (state = %d)", ptr, state);
        return false;
    }

    return true;
}

/** @} */

/**
 * @section Allocator Lifecycle
 * @{
 */

PageAllocator* page_allocator_create(size_t capacity) {
    return hash_map_create(capacity, HASH_PTR);
}

void page_allocator_free(PageAllocator* allocator) {
    if (NULL == allocator) {
        return;
    }
    page_free_all(allocator);
    hash_map_free(allocator);
}

/** @} */

/**
 * @section Debug Utilities
 * @{
 */

void page_allocator_dump(PageAllocator* allocator) {
    if (NULL == allocator) {
        return;
    }

    size_t total = 0;
    HashIt it = hash_iter(allocator);
    HashEntry* entry = NULL;

    while ((entry = hash_iter_next(&it))) {
        PageEntry* page = (PageEntry*) entry->value;
        void* ptr = entry->key;

        if (ptr && page) {
            total += page->size;
            LOG_INFO("[PA_DUMP] %p (%zu bytes, %zu aligned)", ptr, page->size, page->alignment);
        }
    }

    LOG_INFO("[PA_DUMP] Total memory still tracked: %zu bytes", total);
}

/** @} */
