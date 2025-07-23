/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/memory.c
 * @brief Utility functions for memory alignment, padding, and allocation.
 *
 * Provides helper functions to:
 * - Check power-of-two properties
 * - Determine alignment of addresses or sizes
 * - Calculate padding and aligned sizes
 * - Allocate aligned memory blocks with posix_memalign
 */

#include "memory.h"

#include <sys/sysinfo.h>
#include <unistd.h>
#include <string.h>

/**
 * @name RAM Utilities
 * @{
 */

size_t memory_ram_max(void) {
    int64_t pages = sysconf(_SC_PHYS_PAGES);
    int64_t page_size = sysconf(_SC_PAGE_SIZE);

    size_t max_ram;
    if (pages <= 0 || page_size <= 0) {
        max_ram = MEMORY_MAX_FALLBACK;
    } else {
        max_ram = (size_t) pages * (size_t) page_size;
    }

    // Never allow to allocate more than (fallback - reserve), but always at least 16 MiB.
    if (max_ram > MEMORY_MAX_RESERVE) {
        max_ram -= MEMORY_MAX_RESERVE;
    } else {
        max_ram = 16 * 1024 * 1024; // fallback minimum, 16 MiB
    }

    return max_ram;
}

size_t memory_ram_total(void) {
    struct sysinfo info;
    if (-1 == sysinfo(&info)) {
        return 0;
    }
    return info.totalram;
}

size_t memory_ram_free(void) {
    struct sysinfo info;
    if (-1 == sysinfo(&info)) {
        return 0;
    }
    return info.freeram;
}

/**
 * @}
 */

/**
 * @name Alignment Utilities
 * @{
 */

bool memory_is_power_of_two(uintptr_t value) {
    return (0 != value) && (0 == (value & (value - 1)));
}

uintptr_t memory_align_offset(uintptr_t value, uintptr_t alignment) {
    assert(memory_is_power_of_two(alignment));
    return value & (alignment - 1);
}

bool memory_is_aligned(uintptr_t value, uintptr_t alignment) {
    assert(memory_is_power_of_two(alignment));
    return 0 == memory_align_offset(value, alignment);
}

uintptr_t memory_align_up(uintptr_t value, uintptr_t alignment) {
    assert(memory_is_power_of_two(alignment));

    if (value > UINTPTR_MAX - alignment + 1) {
        return UINTPTR_MAX; // Overflow guard
    }

    return (value + alignment - 1) & ~(alignment - 1);
}

uintptr_t memory_align_down(uintptr_t value, uintptr_t alignment) {
    assert(memory_is_power_of_two(alignment));
    return value & ~(alignment - 1);
}

size_t memory_align_up_pagesize(size_t value) {
    int64_t byte_size = (int64_t) value;
    int64_t page_size = (int64_t) sysconf(_SC_PAGESIZE);
    if (0 != byte_size % page_size) {
        byte_size += page_size - (byte_size % page_size);
    }
    return (size_t) byte_size;
}

size_t memory_padding_needed(uintptr_t value, size_t alignment) {
    assert(memory_is_power_of_two(alignment));
    size_t offset = memory_align_offset(value, alignment);
    return (0 != offset) ? alignment - offset : 0;
}

uintptr_t memory_align_unit_count(uintptr_t value, uintptr_t size, uintptr_t alignment) {
    assert(size > 0);

    uintptr_t aligned_size = memory_align_up(value, alignment);
    if (UINTPTR_MAX == aligned_size) {
        return UINTPTR_MAX; // Overflow guard
    }

    return (aligned_size + size - 1) / size;
}

/** @} */

/**
 * @name Aligned Memory Allocation
 * @{
 */

void* memory_alloc(size_t size, size_t alignment) {
    if (0 == size || 0 == alignment) {
        return NULL;
    }

    if (SIZE_MAX - size < size) {
        return NULL; // overflow
    }

    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

    if (!memory_is_power_of_two(alignment)) {
        return NULL;
    }

    void* address = NULL;
    if (0 != posix_memalign(&address, alignment, size)) {
        return NULL;
    }

    return address;
}

void* memory_calloc(size_t n, size_t size, size_t alignment) {
    if (0 == n || 0 == size || 0 == alignment) {
        return NULL;
    }

    size_t total = n * size;
    if (SIZE_MAX - total < total) {
        return NULL; // overflow
    }

    void* address = memory_alloc(total, alignment);
    if (address) {
        return memset(address, 0, total);
    }

    return NULL;
}

void* memory_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    if (NULL == ptr) {
        return memory_alloc(new_size, alignment);
    }

    if (0 == new_size || 0 == alignment) {
        free(ptr);
        return NULL;
    }

    if (!memory_is_power_of_two(alignment)) {
        return NULL;
    }

    void* new_ptr = memory_alloc(new_size, alignment);
    if (NULL == new_ptr) {
        return NULL;
    }

    // Copy only the smaller of the old or new sizes
    size_t min_size = old_size < new_size ? old_size : new_size;
    memcpy(new_ptr, ptr, min_size);
    free(ptr);
    return new_ptr;
}

void memory_free(void* ptr) {
    if (NULL != ptr) {
        free(ptr);
    }
}

/** @} */
