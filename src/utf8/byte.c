/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/utf8/byte.c
 * @brief UTF-8 byte-oriented string utilities.
 *
 * Low-level routines for working directly with bytes in null-terminated UTF-8 strings.
 * These routines operate purely on bytes—not codepoints or graphemes.
 *
 * All allocation routines return newly allocated buffers the caller must free.
 * All functions treat empty strings ("") as valid input.
 */

#include "memory.h"
#include "utf8/byte.h"
#include <stddef.h>
#include <string.h>

// Returns the number of bytes before the null terminator.
int64_t utf8_byte_count(const uint8_t* start) {
    if (!start) {
        return -1;
    }

    int64_t count = 0;
    while (start[count]) {
        count++;
    }

    return count;
}

// Returns the byte offset from start to end. Returns -1 if inputs are NULL.
ptrdiff_t utf8_byte_diff(const uint8_t* start, const uint8_t* end) {
    if (!start || !end) {
        return -1;
    }

    return (ptrdiff_t) end - (ptrdiff_t) start;
}

// Allocates a new null-terminated copy of the string.
uint8_t* utf8_byte_copy(const uint8_t* start) {
    if (!start) {
        return NULL;
    }

    int64_t count = utf8_byte_count(start);
    if (count == -1) {
        return NULL;
    }

    uint8_t* dst = memory_alloc((count + 1) * sizeof(uint8_t), alignof(uint8_t));
    if (!dst) {
        return NULL;
    }

    if (count > 0) {
        memcpy(dst, start, count);
    }
    dst[count] = '\0';

    return dst;
}

// Copies exactly n bytes, as long as n <= count; always null-terminates.
uint8_t* utf8_byte_copy_n(const uint8_t* start, uint64_t n) {
    if (!start) {
        return NULL;
    }

    int64_t count = utf8_byte_count(start);
    if (count == -1 || n > (uint64_t) count) {
        return NULL;
    }

    uint8_t* dst = memory_alloc((n + 1) * sizeof(uint8_t), alignof(uint8_t));
    if (!dst) {
        return NULL;
    }

    if (n > 0) {
        memcpy(dst, start, n);
    }
    dst[n] = '\0';

    return dst;
}

// Copies bytes from start to end (exclusive), if end >= start.
uint8_t* utf8_byte_copy_slice(const uint8_t* start, const uint8_t* end) {
    if (!start || !end) {
        return NULL;
    }

    ptrdiff_t diff = utf8_byte_diff(start, end);
    if (diff < 0) {
        return NULL;
    }

    return utf8_byte_copy_n(start, (uint64_t) diff);
}

uint8_t* utf8_byte_cat(const uint8_t* dst, const uint8_t* src) {
    if (!dst || !src) {
        return NULL;
    }

    int64_t dst_n = utf8_byte_count(dst);
    int64_t src_n = utf8_byte_count(src);
    if (dst_n == -1 || src_n == -1) {
        return NULL;
    }

    size_t out_n = (size_t) (dst_n + src_n);
    uint8_t* out = memory_alloc((out_n + 1) * sizeof(uint8_t), alignof(uint8_t));
    if (!out) {
        return NULL;
    }

    if (dst_n > 0) {
        memcpy(out, dst, dst_n);
    }
    if (src_n > 0) {
        memcpy(out + dst_n, src, src_n);
    }
    out[out_n] = '\0';

    return out;
}
