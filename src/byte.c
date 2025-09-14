/**
 * Copyright © 2023 Austin Berrio
 *
 * @file utf8/src/byte.c
 * @brief UTF-8 byte-oriented string utilities.
 *
 * Low-level routines for working directly with bytes in null-terminated UTF-8 strings.
 * These routines operate purely on bytes—not codepoints or graphemes.
 *
 * All allocation routines return newly allocated buffers the caller must free.
 * All functions treat empty strings ("") as valid input.
 */

#include <stddef.h>
#include <string.h>

#include "regex.h"
#include "byte.h"

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

    uint8_t* dst = calloc((count + 1), sizeof(uint8_t));
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

    uint8_t* dst = calloc((n + 1), sizeof(uint8_t));
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
    uint8_t* out = calloc((out_n + 1), sizeof(uint8_t));
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

int8_t utf8_byte_cmp(const uint8_t* a, const uint8_t* b) {
    if (!a || !b) {
        return UTF8_COMPARE_INVALID;
    }

    const uint8_t* a_stream = a;
    const uint8_t* b_stream = b;

    while (*a_stream && *b_stream) {
        if (*a_stream < *b_stream) {
            return UTF8_COMPARE_LESS;
        }
        if (*a_stream > *b_stream) {
            return UTF8_COMPARE_GREATER;
        }
        // Both bytes are equal, move to the next
        a_stream++;
        b_stream++;
    }

    // Check if strings are of different lengths
    if (*a_stream) {
        return UTF8_COMPARE_GREATER;
    }
    if (*b_stream) {
        return UTF8_COMPARE_LESS;
    }

    return UTF8_COMPARE_EQUAL;
}

uint8_t** utf8_byte_append(const uint8_t* src, uint8_t** parts, uint64_t* count) {
    if (!src || !parts || !count) {
        return NULL;
    }

    size_t new_size = sizeof(uint8_t*) * (*count + 1);
    uint8_t** temp = realloc(parts, new_size);
    if (!temp) {
        return NULL;
    }

    parts = temp;
    parts[(*count)++] = (uint8_t*) src;
    return parts;
}

uint8_t** utf8_byte_append_n(
    const uint8_t* src, const uint64_t n, uint8_t** parts, uint64_t* count
) {
    if (!src || !parts || !count) {
        return NULL;
    }

    uint8_t* dst = utf8_byte_copy_n(src, n);
    if (!dst) {
        return NULL;
    }

    uint8_t** temp = utf8_byte_append(dst, parts, count);
    if (!temp) {
        free(dst);
        return NULL;
    }

    return temp;
}

uint8_t** utf8_byte_append_slice(
    const uint8_t* start, const uint8_t* end, uint8_t** parts, uint64_t* count
) {
    if (!start || !end || !parts || !count) {
        return NULL;
    }

    uint8_t* slice = utf8_byte_copy_slice(start, end);
    if (!slice) {
        return NULL;
    }

    uint8_t** temp = utf8_byte_append(slice, parts, count);
    if (!temp) {
        free(slice);
        return NULL;
    }

    return temp;
}

uint8_t** utf8_byte_split(const uint8_t* src, uint64_t* count) {
    if (!src || !count) {
        return NULL;
    }

    *count = 0;
    uint8_t** parts = calloc(1, sizeof(uint8_t*));
    int64_t len = utf8_byte_count(src);
    if (!parts || len < 0) {
        return NULL;
    }

    for (int64_t i = 0; i < len; i++) {
        uint8_t* chunk = calloc(2, sizeof(uint8_t));
        if (!chunk) {
            // Optionally: free previous parts
            return NULL;
        }
        chunk[0] = src[i];
        chunk[1] = '\0';

        parts = utf8_byte_append(chunk, parts, count);
        if (!parts) {
            free(chunk);
            // Optionally: free previous parts
            return NULL;
        }
    }

    return parts;
}

void utf8_byte_split_free(uint8_t** parts, uint64_t count) {
    if (parts) {
        for (uint64_t i = 0; i < count; i++) {
            free(parts[i]);
        }
        free(parts);
    }
}

uint8_t** utf8_byte_split_delim(const uint8_t* src, const uint8_t* delim, uint64_t* count) {
    if (!src || !count) {
        return NULL;
    }

    int64_t src_len = utf8_byte_count(src);
    if (src_len < 0) {
        return NULL;
    }

    // Empty delimiter means split into bytes
    int64_t delim_len = utf8_byte_count(delim);
    if (!delim || *delim == '\0' || delim_len < 1) {
        return utf8_byte_split(src, count);
    }

    *count = 0;
    uint8_t** parts = calloc(1, sizeof(uint8_t*));
    if (!parts) {
        return NULL;
    }

    const uint8_t* current = src;
    const uint8_t* scan = src;
    const uint8_t* end = src + src_len;

    while (scan <= end - delim_len) {
        if (memcmp(scan, delim, delim_len) == 0) {
            // Delimiter match: copy [current, scan)
            parts = utf8_byte_append_slice(current, scan, parts, count);
            if (!parts) {
                return NULL;
            }
            scan += delim_len;
            current = scan;
        } else {
            scan++;
        }
    }

    // Handle any trailing text after the last delimiter (or if no delimiter at all)
    if (current < end) {
        parts = utf8_byte_append_slice(current, end, parts, count);
        if (!parts) {
            return NULL;
        }
    }

    return parts;
}

uint8_t** utf8_byte_split_regex(const uint8_t* src, const uint8_t* pattern, uint64_t* count) {
    if (!src || !pattern || !count) {
        return NULL;
    }
    *count = 0;

    pcre2_code* code = NULL;
    pcre2_match_data* match = NULL;
    if (!utf8_regex_compile(pattern, &code, &match)) {
        return NULL;
    }

    uint8_t** parts = calloc(1, sizeof(uint8_t*));
    int64_t total_bytes = utf8_byte_count(src);
    if (!parts || total_bytes <= 0) {
        utf8_regex_free(code, match);
        return NULL;
    }

    int64_t offset = 0;
    while (offset < total_bytes) {
        int rc = pcre2_match(
            code, (PCRE2_SPTR) (src + offset), total_bytes - offset, 0, 0, match, NULL
        );
        if (rc < 0) {
            break;
        }

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match);
        size_t match_start = ovector[0];
        size_t match_end = ovector[1];

        if (match_end > match_start) {
            parts = utf8_byte_append_n(
                src + offset + match_start, match_end - match_start, parts, count
            );
            if (!parts) {
                utf8_regex_free(code, match);
                return NULL;
            }
        }
        offset += match_end;
    }

    utf8_regex_free(code, match);
    return parts;
}

uint8_t* utf8_byte_join(uint8_t** parts, uint64_t count, const uint8_t* delim) {
    if (!parts || count == 0) {
        return NULL;
    }

    int64_t delim_len = delim ? utf8_byte_count(delim) : 0;

    // Compute total length
    size_t total = 1;  // For final null terminator
    for (uint64_t i = 0; i < count; i++) {
        int64_t part_len = utf8_byte_count(parts[i]);
        if (part_len < 0) {
            return NULL;  // Defensive
        }
        total += (size_t) part_len;
    }
    if (delim_len > 0 && count > 1) {
        total += (size_t) delim_len * (count - 1);
    }

    // Allocate output buffer
    uint8_t* buffer = calloc(total, sizeof(uint8_t));
    if (!buffer) {
        return NULL;
    }

    // Copy parts and delimiters
    uint8_t* out = buffer;
    for (uint64_t i = 0; i < count; i++) {
        if (i > 0 && delim_len > 0) {
            memcpy(out, delim, delim_len);
            out += delim_len;
        }
        int64_t part_len = utf8_byte_count(parts[i]);
        if (part_len > 0) {
            memcpy(out, parts[i], part_len);
            out += part_len;
        }
    }
    *out = '\0';  // Null-terminate

    return buffer;
}
