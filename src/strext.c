/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/strext.c
 * @brief A transitive wrapper to extend string.h operations.
 *
 * Low-level routines for working directly with bytes in null-terminated strings.
 * These routines operate purely on bytes—not codepoints or graphemes.
 *
 * All allocation routines return newly allocated buffers the caller must free.
 * All functions treat empty strings ("") as valid input.
 */

#include <stddef.h>
#include <ctype.h>

#include "regex.h"
#include "strext.h"

// Returns the byte offset from start to end. Returns -1 if inputs are NULL.
ptrdiff_t string_diff(const char* start, const char* end) {
    if (!start || !end) {
        return -1;
    }

    return (ptrdiff_t) end - (ptrdiff_t) start;
}

// Allocates a new null-terminated copy of the string.
char* string_copy(const char* start) {
    if (!start) {
        return NULL;
    }

    size_t len = strlen(start);
    char* dst = calloc((len + 1), sizeof(char));
    if (!dst) {
        return NULL;
    }

    if (len > 0) {
        memcpy(dst, start, len);
    }

    dst[len] = '\0';
    return dst;
}

// Copies exactly n bytes, as long as n <= count; always null-terminates.
char* string_copy_n(const char* start, size_t n) {
    if (!start) {
        return NULL;
    }

    size_t len = strlen(start);
    if (n > len) {
        return NULL;
    }

    char* dst = calloc((n + 1), sizeof(char));
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
char* string_copy_slice(const char* start, const char* end) {
    if (!start || !end) {
        return NULL;
    }

    ptrdiff_t diff = string_diff(start, end);
    if (diff < 0) {
        return NULL;
    }

    return string_copy_n(start, (size_t) diff);
}

char* string_concat(const char* dst, const char* src) {
    if (!dst || !src) {
        return NULL;
    }

    size_t dst_len = strlen(dst);
    size_t src_len = strlen(src);
    size_t out_len = dst_len + src_len;

    char* out = calloc((out_len + 1), sizeof(char));
    if (!out) {
        return NULL;
    }

    if (dst_len > 0) {
        memcpy(out, dst, dst_len);
    }
    if (src_len > 0) {
        memcpy(out + dst_len, src, src_len);
    }

    out[out_len] = '\0';
    return out;
}

int string_compare(const char* a, const char* b) {
    if (!a || !b) {
        return -2;  // invalid
    }

    const char* a_stream = a;
    const char* b_stream = b;

    while (*a_stream && *b_stream) {
        if (*a_stream < *b_stream) {
            return -1;  // less than
        }

        if (*a_stream > *b_stream) {
            return 1;  // greater than
        }

        // Both bytes are equal, move to the next
        a_stream++;
        b_stream++;
    }

    // Check if strings are of different lengths
    if (*a_stream) {
        return 1;  // greater than
    }

    if (*b_stream) {
        return -1;  // less than
    }

    return 0;  // equal
}

char** string_append(const char* src, char** parts, size_t* count) {
    if (!src || !parts || !count) {
        return NULL;
    }

    size_t new_size = sizeof(char*) * (*count + 1);
    char** temp = realloc(parts, new_size);
    if (!temp) {
        return NULL;
    }

    parts = temp;
    parts[(*count)++] = (char*) src;
    return parts;
}

char** string_append_n(const char* src, const size_t n, char** parts, size_t* count) {
    if (!src || !parts || !count) {
        return NULL;
    }

    char* dst = string_copy_n(src, n);
    if (!dst) {
        return NULL;
    }

    char** temp = string_append(dst, parts, count);
    if (!temp) {
        free(dst);
        return NULL;
    }

    return temp;
}

char** string_append_slice(const char* start, const char* end, char** parts, size_t* count) {
    if (!start || !end || !parts || !count) {
        return NULL;
    }

    char* slice = string_copy_slice(start, end);
    if (!slice) {
        return NULL;
    }

    char** temp = string_append(slice, parts, count);
    if (!temp) {
        free(slice);
        return NULL;
    }

    return temp;
}

char** string_split(const char* src, size_t* count) {
    if (!src || !count) {
        return NULL;
    }

    *count = 0;
    char** parts = calloc(1, sizeof(char*));
    if (!parts) {
        return NULL;
    }

    size_t len = strlen(src);
    for (size_t i = 0; i < len; i++) {
        char* chunk = calloc(2, sizeof(char));
        if (!chunk) {
            string_split_free(parts, *count);
            return NULL;
        }
        chunk[0] = src[i];
        chunk[1] = '\0';

        parts = string_append(chunk, parts, count);
        if (!parts) {
            free(chunk);
            string_split_free(parts, *count);
            return NULL;
        }
    }

    return parts;
}

void string_split_free(char** parts, size_t count) {
    if (parts) {
        for (size_t i = 0; i < count; i++) {
            free(parts[i]);
        }
        free(parts);
    }
}

char** string_split_space(const char* src, size_t* count) {
    if (!src || !count) {
        return NULL;
    }

    // Ensure parts is a valid pointer for the first realloc
    *count = 0;
    char** parts = calloc(1, sizeof(char*));
    if (!parts) {
        return NULL;
    }

    const char* p = src;
    while (*p) {
        // Skip leading whitespace
        while (*p && isspace((unsigned char) *p)) {
            p++;
        }

        if (!*p) {
            break;
        }

        // Mark token start
        const char* start = p;

        // Find end of token
        while (*p && !isspace((unsigned char) *p)) {
            p++;
        }

        // Copy token
        parts = string_append_slice(start, p, parts, count);
        if (!parts) {
            return NULL;
        }
    }

    return parts;
}

char** string_split_delim(const char* src, const char* delim, size_t* count) {
    if (!src || !count) {
        return NULL;
    }

    size_t src_len = strlen(src);
    size_t delim_len = strlen(delim);

    // Empty delimiter means split into bytes
    if (!delim || *delim == '\0' || delim_len < 1) {
        return string_split(src, count);
    }

    *count = 0;
    char** parts = calloc(1, sizeof(char*));
    if (!parts) {
        return NULL;
    }

    const char* current = src;
    const char* scan = src;
    const char* end = src + src_len;

    while (scan <= end - delim_len) {
        if (memcmp(scan, delim, delim_len) == 0) {
            // Delimiter match: copy [current, scan)
            parts = string_append_slice(current, scan, parts, count);
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
        parts = string_append_slice(current, end, parts, count);
        if (!parts) {
            return NULL;
        }
    }

    return parts;
}

char** string_split_regex(const char* src, const char* pattern, size_t* count) {
    if (!src || !pattern || !count) {
        return NULL;
    }
    *count = 0;

    pcre2_code* code = NULL;
    pcre2_match_data* match = NULL;
    if (!regex_compile(pattern, &code, &match)) {
        return NULL;
    }

    char** parts = calloc(1, sizeof(char*));
    size_t total_bytes = strlen(src);
    if (!parts || total_bytes <= 0) {
        regex_free(code, match);
        return NULL;
    }

    size_t offset = 0;
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
            parts = string_append_n(
                src + offset + match_start, match_end - match_start, parts, count
            );
            if (!parts) {
                regex_free(code, match);
                return NULL;
            }
        }
        offset += match_end;
    }

    regex_free(code, match);
    return parts;
}

char* string_join(char** parts, size_t count, const char* delim) {
    if (!parts || count == 0) {
        return NULL;
    }

    size_t delim_len = delim ? strlen(delim) : 0;

    // Compute total length
    size_t total = 1;  // For final null terminator
    for (size_t i = 0; i < count; i++) {
        size_t part_len = strlen(parts[i]);
        total += (size_t) part_len;
    }

    if (delim_len > 0 && count > 1) {
        total += (size_t) delim_len * (count - 1);
    }

    // Allocate output buffer
    char* buffer = calloc(total, sizeof(char));
    if (!buffer) {
        return NULL;
    }

    // Copy parts and delimiters
    char* out = buffer;
    for (size_t i = 0; i < count; i++) {
        if (i > 0 && delim_len > 0) {
            memcpy(out, delim, delim_len);
            out += delim_len;
        }

        size_t part_len = strlen(parts[i]);
        if (part_len > 0) {
            memcpy(out, parts[i], part_len);
            out += part_len;
        }
    }
    *out = '\0';  // Null-terminate

    return buffer;
}
