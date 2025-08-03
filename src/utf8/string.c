/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/utf8/raw.c
 * @brief String API for handling UTF-8 pointer-to-char processing.
 */

#include "memory.h"
#include "logger.h"
#include "utf8/byte.h"
#include "utf8/iterator.h"
#include "utf8/string.h"

// --- UTF-8 Operations ---

bool utf8_is_valid(const char* start) {
    if (!start) {
        return false;
    }

    // Special case: empty string is valid
    if (*start == '\0') {
        return true;
    }

    UTF8IterValidator validator = {
        .is_valid = false,  // Invalid at start
        .error_at = NULL,  // Location unknown
    };

    utf8_iter_byte(start, utf8_iter_is_valid, &validator);
    if (!validator.is_valid && validator.error_at) {
        LOG_ERROR(
            "Invalid UTF-8 sequence detected at byte offset: %ld",
            validator.error_at - (const uint8_t*) start
        );
    }

    return validator.is_valid;
}

int64_t utf8_byte_len(const char* start) {
    if (!start || !utf8_is_valid(start)) {
        return -1;  // Invalid string
    }

    if ('\0' == *start) {
        return 0;  // Empty string
    }

    uint64_t byte_length = 0;
    const char* stream = start;
    while (*stream) {
        byte_length++;  // increment the byte length counter
        stream++;  // move to the next byte in the string
    }

    return byte_length;
}

char* utf8_copy(const char* start) {
    if (!start || !utf8_is_valid(start)) {
        return NULL;
    }

    size_t length = utf8_byte_len(start);
    if (0 == length) {
        char* output = malloc(1);
        if (!output) {
            return NULL;
        }
        output[0] = '\0';
        return output;
    }

    char* segment = malloc((length + 1) * sizeof(char));
    if (!segment) {
        return NULL;
    }

    memcpy(segment, start, length);
    segment[length] = '\0';

    return segment;
}

char* utf8_copy_n(const char* start, const uint64_t length) {
    if (!start) {
        return NULL;
    }

    if (0 == length) {
        char* output = malloc(1);
        if (!output) {
            return NULL;
        }
        output[0] = '\0';
        return output;
    }

    char* segment = malloc(length + 1);
    if (!segment) {
        return NULL;
    }

    memcpy(segment, start, length);
    segment[length] = '\0';

    return segment;
}

char* utf8_copy_range(const char* start, const char* end) {
    if (!start || !end) {
        return NULL;
    }

    ptrdiff_t length = utf8_byte_range((const uint8_t*) start, (const uint8_t*) end);
    if (-1 == length) {
        return NULL;
    }

    return utf8_copy_n(start, (const uint64_t) length);
}

char* utf8_concat(const char* dst, const char* src) {
    // Check for null pointers
    if (!dst || !src) {
        LOG_ERROR("Invalid dst or src parameter");
        return NULL;
    }

    // Validate the left and right operands, but allow empty strings
    if ('\0' != *dst && !utf8_is_valid(dst)) {
        LOG_ERROR("Invalid dst operand");
        return NULL;
    }

    if ('\0' != *src && !utf8_is_valid(src)) {
        LOG_ERROR("Invalid src operand");
        return NULL;
    }

    // Concatenate the right operand to the left operand
    size_t dst_length = utf8_byte_len(dst);
    size_t src_length = utf8_byte_len(src);
    size_t output_length = dst_length + src_length;

    // Add 1 for null terminator
    char* output = (char*) malloc(output_length + 1);
    if (output == NULL) {
        LOG_ERROR("Failed to allocate memory for concatenated string.");
        return NULL;
    }

    // Copy string bytes into output
    memcpy(output, dst, dst_length);
    memcpy(output + dst_length, src, src_length);
    output[output_length] = '\0';  // Null-terminate the string

    return output;
}

// --- UTF-8 Compare ---

int32_t utf8_compare(const char* a, const char* b) {
    if (!a || !b) {
        LOG_ERROR("One or both source strings are NULL.");
        return UTF8_COMPARE_INVALID;  // NULL strings are invalid inputs.
    }

    if (!utf8_is_valid(a)) {
        LOG_ERROR("First source string is not a valid UTF-8 string.");
        return UTF8_COMPARE_INVALID;  // Indicate invalid UTF-8 string.
    }

    if (!utf8_is_valid(b)) {
        LOG_ERROR("Second source string is not a valid UTF-8 string.");
        return UTF8_COMPARE_INVALID;  // Indicate invalid UTF-8 string.
    }

    const uint8_t* a_stream = (const uint8_t*) a;
    const uint8_t* b_stream = (const uint8_t*) b;

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

/** UTF-8 Split Tools */

char** utf8_split_push(const char* start, char** parts, uint64_t* capacity) {
    if (!start || !parts || !capacity) {
        return NULL;
    }

    char** temp = realloc(parts, sizeof(char*) * (*capacity + 1));
    if (!temp) {
        return NULL;
    }

    parts = temp;
    parts[(*capacity)++] = (char*) start;
    return parts;
}

char** utf8_split_push_n(
    const char* start, const uint64_t length, char** parts, uint64_t* capacity
) {
    if (!start || !parts || !capacity) {
        return NULL;
    }

    char* segment = utf8_copy_n(start, length);
    if (!segment) {
        return NULL;
    }

    char** temp = utf8_split_push(segment, parts, capacity);
    if (!temp) {
        free(segment);
        return NULL;
    }

    return temp;
}

char** utf8_split_push_range(const char* start, const char* end, char** parts, uint64_t* capacity) {
    if (!start || !end || !parts || !capacity) {
        return NULL;
    }

    char* segment = utf8_copy_range(start, end);
    if (!segment) {
        return NULL;
    }

    char** temp = utf8_split_push(segment, parts, capacity);
    if (!temp) {
        free(segment);
        return NULL;
    }

    return temp;
}

/** UTF-8 Split */

char** utf8_split(const char* src, uint64_t* capacity) {
    if (!src || !capacity) {
        return NULL;
    }

    *capacity = 0;
    char** parts = malloc(sizeof(char*));  // Start empty array
    const uint8_t* ptr = (const uint8_t*) src;

    while (*ptr) {
        int8_t width = utf8_byte_width(ptr);
        if (width <= 0) {
            break;
        }

        parts = utf8_split_push_n((const char*) ptr, width, parts, capacity);
        ptr += width;
    }

    return parts;
}

char** utf8_split_delim(const char* src, const char* delimiter, uint64_t* capacity) {
    if (!delimiter) {
        return utf8_split(src, capacity);
    }

    UTF8IterSplit split = {
        .current = (char*) src,
        .delimiter = delimiter,
        .capacity = 0,
        .parts = malloc(sizeof(char*)),
    };

    utf8_iter_byte(src, utf8_iter_split, &split);

    const char* end = (const char*) src + utf8_byte_len(src);
    if (split.current < end) {
        split.parts = utf8_split_push_range(split.current, end, split.parts, &split.capacity);
    }

    *capacity = split.capacity;
    return split.parts;
}

char** utf8_split_regex(const char* start, const char* pattern, uint64_t* capacity) {
    if (!start || !pattern || !capacity) {
        return NULL;
    }
    *capacity = 0;

    char** parts = malloc(sizeof(char*));
    if (!parts) {
        LOG_ERROR("Failed to create parts");
        return NULL;
    }

    int error_code;
    PCRE2_SIZE error_offset;
    PCRE2_UCHAR8 error_message[256];
    pcre2_code* code = pcre2_compile(
        (PCRE2_SPTR) pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &error_code,
        &error_offset,
        NULL
    );
    if (!code) {
        pcre2_get_error_message(error_code, error_message, sizeof(error_message));
        LOG_ERROR("PCRE2 compile error at offset %zu: %s", error_offset, error_message);
        return NULL;
    }

    pcre2_match_data* match = pcre2_match_data_create_from_pattern(code, NULL);
    if (!match) {
        LOG_ERROR("Failed to create PCRE2 match data");
        pcre2_code_free(code);
        return NULL;
    }

    int64_t offset = 0;
    int64_t total_bytes = utf8_byte_len(start);
    if (0 == total_bytes || -1 == total_bytes) {
        goto fail;
    }

    while (offset < total_bytes) {
        int rc = pcre2_match(
            code, (PCRE2_SPTR) (start + offset), total_bytes - offset, 0, 0, match, NULL
        );
        if (rc < 0) {
            break;
        }

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match);
        size_t match_start = ovector[0];
        size_t match_end = ovector[1];

        // Only push the matched region (GPT-2 style)
        if (match_end > match_start) {
            parts = utf8_split_push_n(
                start + offset + match_start, match_end - match_start, parts, capacity
            );
            if (!parts) {
                goto fail;
            }
        }
        offset += match_end;
    }

    // No trailing text push: GPT-2 pattern is designed to consume everything
    pcre2_match_data_free(match);
    pcre2_code_free(code);
    return parts;

fail:
    pcre2_match_data_free(match);
    pcre2_code_free(code);
    utf8_split_free(parts, *capacity);
    return NULL;
}

char* utf8_split_join(char** parts, const char* delimiter, uint64_t capacity) {
    if (!parts || 0 == capacity) {
        return NULL;
    }

    uint64_t total = 1;  // add null
    uint64_t sep_len = delimiter ? utf8_byte_len(delimiter) : 0;
    for (uint64_t i = 0; i < capacity; i++) {
        total += utf8_byte_len(parts[i]);
    }
    total += sep_len * (capacity > 1 ? capacity - 1 : 0);

    char* buffer = memory_alloc(total, alignof(char));
    if (!buffer) {
        return NULL;
    }
    buffer[0] = '\0';

    char* previous = NULL;
    for (uint64_t i = 0; i < capacity; i++) {
        if (i > 0 && sep_len) {
            previous = buffer;
            buffer = utf8_concat(buffer, delimiter);
            memory_free(previous);
        }
        previous = buffer;
        buffer = utf8_concat(buffer, parts[i]);
        memory_free(previous);
    }

    return buffer;
}

void utf8_split_free(char** parts, uint64_t capacity) {
    if (parts) {
        for (uint64_t i = 0; i < capacity; i++) {
            if (parts[i]) {
                free(parts[i]);
            }
        }
        free(parts);
    }
}
