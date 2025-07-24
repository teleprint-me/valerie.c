/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/utf8/raw.c
 * @brief Mid-level API for handling raw UTF-8 pointer-to-char processing.
 */

#include "logger.h"
#include "utf8/byte.h"
#include "utf8/raw.h"

// --- UTF-8 Raw Validator ---

void* utf8_raw_iter_is_valid(const uint8_t* start, const int8_t width, void* context) {
    UTF8RawValidator* validator = (UTF8RawValidator*) context;

    if (width == -1) {
        // Invalid UTF-8 sequence detected
        validator->is_valid = false;
        validator->error_at = start; // Capture the error location
        return (void*) validator; // Stop iteration immediately
    }

    validator->is_valid = true; // Mark as valid for this character
    return NULL; // Continue iteration
}

// --- UTF-8 Raw Counter ---

void* utf8_raw_iter_count(const uint8_t* start, const int8_t width, void* context) {
    (void) start;
    (void) width;
    UTF8RawCounter* counter = (UTF8RawCounter*) context;
    counter->value++;
    return NULL; // Continue iteration as long as the source is valid
}

// --- UTF-8 Raw Splitter ---

void* utf8_raw_iter_split(const uint8_t* start, const int8_t width, void* context) {
    UTF8RawSplitter* split = (UTF8RawSplitter*) context;

    const uint8_t* delimiter = (const uint8_t*) split->delimiter;
    while (*delimiter) {
        uint8_t d_length = utf8_byte_width(delimiter);
        if (utf8_byte_is_equal(start, delimiter)) {
            // offset = start, start = end
            split->parts = utf8_raw_split_push_range(
                split->offset, (char*) start, split->parts, &split->capacity
            );
            split->offset = (char*) (start + width); // set next segment start
            break;
        }
        delimiter += d_length;
    }

    return NULL;
}

// --- UTF-8 Raw Operations ---

bool utf8_raw_is_valid(const char* start) {
    if (!start) {
        return false;
    }

    // Special case: empty string is valid
    if (*start == '\0') {
        return true;
    }

    UTF8RawValidator validator = {
        .is_valid = false, // Invalid at start
        .error_at = NULL, // Location unknown
    };

    utf8_byte_iterate(start, utf8_raw_iter_is_valid, &validator);
    if (!validator.is_valid && validator.error_at) {
        LOG_ERROR(
            "Invalid UTF-8 sequence detected at byte offset: %ld",
            validator.error_at - (const uint8_t*) start
        );
    }

    return validator.is_valid;
}

int64_t utf8_raw_byte_count(const char* start) {
    if (!start || !utf8_raw_is_valid(start)) {
        return -1; // Invalid string
    }

    if ('\0' == *start) {
        return 0; // Empty string
    }

    uint64_t byte_length = 0;
    const char* stream = start;
    while (*stream) {
        byte_length++; // increment the byte length counter
        stream++; // move to the next byte in the string
    }

    return byte_length;
}

int64_t utf8_raw_char_count(const char* start) {
    if (!start || !utf8_raw_is_valid(start)) {
        return -1; // Invalid string
    }

    if ('\0' == *start) {
        return 0; // Empty string
    }

    UTF8RawCounter counter = {.value = 0};
    if (utf8_byte_iterate(start, utf8_raw_iter_count, &counter) == NULL) {
        return counter.value;
    }

    return -1;
}

char* utf8_raw_copy(const char* start) {
    if (!start || !utf8_raw_is_valid(start)) {
        return NULL;
    }

    size_t length = utf8_raw_byte_count(start);
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

char* utf8_raw_copy_n(const char* start, const uint64_t length) {
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

char* utf8_raw_copy_range(const char* start, const char* end) {
    if (!start || !end) {
        return NULL;
    }

    ptrdiff_t length = utf8_byte_range((const uint8_t*) start, (const uint8_t*) end);
    if (-1 == length) {
        return NULL;
    }

    return utf8_raw_copy_n(start, (const uint64_t) length);
}

char* utf8_raw_concat(const char* head, const char* tail) {
    // Check for null pointers
    if (!head || !tail) {
        LOG_ERROR("Invalid head or tail parameter");
        return NULL;
    }

    // Validate the left and right operands, but allow empty strings
    if ('\0' != *head && !utf8_raw_is_valid(head)) {
        LOG_ERROR("Invalid head operand");
        return NULL;
    }

    if ('\0' != *tail && !utf8_raw_is_valid(tail)) {
        LOG_ERROR("Invalid tail operand");
        return NULL;
    }

    // Concatenate the right operand to the left operand
    size_t head_length = utf8_raw_byte_count(head);
    size_t tail_length = utf8_raw_byte_count(tail);
    size_t output_length = head_length + tail_length;

    // Add 1 for null terminator
    char* output = (char*) malloc(output_length + 1);
    if (output == NULL) {
        LOG_ERROR("Failed to allocate memory for concatenated string.");
        return NULL;
    }

    // Copy string bytes into output
    memcpy(output, head, head_length);
    memcpy(output + head_length, tail, tail_length);
    output[output_length] = '\0'; // Null-terminate the string

    return output;
}

// --- UTF-8 Raw Compare ---

int32_t utf8_raw_compare(const char* a, const char* b) {
    if (!a || !b) {
        LOG_ERROR("One or both source strings are NULL.");
        return UTF8_RAW_COMPARE_INVALID; // NULL strings are invalid inputs.
    }

    if (!utf8_raw_is_valid(a)) {
        LOG_ERROR("First source string is not a valid UTF-8 string.");
        return UTF8_RAW_COMPARE_INVALID; // Indicate invalid UTF-8 string.
    }

    if (!utf8_raw_is_valid(b)) {
        LOG_ERROR("Second source string is not a valid UTF-8 string.");
        return UTF8_RAW_COMPARE_INVALID; // Indicate invalid UTF-8 string.
    }

    const uint8_t* a_stream = (const uint8_t*) a;
    const uint8_t* b_stream = (const uint8_t*) b;

    while (*a_stream && *b_stream) {
        if (*a_stream < *b_stream) {
            return UTF8_RAW_COMPARE_LESS;
        }
        if (*a_stream > *b_stream) {
            return UTF8_RAW_COMPARE_GREATER;
        }
        // Both bytes are equal, move to the next
        a_stream++;
        b_stream++;
    }

    // Check if strings are of different lengths
    if (*a_stream) {
        return UTF8_RAW_COMPARE_GREATER;
    }
    if (*b_stream) {
        return UTF8_RAW_COMPARE_LESS;
    }

    return UTF8_RAW_COMPARE_EQUAL;
}

// --- UTF-8 Raw Split ---

char** utf8_raw_split_push(const char* start, char** parts, uint64_t* capacity) {
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

char** utf8_raw_split_push_n(const char* start, const uint64_t length, char** parts, uint64_t* capacity) {
    if (!start || !parts || !capacity) {
        return NULL;
    }

    char* segment = utf8_raw_copy_n(start, length);
    if (!segment) {
        return NULL;
    }

    char** temp = utf8_raw_split_push(segment, parts, capacity);
    if (!temp) {
        free(segment);
        return NULL;
    }

    return temp;
}

char** utf8_raw_split_push_range(const char* start, const char* end, char** parts, uint64_t* capacity) {
    if (!start || !end || !parts || !capacity) {
        return NULL;
    }

    char* segment = utf8_raw_copy_range(start, end);
    if (!segment) {
        return NULL;
    }

    char** temp = utf8_raw_split_push(segment, parts, capacity);
    if (!temp) {
        free(segment);
        return NULL;
    }

    return temp;
}

char** utf8_raw_split(const char* start, const char* delimiter, uint64_t* capacity) {
    UTF8RawSplitter split = {
        .offset = (char*) start,
        .delimiter = delimiter,
        .capacity = 0,
        .parts = malloc(sizeof(char*)),
    };

    utf8_byte_iterate(start, utf8_raw_iter_split, &split);

    const char* end = (const char*) start + utf8_raw_byte_count(start);
    if (split.offset < end) {
        split.parts = utf8_raw_split_push_range(split.offset, end, split.parts, &split.capacity);
    }

    *capacity = split.capacity;
    return split.parts;
}

char** utf8_raw_split_char(const char* start, uint64_t* capacity) {
    if (!start || !capacity) {
        return NULL;
    }

    *capacity = 0;
    char** parts = malloc(sizeof(char*)); // Start empty array
    const uint8_t* ptr = (const uint8_t*) start;

    while (*ptr) {
        int8_t width = utf8_byte_width(ptr);
        if (width <= 0) {
            break;
        }

        parts = utf8_raw_split_push_n((const char*) ptr, width, parts, capacity);
        ptr += width;
    }

    return parts;
}

char** utf8_raw_split_regex(const char* start, const char* pattern, uint64_t* capacity) {
    if (!start || !pattern || !capacity) return NULL;
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
    int64_t total_bytes = utf8_raw_byte_count(start);
    if (0 == total_bytes || -1 == total_bytes) {
        goto fail;
    }

    while (offset < total_bytes) {
        int rc = pcre2_match(code, (PCRE2_SPTR)(start + offset), total_bytes - offset, 0, 0, match, NULL);
        if (rc < 0) break;

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match);
        size_t match_start = ovector[0];
        size_t match_end   = ovector[1];

        // Only push the matched region (GPT-2 style)
        if (match_end > match_start) {
            parts = utf8_raw_split_push_n(start + offset + match_start, match_end - match_start, parts, capacity);
            if (!parts) goto fail;
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
    utf8_raw_split_free(parts, *capacity);
    return NULL;
}

void utf8_raw_split_free(char** parts, uint64_t capacity) {
    if (parts) {
        for (uint64_t i = 0; i < capacity; i++) {
            if (parts[i]) {
                free(parts[i]);
            }
        }
        free(parts);
    }
}
