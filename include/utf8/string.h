/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/string.h
 * @brief ASCII and UTF-8 String API.
 *
 * String API for handling UTF-8 pointer-to-char processing.
 *
 * - A UTF-8 byte represents a valid ASCII or UTF-8 code point.
 * - Library functions are prefixed with `utf8_`.
 * - Low Level: Byte functions are prefixed with `utf8_byte_`.
 * - High Level: String functions are prefixed with `utf8_`.
 */

#ifndef UTF8_STRING_H
#define UTF8_STRING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

// --- UTF-8 Validator ---

typedef struct UTF8Validator {
    bool is_valid;
    const uint8_t* error_at;
} UTF8Validator;

void* utf8_iter_is_valid(const uint8_t* start, const int8_t width, void* context);

// --- UTF-8 Counter ---

typedef struct UTF8Counter {
    int64_t value;
} UTF8Counter;

void* utf8_iter_count(const uint8_t* start, const int8_t width, void* context);

// --- UTF-8 Splitter ---

typedef struct UTF8Splitter {
    const char* delimiter;
    char* offset;
    char** parts;
    uint64_t capacity;
} UTF8Splitter;

void* utf8_iter_split(const uint8_t* start, const int8_t width, void* context);

// --- UTF-8 Operations ---

bool utf8_is_valid(const char* start);

int64_t utf8_len_bytes(const char* start);  // Physical byte size
int64_t utf8_len_codepoints(const char* start);  // Logical character count

char* utf8_copy(const char* start);
char* utf8_copy_n(const char* start, const uint64_t length);
char* utf8_copy_range(const char* start, const char* end);

/**
 * @note dst is **not** modified in-place.
 * @returns A new buffer where src is copied to dst.
 */
char* utf8_concat(const char* dst, const char* src);

// --- UTF-8 Compare ---

typedef enum UTF8Compare {
    UTF8_COMPARE_INVALID = -2,
    UTF8_COMPARE_LESS = -1,
    UTF8_COMPARE_EQUAL = 0,
    UTF8_COMPARE_GREATER = 1
} UTF8Compare;

int32_t utf8_compare(const char* a, const char* b);

// --- UTF-8 Split ---

/**
 * @note Caller must always assign the return value back to parts.
 */
char** utf8_split_push(const char* start, char** parts, uint64_t* capacity);
char** utf8_split_push_n(const char* start, const uint64_t length, char** parts, uint64_t* capacity);
char** utf8_split_push_range(const char* start, const char* end, char** parts, uint64_t* capacity);

/**
 * @note Delimiters must be valid. Passing NULL as a delimiter cancels the operation.
 */
char** utf8_split(const char* start, const char* delimiter, uint64_t* capacity);
char** utf8_split_char(const char* start, uint64_t* capacity);
char** utf8_split_regex(const char* start, const char* pattern, uint64_t* capacity);

char* utf8_split_join(char** parts, const char* delimiter, uint64_t capacity);
void utf8_split_free(char** parts, uint64_t capacity);

#endif  // UTF8_STRING_H
