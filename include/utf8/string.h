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
 */

#ifndef UTF8_STRING_H
#define UTF8_STRING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

// --- UTF-8 Raw Validator ---

typedef struct UTF8RawValidator {
    bool is_valid;
    const uint8_t* error_at;
} UTF8RawValidator;

void* utf8_raw_iter_is_valid(const uint8_t* start, const int8_t width, void* context);

// --- UTF-8 Raw Counter ---

typedef struct UTF8RawCounter {
    int64_t value;
} UTF8RawCounter;

void* utf8_raw_iter_count(const uint8_t* start, const int8_t width, void* context);

// --- UTF-8 Raw Splitter ---

typedef struct UTF8RawSplitter {
    const char* delimiter;
    char* offset;
    char** parts;
    uint64_t capacity;
} UTF8RawSplitter;

void* utf8_raw_iter_split(const uint8_t* start, const int8_t width, void* context);

// --- UTF-8 Raw Operations ---

bool utf8_raw_is_valid(const char* start);

int64_t utf8_raw_byte_count(const char* start); // Physical byte size
int64_t utf8_raw_char_count(const char* start); // Logical character count

char* utf8_raw_copy(const char* start);
char* utf8_raw_copy_n(const char* start, const uint64_t length);
char* utf8_raw_copy_range(const char* start, const char* end);

char* utf8_raw_concat(const char* head, const char* tail);

// --- UTF-8 Raw Compare ---

typedef enum UTF8RawCompare {
    UTF8_RAW_COMPARE_INVALID = -2,
    UTF8_RAW_COMPARE_LESS = -1,
    UTF8_RAW_COMPARE_EQUAL = 0,
    UTF8_RAW_COMPARE_GREATER = 1
} UTF8RawCompare;

int32_t utf8_raw_compare(const char* a, const char* b);

// --- UTF-8 Raw Split ---

/**
 * @note Caller must always assign the return value back to parts.
 */
char** utf8_raw_split_push(const char* start, char** parts, uint64_t* capacity);
char** utf8_raw_split_push_n(const char* start, const uint64_t length, char** parts, uint64_t* capacity);
char** utf8_raw_split_push_range(const char* start, const char* end, char** parts, uint64_t* capacity);

char** utf8_raw_split(const char* start, const char* delimiter, uint64_t* capacity);
char** utf8_raw_split_char(const char* start, uint64_t* capacity);
char** utf8_raw_split_regex(const char* start, const char* pattern, uint64_t* capacity);
char* utf8_raw_split_join(char** parts, const char* delimiter, uint64_t capacity);
void utf8_raw_split_free(char** parts, uint64_t capacity);

#endif // UTF8_STRING_H
