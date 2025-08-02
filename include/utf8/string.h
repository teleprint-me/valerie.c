/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/string.h
 * @brief ASCII and UTF-8 String API.
 *
 * High-level string operations for UTF-8 pointer-to-char processing.
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

/** UTF-8 Validator */

typedef struct UTF8Validator {
    bool is_valid; // Final validation result
    const uint8_t* error_at; // Location of first invalid sequence
} UTF8Validator;

// Iteration callback: validates codepoints, stops on invalid
void* utf8_iter_is_valid(const uint8_t* start, const int8_t width, void* context);

/** UTF-8 Counter */

typedef struct UTF8Counter {
    int64_t value; // Running count of codepoints
} UTF8Counter;

// Iteration callback: counts codepoints
void* utf8_iter_count(const uint8_t* start, const int8_t width, void* context);

/** UTF-8 Splitter */

typedef struct UTF8Splitter {
    const char* delimiter; // Delimiter string (must be valid UTF-8)
    char* offset; // Current segment start
    char** parts; // Accumulated split parts
    uint64_t capacity; // Number of parts pushed
} UTF8Splitter;

// Iteration callback: splits into segments at delimiters
void* utf8_iter_split(const uint8_t* start, const int8_t width, void* context);

/** UTF-8 Iterator */

typedef struct UTF8Iterator {
    const uint8_t* current; // Current position in string
    char buffer[5]; // UTF-8 codepoint (4 bytes max + null)
} UTF8Iterator;

// Initialize iterator from string start
UTF8Iterator utf8_iter(const char* start);
// Get next codepoint (returns pointer to buffer, advances position)
const char* utf8_iter_next(UTF8Iterator* it);

/** UTF-8 Operations */

// Validate entire string (logs on failure)
bool utf8_is_valid(const char* src);

// Physical byte length (returns -1 if invalid)
int64_t utf8_len_bytes(const char* src);
// Logical codepoint count (returns -1 if invalid)
int64_t utf8_len_codepoints(const char* src);

// Get a single codepoint copy by index (caller frees)
char* utf8_codepoint_index(const char* src, uint64_t index);
// @todo Not implemented
char* utf8_codepoint_range(const char* src, uint64_t start, uint64_t end);

// Allocate full copy (caller frees)
char* utf8_copy(const char* src);
// Copy first N bytes (not codepoints) — caller frees
char* utf8_copy_n(const char* src, const uint64_t length);
// Copy range between two pointers (caller frees)
char* utf8_copy_range(const char* start, const char* end);

// Concatenate two strings (returns new buffer, not in-place)
char* utf8_concat(const char* dst, const char* src);

/** UTF-8 Compare */

typedef enum UTF8Compare {
    UTF8_COMPARE_INVALID = -2, // Invalid input
    UTF8_COMPARE_LESS = -1,
    UTF8_COMPARE_EQUAL = 0,
    UTF8_COMPARE_GREATER = 1
} UTF8Compare;

// Lexicographical comparison
int32_t utf8_compare(const char* a, const char* b);

/** UTF-8 Split Tools */

// Push a raw pointer to parts array
char** utf8_split_push(const char* src, char** parts, uint64_t* capacity);
// Push a copy of N bytes
char** utf8_split_push_n(const char* src, const uint64_t length, char** parts, uint64_t* capacity);
// Push a copy of [start, end)
char** utf8_split_push_range(const char* start, const char* end, char** parts, uint64_t* capacity);

/** UTF-8 Split */

// Split into individual codepoints
char** utf8_split(const char* src, uint64_t* capacity);

// Split by delimiter (returns array of parts, caller frees)
char** utf8_split_delim(const char* src, const char* delimiter, uint64_t* capacity);

// Split using PCRE2 regex pattern
char** utf8_split_regex(const char* src, const char* pattern, uint64_t* capacity);

// Join parts with delimiter (returns new string, caller frees)
char* utf8_split_join(char** parts, const char* delimiter, uint64_t capacity);

// Free parts array and all allocated segments
void utf8_split_free(char** parts, uint64_t capacity);

#endif // UTF8_STRING_H
