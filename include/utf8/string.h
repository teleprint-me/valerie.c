/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/string.h
 * @brief ASCII and UTF-8 String API.
 *
 * High-level string operations for UTF-8 pointer-to-char processing.
 * - Low Level: Byte functions are prefixed with `utf8_cp_`.
 * - High Level: String functions are prefixed with `utf8_`.
 */

#ifndef UTF8_STRING_H
#define UTF8_STRING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

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
