/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/codepoint.h
 * @brief Core API for processing Unicode code points in UTF-8 encoded data.
 *
 * This API provides low-level tools for navigating and operating on UTF-8 code points,
 * distinct from byte-oriented interfaces. It helps ensure correct boundaries and
 * prevents confusion between byte counts and codepoint counts.
 *
 * Concepts:
 * - A **code point** is a single Unicode value (e.g., U+0041 for 'A').
 * - A **code unit** in UTF-8 is an 8-bit byte; each code point is encoded as 1–4 bytes.
 * - A `char*` or `uint8_t*` points to a sequence of code units representing code points.
 *
 * API Conventions:
 * - All functions are prefixed with `utf8_`.
 * - Codepoint-level operations are prefixed with `utf8_cp_`.
 *   - These operate on code point boundaries, never partial bytes.
 * - Byte-level operations (see `utf8/byte.h`) are prefixed with `utf8_byte_`.
 *
 * Safety:
 * - Use `utf8_cp_count()` to count code points, and `utf8_byte_count()` (from the byte API)
 *   to count literal bytes. Never conflate the two—buffer overflows can occur if you
 *   confuse the API boundaries.
 */

#ifndef UTF8_CODEPOINT_H
#define UTF8_CODEPOINT_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

// --- UTF-8 Codepoint Operations ---

int8_t utf8_cp_width(const uint8_t* start);
int32_t utf8_cp_decode(const uint8_t* start);
bool utf8_cp_is_valid(const uint8_t* start);
bool utf8_cp_is_equal(const uint8_t* a, const uint8_t* b);
ptrdiff_t utf8_cp_range(const uint8_t* start, const uint8_t* end);
int64_t utf8_cp_count(const uint8_t* start);
uint8_t* utf8_cp_copy(const uint8_t* start);
uint8_t* utf8_cp_index(const uint8_t* start, uint32_t index);
void utf8_cp_dump(const uint8_t* start);

// --- UTF-8 Codepoint Types ---

bool utf8_cp_is_char(const uint8_t* start);
bool utf8_cp_is_digit(const uint8_t* start);
bool utf8_cp_is_alpha(const uint8_t* start);
bool utf8_cp_is_alnum(const uint8_t* start);
bool utf8_cp_is_lower(const uint8_t* start);
bool utf8_cp_is_space(const uint8_t* start);
bool utf8_cp_is_punct(const uint8_t* start);

// --- UTF-8 Codepoint Visitor ---

const uint8_t* utf8_cp_next(const uint8_t* current);
const uint8_t* utf8_cp_next_width(const uint8_t* current, int8_t* out_width);
const uint8_t* utf8_cp_prev(const uint8_t* start, const uint8_t* current);
const uint8_t* utf8_cp_prev_width(
    const uint8_t* start, const uint8_t* current, int8_t* out_width
);
const uint8_t* utf8_cp_peek(const uint8_t* current, const size_t ahead);

// --- UTF-8 Codepoint Iterator

typedef struct UTF8CpIter {
    const uint8_t* current;  // Current position in string
    char buffer[5];  // UTF-8 codepoint (4 bytes max + null)
} UTF8CpIter;

// Initialize iterator from string start
UTF8CpIter utf8_cp_iter(const uint8_t* start);
// Get next codepoint (returns pointer to buffer, advances position)
const char* utf8_cp_iter_next(UTF8CpIter* it);

// --- UTF-8 Codepoint Split ---

uint8_t** utf8_cp_split(const uint8_t* start, size_t* capacity);
void utf8_cp_split_free(uint8_t** parts, size_t capacity);
void utf8_cp_split_dump(uint8_t** parts, size_t capacity);

#endif  // UTF8_CODEPOINT_H
