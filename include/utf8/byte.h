/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/byte.h
 * @brief ASCII and UTF-8 Codepoint API.
 *
 * Low-level API for handling core UTF-8 codepoint pre-processing.
 *
 * - A UTF-8 byte represents a valid ASCII or UTF-8 codepoint.
 * - All Library functions are prefixed with `utf8_`.
 * - Byte-level operations are prefixed with `utf8_byte_`.
 */

#ifndef UTF8_BYTE_H
#define UTF8_BYTE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

// --- UTF-8 Codepoint Operations ---

int8_t utf8_byte_width(const uint8_t* start);
int32_t utf8_byte_decode(const uint8_t* start);
bool utf8_byte_is_valid(const uint8_t* start);
bool utf8_byte_is_equal(const uint8_t* a, const uint8_t* b);
ptrdiff_t utf8_byte_range(const uint8_t* start, const uint8_t* end);
int64_t utf8_byte_count(const uint8_t* start);
uint8_t* utf8_byte_copy(const uint8_t* start);
uint8_t* utf8_byte_index(const uint8_t* start, uint32_t index);
void utf8_byte_dump(const uint8_t* start);

// --- UTF-8 Codepoint Types ---

bool utf8_byte_is_char(const uint8_t* start);
bool utf8_byte_is_digit(const uint8_t* start);
bool utf8_byte_is_alpha(const uint8_t* start);
bool utf8_byte_is_alnum(const uint8_t* start);
bool utf8_byte_is_lower(const uint8_t* start);
bool utf8_byte_is_space(const uint8_t* start);
bool utf8_byte_is_punct(const uint8_t* start);

// --- UTF-8 Codepoint Visitor ---

const uint8_t* utf8_byte_next(const uint8_t* current);
const uint8_t* utf8_byte_next_width(const uint8_t* current, int8_t* out_width);
const uint8_t* utf8_byte_prev(const uint8_t* start, const uint8_t* current);
const uint8_t* utf8_byte_prev_width(
    const uint8_t* start, const uint8_t* current, int8_t* out_width
);
const uint8_t* utf8_byte_peek(const uint8_t* current, const size_t ahead);

// --- UTF-8 Codepoint Iterator

typedef struct UTF8ByteIter {
    const uint8_t* current; // Current position in string
    char buffer[5]; // UTF-8 codepoint (4 bytes max + null)
} UTF8ByteIter;

// Initialize iterator from string start
UTF8ByteIter utf8_byte_iter(const uint8_t* start);
// Get next codepoint (returns pointer to buffer, advances position)
const char* utf8_byte_iter_next(UTF8ByteIter* it);

// --- UTF-8 Codepoint Split ---

uint8_t** utf8_byte_split(const uint8_t* start, size_t* capacity);
void utf8_byte_split_free(uint8_t** parts, size_t capacity);
void utf8_byte_split_dump(uint8_t** parts, size_t capacity);

#endif // UTF8_BYTE_H
