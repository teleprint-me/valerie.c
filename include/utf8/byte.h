/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/byte.h
 * @brief ASCII and UTF-8 Codepoint API.
 *
 * Low-level API for handling core UTF-8 codepoint pre-processing.
 *
 * - A UTF-8 byte represents a valid ASCII or UTF-8 code point.
 * - Library functions are prefixed with `utf8_`.
 * - Low Level: Byte functions are prefixed with `utf8_byte_`.
 * - Mid Level: Raw pointer-to-char functions are prefixed with `utf8_raw_`.
 * - High Level: interface functions are prefixed with `utf8_string_`.
 */

#ifndef UTF8_BYTE_H
#define UTF8_BYTE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

// --- UTF-8 Byte Operations ---

int8_t utf8_byte_width(const uint8_t* start);
int32_t utf8_byte_decode(const uint8_t* start);
bool utf8_byte_is_valid(const uint8_t* start);
bool utf8_byte_is_equal(const uint8_t* a, const uint8_t* b);
ptrdiff_t utf8_byte_range(const uint8_t* start, const uint8_t* end);
void utf8_byte_dump(const uint8_t* start);

// --- UTF-8 Byte Types ---

bool utf8_byte_is_char(const uint8_t* start);
bool utf8_byte_is_digit(const uint8_t* start);
bool utf8_byte_is_alpha(const uint8_t* start);
bool utf8_byte_is_alnum(const uint8_t* start);
bool utf8_byte_is_upper(const uint8_t* start);
bool utf8_byte_is_lower(const uint8_t* start);
bool utf8_byte_is_space(const uint8_t* start);
bool utf8_byte_is_punct(const uint8_t* start);

// --- UTF-8 Byte Visitor ---

const uint8_t* utf8_byte_next(const uint8_t* current);
const uint8_t* utf8_byte_next_width(const uint8_t* current, int8_t* out_width);

const uint8_t* utf8_byte_prev(const uint8_t* start, const uint8_t* current);
const uint8_t* utf8_byte_prev_width(const uint8_t* start, const uint8_t* current, int8_t* out_width);

/**
 * Peek `ahead` valid codepoints from `current`, skipping invalid bytes.
 * Returns a pointer to the codepoint, or NULL if out of bounds or invalid.
 */
const uint8_t* utf8_byte_peek(const uint8_t* current, const size_t ahead);

// --- UTF-8 Byte Iterator ---

typedef void* (*UTF8ByteIterator)(const uint8_t* start, const int8_t width, void* context);
void* utf8_byte_iterate(const char* start, UTF8ByteIterator callback, void* context);

#endif // UTF8_BYTE_H
