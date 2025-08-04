/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/byte.h
 * @brief UTF-8 byte-oriented string utilities.
 *
 * Low-level routines for working directly with bytes in null-terminated UTF-8 strings.
 * These routines operate purely on bytes—not codepoints or graphemes.
 *
 * All allocation routines return newly allocated buffers the caller must free.
 * All functions treat empty strings ("") as valid input.
 */

#ifndef UTF8_BYTE_H
#define UTF8_BYTE_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Returns the number of bytes before the null terminator in a UTF-8 string.
 *        (Analogous to strlen, but returns -1 for NULL input.)
 *
 * @param start Pointer to a null-terminated UTF-8 string.
 * @return Number of bytes (>=0), or -1 if start is NULL.
 */
int64_t utf8_byte_count(const uint8_t* start);

/**
 * @brief Returns the byte offset from start to end.
 *
 * @param start Pointer to start of buffer.
 * @param end   Pointer to end of buffer.
 * @return      Byte difference (end - start), or -1 if either is NULL.
 */
ptrdiff_t utf8_byte_diff(const uint8_t* start, const uint8_t* end);

/**
 * @brief Allocates a new null-terminated copy of the input string.
 *
 * @param start Pointer to a null-terminated UTF-8 string.
 * @return      Newly allocated buffer, or NULL on error. Caller must free.
 */
uint8_t* utf8_byte_copy(const uint8_t* start);

/**
 * @brief Allocates a new null-terminated copy of up to n bytes from input.
 *
 * Copies exactly n bytes from start, provided n <= length of input.
 * Returns NULL if n exceeds input length, or on allocation error.
 *
 * @param start Pointer to input string.
 * @param n     Number of bytes to copy (must be <= utf8_byte_count(start)).
 * @return      Newly allocated buffer, or NULL on error. Caller must free.
 */
uint8_t* utf8_byte_copy_n(const uint8_t* start, uint64_t n);

/**
 * @brief Allocates a null-terminated copy of bytes from [start, end).
 *
 * Copies bytes from start up to (but not including) end.
 * Returns NULL if end < start or inputs are invalid.
 *
 * @param start Pointer to start of slice.
 * @param end   Pointer to end of slice (exclusive).
 * @return      Newly allocated buffer, or NULL on error. Caller must free.
 */
uint8_t* utf8_byte_copy_slice(const uint8_t* start, const uint8_t* end);

/**
 * @brief Allocates and returns a new string which is the concatenation of dst and src.
 *
 * @param dst  Pointer to a null-terminated UTF-8 string (left operand).
 * @param src  Pointer to a null-terminated UTF-8 string (right operand).
 * @return     Newly allocated buffer, or NULL on allocation error or invalid input.
 *             Caller must free the returned buffer.
 *
 * @note If either input is an empty string, result is a copy of the other.
 * @note If both inputs are empty, result is an empty string ("").
 */
uint8_t* utf8_byte_cat(const uint8_t* dst, const uint8_t* src);

#endif  // UTF8_BYTE_H
