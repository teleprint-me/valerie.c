/**
 * @file include/utf8/byte.h
 */

#ifndef UTF8_BYTE_H
#define UTF8_BYTE_H

#include <stdint.h>

/**
 * @brief Returns the number of bytes in a UTF-8 encoded, null-terminated string.
 *
 * Counts literal bytes (not code points). Equivalent to strlen, but returns -1 if input is NULL.
 *
 * @param start Pointer to UTF-8 null-terminated string.
 * @return Number of bytes before the null terminator, 0 if empty string, -1 if start is NULL.
 */
int64_t utf8_byte_count(const uint8_t* start);

/**
 * @brief Allocates and returns a null-terminated byte-for-byte copy of the input UTF-8 string.
 *
 * @param start Pointer to a null-terminated UTF-8 string.
 * @return Newly allocated copy, or NULL on allocation error or if start is NULL.
 *         Caller is responsible for freeing the returned buffer.
 */
uint8_t* utf8_byte_copy(const uint8_t* start);

#endif  // UTF8_BYTE_H
