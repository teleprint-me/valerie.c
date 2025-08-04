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

// Useful for self documenting code
typedef enum UTF8ByteCompare {
    UTF8_COMPARE_INVALID = -2,
    UTF8_COMPARE_LESS = -1,
    UTF8_COMPARE_EQUAL = 0,
    UTF8_COMPARE_GREATER = 1
} UTF8ByteCompare;

/**
 * @brief Compares two null-terminated UTF-8 byte strings lexicographically.
 *
 * Performs a byte-wise comparison of the two strings.
 *
 * @param a Pointer to the first null-terminated UTF-8 string.
 * @param b Pointer to the second null-terminated UTF-8 string.
 * @return
 *   - UTF8_COMPARE_EQUAL (0) if strings are equal,
 *   - UTF8_COMPARE_LESS (-1) if a < b,
 *   - UTF8_COMPARE_GREATER (1) if a > b,
 *   - UTF8_COMPARE_INVALID (-2) if either input is NULL.
 *
 * @note This function compares raw bytes, not Unicode codepoints or grapheme clusters.
 * @note Comparison stops at the first differing byte or at the null terminator.
 */
int8_t utf8_byte_cmp(const uint8_t* a, const uint8_t* b);

/**
 * @brief Appends a pointer to a dynamic array of uint8_t* pointers, resizing as needed.
 *
 * @param src      Pointer to add to the array.
 * @param parts    Dynamic array of pointers (may be reallocated). Must not be NULL.
 * @param count    Pointer to the current count. Will be incremented on success.
 * @return         New pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note Caller must assign the return value back to the parts variable.
 *       (e.g., parts = utf8_byte_append(...))
 * @note The array is grown by one; previous contents are preserved.
 * @note On allocation failure, NULL is returned and *count is not incremented.
 */
uint8_t** utf8_byte_append(const uint8_t* src, uint8_t** parts, uint64_t* count);

/**
 * @brief Makes a heap-allocated copy of the first n bytes from src,
 *        then appends the copy to parts.
 *
 * @param src   Pointer to the bytes to copy.
 * @param n     Number of bytes to copy from src (must not exceed length of src).
 * @param parts Dynamic array of uint8_t* pointers (may be reallocated). Must not be NULL.
 * @param count Pointer to the current count; will be incremented on success.
 * @return      New pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note Caller must assign the return value back to parts.
 * @note On allocation failure, no memory is leaked.
 * @note The appended entry is always a heap-allocated, null-terminated copy.
 */
uint8_t** utf8_byte_append_n(
    const uint8_t* src, const uint64_t n, uint8_t** parts, uint64_t* count
);

/**
 * @brief Makes a null-terminated copy of the bytes from [start, end),
 *        then appends the copy to parts.
 *
 * @param start  Pointer to the beginning of the slice (inclusive).
 * @param end    Pointer to the end of the slice (exclusive).
 * @param parts  Dynamic array of uint8_t* pointers (may be reallocated).
 * @param count  Pointer to the current count; incremented on success.
 * @return       Pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note Caller must assign the return value back to the parts variable.
 * @note If [start, end) is empty, appends an empty string.
 * @note On allocation failure, no memory is leaked.
 */
uint8_t** utf8_byte_append_slice(
    const uint8_t* start, const uint8_t* end, uint8_t** parts, uint64_t* count
);

/**
 * @brief Splits a UTF-8 byte string into individual bytes as null-terminated strings.
 *
 * @param src   Pointer to the null-terminated byte string.
 * @param count Pointer to count, set to number of parts on return.
 * @return      Array of pointers to newly allocated 1-byte strings (each null-terminated),
 *              or NULL on error. Caller must free each part and the array.
 */
uint8_t** utf8_byte_split(const uint8_t* src, uint64_t* count);

/**
 * @brief Free memory allocated by `utf8_byte_split`.
 *
 * Frees each individual string in the array and then frees the array itself.
 * The caller must ensure that the array was allocated via `utf8_byte_split`.
 *
 * @param parts Pointer to the array of pointers to null-terminated strings.
 * @param count Number of elements in the array (must match the actual count).
 */
void utf8_byte_split_free(uint8_t** parts, uint64_t count);

/**
 * @brief Splits a UTF-8 string by the specified delimiter (literal byte sequence).
 *
 * @param src   Null-terminated input string.
 * @param delim Null-terminated delimiter string (multi-byte supported).
 * @param count Output: set to the number of parts.
 * @return      Array of pointers to null-terminated slices (each newly allocated).
 *              NULL on error. Caller must free each part and the array.
 *
 * @note If delim is NULL or empty, splits into individual bytes.
 * @note Empty substrings between consecutive delimiters are included.
 */
uint8_t** utf8_byte_split_delim(const uint8_t* src, const uint8_t* delim, uint64_t* count);

/**
 * @brief Splits a UTF-8 byte string into parts matching a PCRE2 regex pattern.
 *
 * @param src      Null-terminated UTF-8 byte string.
 * @param pattern  Null-terminated regex pattern (PCRE2).
 * @param count    Output: number of parts.
 * @return         Array of pointers to null-terminated substrings (each newly allocated),
 *                 or NULL on error.
 *
 * @note Only matched regions are included in output (GPT-2 BPE style).
 * @note Caller must free each result and the array.
 */
uint8_t** utf8_byte_split_regex(const uint8_t* src, const uint8_t* pattern, uint64_t* count);

/**
 * @brief Joins an array of null-terminated byte strings into one string, with optional delimiter.
 *
 * @param parts    Array of null-terminated byte strings to join.
 * @param count    Number of elements in parts.
 * @param delim    Optional delimiter to insert between each part (may be NULL).
 * @return         Newly allocated, null-terminated string; NULL on error.
 *
 * @note Caller must free the result.
 * @note If count is 0, returns NULL.
 */
uint8_t* utf8_byte_join(uint8_t** parts, uint64_t count, const uint8_t* delim);

#endif  // UTF8_BYTE_H
