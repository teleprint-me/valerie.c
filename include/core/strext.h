/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/strext.h
 * @brief A transitive wrapper to extend string.h operations.
 *
 * Low-level routines for working directly with bytes in null-terminated strings.
 * These routines operate purely on bytes—not codepoints or graphemes.
 *
 * All allocation routines return newly allocated buffers the caller must free.
 * All functions treat empty strings ("") as valid input.
 */

#ifndef STREXT_H
#define STREXT_H

#include <stdint.h>
#include <stddef.h>

#ifndef _STRING
    #include <string.h>  // IWYU pragma: keep
#endif

/**
 * @brief Returns the byte offset from start to end.
 *
 * @param start Pointer to start of buffer.
 * @param end   Pointer to end of buffer.
 * @return      Byte difference (end - start), or -1 if either is NULL.
 */
ptrdiff_t string_diff(const char* start, const char* end);

/**
 * @brief Allocates a new null-terminated copy of the input string.
 *
 * @param start Pointer to a null-terminated string.
 * @return      Newly allocated buffer, or NULL on error. Caller must free.
 */
char* string_copy(const char* start);

/**
 * @brief Allocates a new null-terminated copy of up to n bytes from input.
 *
 * Copies exactly n bytes from start, provided n <= length of input.
 * Returns NULL if n exceeds input length, or on allocation error.
 *
 * @param start Pointer to input string.
 * @param n     Number of bytes to copy (must be <= string_count(start)).
 * @return      Newly allocated buffer, or NULL on error. Caller must free.
 */
char* string_copy_n(const char* start, size_t n);

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
char* string_copy_slice(const char* start, const char* end);

/**
 * @brief Allocates and returns a new string which is the concatenation of dst and src.
 *
 * @param dst  Pointer to a null-terminated string (left operand).
 * @param src  Pointer to a null-terminated string (right operand).
 * @return     Newly allocated buffer, or NULL on allocation error or invalid input.
 *             Caller must free the returned buffer.
 *
 * @note If either input is an empty string, result is a copy of the other.
 * @note If both inputs are empty, result is an empty string ("").
 */
char* string_concat(const char* dst, const char* src);

/**
 * @brief Compares two null-terminated strings lexicographically.
 *
 * Performs a byte-wise comparison of the two strings.
 *
 * @param a Pointer to the first null-terminated string.
 * @param b Pointer to the second null-terminated string.
 * @return
 *   - 0 if strings are equal,
 *   - -1 if a < b,
 *   - 1 if a > b,
 *   - -2 if either input is NULL.
 *
 * @note Comparison stops at the first differing byte or at the null terminator.
 */
int string_compare(const char* a, const char* b);

/**
 * @brief Inserts a given string into a dynamic array of strings at a specified index.
 *
 * @param src      The string to be inserted.
 * @param parts    Dynamic array of pointers (may be reallocated). Must not be NULL.
 * @param count    Pointer to the current count. Will be incremented on success.
 * @param index    The index at which the new string will be inserted.
 *
 * @return         New pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note src is not duplicated and references the input string.
 * @note Caller must assign the return value back to parts.
 * @note On allocation failure, no memory is leaked.
 * @note The appended entry is always a heap-allocated, null-terminated copy.
 */
char** string_insert(const char* src, char** parts, size_t* count, size_t index);

/**
 * @brief Appends a pointer to a dynamic array of char* pointers, resizing as needed.
 *
 * @param src      Pointer to add to the array.
 * @param parts    Dynamic array of pointers (may be reallocated). Must not be NULL.
 * @param count    Pointer to the current count. Will be incremented on success.
 * @return         New pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note src is not duplicated and references the input string.
 * @note Caller must assign the return value back to the parts variable.
 *       (e.g., parts = string_append(...))
 * @note The array is grown by one; previous contents are preserved.
 * @note On allocation failure, NULL is returned and *count is not incremented.
 */
char** string_append(const char* src, char** parts, size_t* count);

/**
 * @brief Makes a heap-allocated copy of the first n bytes from src,
 *        then appends the copy to parts.
 *
 * @param src   Pointer to the bytes to copy.
 * @param n     Number of bytes to copy from src (must not exceed length of src).
 * @param parts Dynamic array of char* pointers (may be reallocated). Must not be NULL.
 * @param count Pointer to the current count; will be incremented on success.
 * @return      New pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note src is not duplicated and references the input string.
 * @note Caller must assign the return value back to parts.
 * @note On allocation failure, no memory is leaked.
 * @note The appended entry is always a heap-allocated, null-terminated copy.
 */
char** string_append_n(const char* src, const size_t n, char** parts, size_t* count);

/**
 * @brief Makes a null-terminated copy of the bytes from [start, end),
 *        then appends the copy to parts.
 *
 * @param start  Pointer to the beginning of the slice (inclusive).
 * @param end    Pointer to the end of the slice (exclusive).
 * @param parts  Dynamic array of char* pointers (may be reallocated).
 * @param count  Pointer to the current count; incremented on success.
 * @return       Pointer to the (possibly reallocated) array, or NULL on error.
 *
 * @note src is not duplicated and references the input string.
 * @note Caller must assign the return value back to the parts variable.
 * @note If [start, end) is empty, appends an empty string.
 * @note On allocation failure, no memory is leaked.
 */
char** string_append_slice(const char* start, const char* end, char** parts, size_t* count);

/**
 * @brief Splits a string into individual bytes as null-terminated strings.
 *
 * @param src   Pointer to the null-terminated byte string.
 * @param count Pointer to count, set to number of parts on return.
 * @return      Array of pointers to newly allocated 1-byte strings (each null-terminated),
 *              or NULL on error. Caller must free each part and the array.
 */
char** string_split(const char* src, size_t* count);

/**
 * @brief Free memory allocated by `string_split`.
 *
 * Frees each individual string in the array and then frees the array itself.
 * The caller must ensure that the array was allocated via `string_split`.
 *
 * @param parts Pointer to the array of pointers to null-terminated strings.
 * @param count Number of elements in the array (must match the actual count).
 */
void string_split_free(char** parts, size_t count);

/**
 * @brief Splits a given string into an array of tokens separated by whitespace.
 *
 * This function takes a string as input and splits it into an array of tokens separated by
 * whitespace. It skips leading whitespace and ignores any trailing whitespace.
 *
 * @param src The input string to be split.
 * @param count A pointer to a variable that will hold the number of tokens in the resulting array.
 *
 * @return A dynamically allocated array of char pointers representing the tokens. The caller is
 * responsible for freeing the memory using `free()` when it is no longer needed.
 */
char** string_split_space(const char* src, size_t* count);

/**
 * @brief Splits a string by the specified delimiter.
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
char** string_split_delim(const char* src, const char* delim, size_t* count);

/**
 * @brief Splits a string into parts matching a PCRE2 regex pattern.
 *
 * @param src      Null-terminated string.
 * @param pattern  Null-terminated regex pattern (PCRE2).
 * @param count    Output: number of parts.
 * @return         Array of pointers to null-terminated substrings (each newly allocated),
 *                 or NULL on error.
 *
 * @note Only matched regions are included in output (GPT-2 BPE style).
 * @note Caller must free each result and the array.
 */
char** string_split_regex(const char* src, const char* pattern, size_t* count);

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
char* string_join(char** parts, size_t count, const char* delim);

#endif  // STREXT_H
