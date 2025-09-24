/**
 * @file sort.h
 * @brief Heap sort routines for integer and string arrays.
 */

#ifndef CORE_SORT_H
#define CORE_SORT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sorts an array of integers in ascending order (in-place).
 *
 * @param data  Array of integers to sort.
 * @param count Number of elements in the array.
 */
void heap_sort_int(int* data, size_t count);

/**
 * @brief Sorts an array of NUL-terminated strings in lexicographic ascending order (in-place).
 *
 * @param data  Array of string pointers (char*).
 * @param count Number of elements in the array.
 */
void heap_sort_str(char** data, size_t count);

#ifdef __cplusplus
}
#endif

#endif  // CORE_SORT_H
