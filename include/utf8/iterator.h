/**
 * @file include/utf8/iterator.h
 * @brief ASCII and UTF-8 Codepoint API.
 *
 * Low-level API for handling core UTF-8 codepoint pre-processing.
 *
 * - A UTF-8 byte represents a valid ASCII or UTF-8 code point.
 * - Library functions are prefixed with `utf8_`.
 * - Low Level: Byte functions are prefixed with `utf8_byte_`.
 * - Low Level: Byte iterators are prefixed with `utf8_iter_`.
 * - Low Level: Codepoint functions are prefixed with `utf8_cpt_`.
 * - Low Level: Grapheme clusters are prefixed with `utf8_gph_`.
 * - High Level: String functions are simply prefixed with `utf8_`.
 *
 * @note Most iterators are callbacks for utf8_iter_byte().
 *       The only exception so far is the codepoint iterator.
 */

#ifndef UTF8_ITERATOR_H
#define UTF8_ITERATOR_H

#include "posix.h" // IWYU pragma: keep
#include <stdint.h>

/**
 * @defgroup utf8_byte_iterator UTF-8 Byte Iterator
 * @brief Iteration helper for processing UTF-8 sequences with callbacks.
 * @{
 */

/**
 * @brief Iterator callback signature for utf8_byte_iterate().
 *
 * @param start Pointer to the current sequence.
 * @param width Width of the sequence, or -1 if invalid.
 * @param context User-defined context pointer.
 * @return Optional early-exit value; if non-NULL, iteration stops and the value is returned.
 */
typedef void* (*UTF8IterByte)(const uint8_t* start, const int8_t width, void* context);

/**
 * @brief Iterates over each UTF-8 sequence in a string.
 *
 * Calls the provided callback for every sequence, including invalid ones.
 *
 * @param start NULL-terminated UTF-8 string.
 * @param callback User-defined function to invoke per sequence.
 * @param context User-defined context pointer passed to the callback.
 * @return
 *  - NULL if fully processed.
 *  - Callback return value if early exit occurred.
 */
void* utf8_iter_byte(const char* start, UTF8IterByte callback, void* context);

/** @} */

/**
 * UTF-8 Validator
 * @{
 */

typedef struct UTF8IterValidator {
    bool is_valid; // Final validation result
    const uint8_t* error_at; // Location of first invalid sequence
} UTF8IterValidator;

// Iteration callback: validates codepoints, stops on invalid
void* utf8_iter_is_valid(const uint8_t* start, const int8_t width, void* context);

/** @} */

/**
 * UTF-8 Counter
 * @{
 */

typedef struct UTF8IterCounter {
    int64_t value; // Running count of codepoints
} UTF8IterCounter;

// Iteration callback: counts codepoints
void* utf8_iter_count(const uint8_t* start, const int8_t width, void* context);

/** @} */

/**
 * UTF-8 Splitter
 * @{
 */

typedef struct UTF8IterSplit {
    const char* delimiter; // Delimiter string (must be valid UTF-8)
    char* current; // Current segment start
    char** parts; // Accumulated split parts
    uint64_t capacity; // Number of parts pushed
} UTF8IterSplit;

// Iteration callback: splits into segments at delimiters
void* utf8_iter_split(const uint8_t* start, const int8_t width, void* context);

/** @} */

/**
 * UTF-8 Iterator
 * @{
 */

typedef struct UTF8IterCodepoint {
    const uint8_t* current; // Current position in string
    char buffer[5]; // UTF-8 codepoint (4 bytes max + null)
} UTF8IterCodepoint;

// Initialize iterator from string start
UTF8IterCodepoint utf8_iter_codepoint(const char* start);
// Get next codepoint (returns pointer to buffer, advances position)
const char* utf8_iter_next_codepoint(UTF8IterCodepoint* it);

/** @} */

#endif // UTF8_ITERATOR_H
