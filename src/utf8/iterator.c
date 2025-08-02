/**
 * @file src/utf8/iterator.c
 * @brief ASCII and UTF-8 Codepoint API.
 *
 * Low-level API for handling core UTF-8 codepoint pre-processing.
 *
 * - A UTF-8 byte represents a valid ASCII or UTF-8 code point.
 * - Library functions are prefixed with `utf8_`.
 * - Low Level: Byte functions are prefixed with `utf8_byte_`.
 * - Low Level: Byte iterators are prefixed with `utf8_iter_`.
 *
 * @note Most iterators are callbacks for utf8_iter_byte().
 *       The only exception so far is the codepoint iterator.
 */

#include "memory.h"
#include "utf8/byte.h"
#include "utf8/iterator.h"

#include <stdint.h>
#include <string.h>

/**
 * UTF-8 Byte Iterator
 * @{
 */

void* utf8_iter_byte(const char* start, UTF8IterByte callback, void* context) {
    if (!start || !callback) {
        return NULL; // Invalid source or callback
    }

    const uint8_t* stream = (const uint8_t*) start;
    while (*stream) {
        // Determine the width of the current UTF-8 character
        int8_t width = utf8_byte_width(stream);
        if (width == -1 || !utf8_byte_is_valid(stream)) {
            // Notify the callback of an invalid sequence and allow it to decide
            void* result = callback(stream, -1, context);
            if (result) {
                return result; // Early return based on callback result
            }
            stream++; // Move past the invalid byte to prevent infinite loops
            continue;
        }

        // Invoke the callback with the current character
        void* result = callback(stream, width, context);
        if (result) {
            return result; // Early return based on callback result
        }

        stream += width; // Advance to the next character
    }

    return NULL; // Completed iteration without finding a result
}

/** @} */

/**
 * UTF-8 Validator
 * @{
 */

void* utf8_iter_is_valid(const uint8_t* start, const int8_t width, void* context) {
    UTF8IterValidator* validator = (UTF8IterValidator*) context;

    if (width == -1) {
        // Invalid UTF-8 sequence detected
        validator->is_valid = false;
        validator->error_at = start; // Capture the error location
        return (void*) validator; // Stop iteration immediately
    }

    validator->is_valid = true; // Mark as valid for this character
    return NULL; // Continue iteration
}

/** @} */

/**
 * UTF-8 Counter
 * @{
 */

void* utf8_iter_count(const uint8_t* start, const int8_t width, void* context) {
    (void) start;
    (void) width;
    UTF8IterCounter* counter = (UTF8IterCounter*) context;
    counter->value++;
    return NULL; // Continue iteration as long as the source is valid
}

/** @} */

/**
 * UTF-8 Splitter
 * @{
 */

void* utf8_iter_split(const uint8_t* start, const int8_t width, void* context) {
    UTF8IterSplit* split = (UTF8IterSplit*) context;
    if (!start || !split || !split->delimiter) {
        return NULL;
    }

    const uint8_t* delimiter = (const uint8_t*) split->delimiter;

    // check current codepoint
    while (*delimiter) {
        int8_t d_width = utf8_byte_width(delimiter);
        if (-1 == d_width) {
            return NULL;
        }

        if (utf8_byte_is_equal(start, delimiter)) {
            // current = start, start = end
            ptrdiff_t range = utf8_byte_range((const uint8_t*) split->current, start);
            if (-1 == range) {
                return NULL;
            }

            // copy up to n bytes
            uint8_t* segment = memory_alloc((size_t) range + 1, alignof(uint8_t));
            if (!segment) {
                return NULL;
            }

            if (0 < range) {
                memcpy(segment, split->current, (size_t) range);
            }
            segment[range] = '\0';

            // copy n bytes into parts
            char** temp = memory_realloc(
                split->parts,
                sizeof(char*) * split->capacity, // old_size
                sizeof(char*) * (split->capacity + 1), // new_size
                alignof(char*)
            );
            if (!temp) {
                return NULL;
            }

            split->parts = temp;
            split->parts[split->capacity++] = (char*) segment;
            split->current = (char*) (start + width); // Move offset past delimiter
            break;
        }
        delimiter += d_width;
    }

    return NULL; // do not free allocated memory!
}

/** @} */

/**
 * UTF-8 Codepoint
 */

UTF8IterCodepoint utf8_iter_codepoint(const char* start) {
    return (UTF8IterCodepoint) {
        .current = (const uint8_t*) start,
        .buffer = {0},
    };
}

const char* utf8_iter_next_codepoint(UTF8IterCodepoint* it) {
    if (!it || !it->current || !*it->current) {
        return NULL;
    }

    int8_t width = utf8_byte_width(it->current);
    if (-1 == width || !utf8_byte_is_valid(it->current)) {
        return NULL; // invalid or corrupt
    }

    // Copy this codepoint into buffer
    for (int i = 0; i < width; i++) {
        it->buffer[i] = (char) it->current[i];
    }
    it->buffer[width] = '\0'; // null terminate

    it->current += width; // advance current position
    return it->buffer;
}

/** @} */
