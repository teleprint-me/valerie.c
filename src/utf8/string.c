/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/utf8/raw.c
 * @brief String API for handling UTF-8 pointer-to-char processing.
 */

#include "memory.h"
#include "logger.h"
#include "utf8/byte.h"
#include "utf8/codepoint.h"
#include "utf8/grapheme.h"
#include "utf8/string.h"

// --- UTF-8 Operations ---

bool utf8_str_is_valid(const char* start) {
    if (!start) {
        return false;
    }

    // Special case: empty string is valid
    if (*start == '\0') {
        return true;
    }

    const uint8_t* stream = (const uint8_t*) start;
    while (*stream) {
        int8_t width = utf8_cp_width(stream);

        // Invalid UTF-8 sequence detected
        if (-1 == width || !utf8_cp_is_valid(stream)) {
            // Capture the error location
            intptr_t diff = utf8_cp_range((const uint8_t*) start, stream);
            LOG_ERROR("Invalid UTF-8 sequence detected at byte offset: %ld", diff);
            return false;  // Stop iteration immediately
        }

        stream += width;  // Advance to the next codepoint
    }

    return true;
}
