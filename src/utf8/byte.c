/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/utf8/byte.c
 * @brief Low-level API for handling core UTF-8 codepoint pre-processing.
 */

#include "logger.h"
#include "utf8/byte.h"

// --- UTF-8 Byte Operations ---

int8_t utf8_byte_width(const uint8_t* start) {
    if (!start) {
        return -1;
    }

    uint8_t lead_byte = *start;
    if ((lead_byte & 0x80) == 0x00) {
        return 1;
    } else if ((lead_byte & 0xE0) == 0xC0) {
        return 2;
    } else if ((lead_byte & 0xF0) == 0xE0) {
        return 3;
    } else if ((lead_byte & 0xF8) == 0xF0) {
        return 4;
    }

    return -1; // Invalid lead byte
}

// Decode a UTF-8 byte sequence into a codepoint without validation.
int32_t utf8_byte_decode(const uint8_t* start) {
    switch (utf8_byte_width(start)) {
        case 1:
            return start[0];
        case 2:
            return ((start[0] & 0x1F) << 6) | (start[1] & 0x3F);
        case 3:
            return ((start[0] & 0x0F) << 12) | ((start[1] & 0x3F) << 6) | (start[2] & 0x3F);
        case 4:
            return ((start[0] & 0x07) << 18) | ((start[1] & 0x3F) << 12) | ((start[2] & 0x3F) << 6)
                   | (start[3] & 0x3F);
        default:
            return -1;
    }
}

bool utf8_byte_is_valid(const uint8_t* start) {
    if (!start) {
        return false;
    }

    int8_t width = utf8_byte_width(start);
    if (width == -1) {
        return false; // Early exit
    }

    if (width == 1) {
        // Reject continuation bytes as standalone sequences
        if ((start[0] & 0xC0) == 0x80) {
            return false;
        }
        // ASCII (1-byte) characters are always valid
        return true;
    }

    // Validate continuation bytes for multi-byte characters
    for (int8_t i = 1; i < width; i++) {
        if ((start[i] & 0xC0) != 0x80) {
            return false; // Invalid continuation byte
        }
    }

    // Additional checks for overlongs, surrogates, and invalid ranges
    if (width == 2) {
        if (start[0] < 0xC2) {
            return false; // Overlong encoding
        }
    } else if (width == 3) {
        if (start[0] == 0xE0 && start[1] < 0xA0) {
            return false; // Overlong encoding
        }
        if (start[0] == 0xED && start[1] >= 0xA0) {
            return false; // Surrogate halves
        }
    } else if (width == 4) {
        if (start[0] == 0xF0 && start[1] < 0x90) {
            return false; // Overlong encoding
        }
        if (start[0] == 0xF4 && start[1] > 0x8F) {
            return false; // Above U+10FFFF
        }
    }

    // If all checks passed, the character is valid
    return true;
}

bool utf8_byte_is_equal(const uint8_t* a, const uint8_t* b) {
    if (!a || !b) {
        return false;
    }

    int8_t a_width = utf8_byte_width(a);
    int8_t b_width = utf8_byte_width(b);
    if (-1 == a_width || -1 == b_width || a_width != b_width) {
        return false;
    }

    for (int i = 0; i < a_width; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

ptrdiff_t utf8_byte_range(const uint8_t* start, const uint8_t* end) {
    if (!start || !end) {
        return -1;
    }
    return end - start;
}

void utf8_byte_dump(const uint8_t* start) {
    size_t i = 0;
    while (start[i]) {
        int8_t width = utf8_byte_width(&start[i]);
        if (-1 == width) {
            printf("Invalid byte width detected!\n");
            break;
        }
        if (!utf8_byte_is_valid(&start[i])) {
            printf("Invalid byte detected!\n");
            break;
        }

        printf("byte[%zu] = 0x%02X | ", i, start[i]);
        for (int64_t j = 7; j >= 0; --j) {
            printf("%c", (start[i] & (1 << j)) ? '1' : '0');
        }
        printf("\n");

        i += width;
    }
}

// --- UTF-8 Byte Types ---

bool utf8_byte_is_char(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    if (codepoint < 0x20 || (codepoint >= 0x80 && codepoint <= 0x9F) || codepoint > 0x03FF) {
        return false;
    }

    return true;
}

bool utf8_byte_is_digit(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    if (codepoint < 0x30 || codepoint > 0x39) {
        return false;
    }

    return true;
}

bool utf8_byte_is_alpha(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    if ((codepoint >= 0x41 && codepoint <= 0x5A) || (codepoint >= 0x61 && codepoint <= 0x7A)) {
        return true;
    }

    return false;
}

bool utf8_byte_is_alnum(const uint8_t* start) {
    return utf8_byte_is_alpha(start) || utf8_byte_is_digit(start);
}

bool utf8_byte_is_upper(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    return (codepoint >= 0x41 && codepoint <= 0x5A);
}

bool utf8_byte_is_lower(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    return (codepoint >= 0x61 && codepoint <= 0x7A);
}

bool utf8_byte_is_space(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    switch (codepoint) {
        case 0x20: // ' '
        case 0x09: // '\t'
        case 0x0A: // '\n'
        case 0x0D: // '\r'
            return true;
        default:
            return false;
    }
}

bool utf8_byte_is_punct(const uint8_t* start) {
    if (!utf8_byte_is_valid(start)) {
        return false;
    }

    int32_t codepoint = utf8_byte_decode(start);
    if (-1 == codepoint) {
        return false;
    }

    // Punctuation ranges and single points (based on ASCII table)
    if ((codepoint >= 0x21 && codepoint <= 0x2F) || // !"#$%&'()*+,-./
        (codepoint >= 0x3A && codepoint <= 0x3F) || // :;<=>?@
        (codepoint >= 0x5B && codepoint <= 0x5D) || // [\]
        (codepoint == 0x5F) ||                     // _
        (codepoint >= 0x7B && codepoint <= 0x7E)) { // {|}~
        return true;
    }

    return false;
}

// --- UTF-8 Byte Visitor ---

const uint8_t* utf8_byte_next(const uint8_t* current) {
    if (!current || '\0' == *current) {
        return NULL;
    }

    const int8_t width = utf8_byte_width(current);
    if (width < 1 || !utf8_byte_is_valid(current)) {
        return NULL;
    }

    const uint8_t* next = current + width;
    return (*next) ? next : NULL;
}

const uint8_t* utf8_byte_next_width(const uint8_t* current, int8_t* out_width) {
    if (!current || '\0' == *current) {
        return NULL;
    }

    int8_t width = utf8_byte_width(current);
    if (-1 == width || !utf8_byte_is_valid(current)) {
        *out_width = -1;
        return current + 1; // Skip bad byte
    }

    *out_width = width;
    return current + width;
}

const uint8_t* utf8_byte_prev(const uint8_t* start, const uint8_t* current) {
    if (!start || !current || current <= start) {
        return NULL;
    }

    // Walk backwards at most 3 bytes to locate lead byte
    const uint8_t* prev = current - 1;
    for (int i = 0; i < 4 && prev >= start; ++i, --prev) {
        int8_t width = utf8_byte_width(prev);
        if (width > 0 && prev + width == current && utf8_byte_is_valid(prev)) {
            return prev;
        }
    }

    return NULL;
}

const uint8_t* utf8_byte_prev_width(const uint8_t* start, const uint8_t* current, int8_t* out_width) {
    if (!start || !current || current <= start || !out_width) {
        return NULL;
    }

    // Walk backwards at most 3 bytes to locate lead byte
    const uint8_t* prev = current - 1;
    for (int i = 0; i < 4 && prev >= start; ++i, --prev) {
        int8_t width = utf8_byte_width(prev);

        if (1 > width) {
            continue; // Not a valid lead byte
        }

        // Check if this forms a valid sequence ending exactly at current
        if (prev + width == current && utf8_byte_is_valid(prev)) {
            *out_width = width;
            return prev;
        }
    }

    *out_width = -1;
    return NULL;
}

const uint8_t* utf8_byte_peek(const uint8_t* current, const size_t ahead) {
    const uint8_t* ptr = current;
    for (size_t i = 0; i < ahead && ptr && *ptr; ++i) {
        ptr = utf8_byte_next(ptr);
    }
    return ptr;
}

// --- UTF-8 Byte Iterator ---

void* utf8_byte_iterate(const char* start, UTF8ByteIterator callback, void* context) {
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
