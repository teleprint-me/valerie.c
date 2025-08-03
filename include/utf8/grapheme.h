/**
 * @file include/utf8/grapheme.h
 * @brief ASCII and UTF-8 Grapheme API.
 *
 * Low-level API for handling core UTF-8 grapheme pre-processing.
 *
 * - A UTF-8 grapheme represents a valid ASCII or UTF-8 cluster of codepoints.
 * - All Library functions are prefixed with `utf8_`.
 * - Grapheme-level operations are prefixed with `utf8_gcb_`.
 *
 * @ref https://www.unicode.org/reports/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/
 */

#ifndef UTF8_GRAPHEME_H

#include "posix.h" // IWYU pragma: keep
#include "utf8/grapheme-data.h"
#include <stdint.h>
#include <string.h> // only for mem*() functions
#include <stdio.h>

#define UTF8_GCB_COUNT 8 // 32 bytes = 4 bytes * 8 bytes/codepoint
#define UTF8_GCB_SIZE UTF8_GCB_COUNT * 4 + 1 // max 4 bytes/codepoint + NULL char

// sliding window of seen codepoints
typedef struct UTF8GraphemeBuffer {
    uint32_t cp[UTF8_GCB_COUNT]; // previous codepoints
    size_t count; // number of valid codepoints
} UTF8GraphemeBuffer;

UTF8GraphemeClass utf8_gcb_class(uint32_t cp);
bool utf8_gcb_is_break(UTF8GraphemeBuffer* gb, int32_t cp);
void utf8_gcb_buffer_push(UTF8GraphemeBuffer* gb, uint32_t cp);

int64_t utf8_gcb_count(const char* src);

typedef struct UTF8GraphemeIter {
    const char* current;  // Current input pointer
    UTF8GraphemeBuffer gb; // codepoint history
    char buffer[UTF8_GCB_SIZE]; // current cluster
    bool first; // init cluster sequence
} UTF8GraphemeIter;

UTF8GraphemeIter utf8_gcb_iter(const char* start);
const char* utf8_gcb_iter_next(UTF8GraphemeIter* it);

char** utf8_gcb_split(const char* src, size_t* capacity);
void utf8_gcb_split_free(char** parts, size_t capacity);
void utf8_gcb_split_dump(char** parts, size_t capacity);

#endif // UTF8_GRAPHEME_H
