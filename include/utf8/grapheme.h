/**
 * @file examples/grapheme.c
 * @ref https://www.unicode.org/reports/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/emoji/
 */

#ifndef UTF8_GRAPHEME_H

#include "posix.h" // IWYU pragma: keep
#include "utf8/grapheme-data.h"
#include <stdint.h>
#include <string.h> // only for mem*() functions
#include <stdio.h>

#define GRAPHEME_BUFFER_MAX 8

// sliding window of seen codepoints
typedef struct GraphemeBuffer {
    uint32_t cp[GRAPHEME_BUFFER_MAX]; // previous codepoints
    size_t count; // number of valid codepoints
} GraphemeBuffer;

GraphemeClass utf8_gcb_class(uint32_t cp);
bool utf8_gcb_is_break(GraphemeBuffer* gb, int32_t cp);
void utf8_gcb_buffer_push(GraphemeBuffer* gb, uint32_t cp);

int64_t utf8_gcb_count(const char* src);

char** utf8_gcb_split(const char* src, size_t* capacity);
void utf8_gcb_split_free(char** parts, size_t capacity);
void utf8_gcb_split_dump(char** parts, size_t capacity);

#endif // UTF8_GRAPHEME_H
