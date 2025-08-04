/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/utf8/string.h
 * @brief ASCII and UTF-8 String API.
 *
 * High-level string operations for UTF-8 pointer-to-char processing.
 * - Low Level: Byte functions are prefixed with `utf8_cp_`.
 * - High Level: String functions are prefixed with `utf8_`.
 */

#ifndef UTF8_STRING_H
#define UTF8_STRING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

/** UTF-8 Operations */

// Validate entire string (logs on failure)
bool utf8_str_is_valid(const char* src);

#endif // UTF8_STRING_H
