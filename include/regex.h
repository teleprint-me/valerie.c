/**
 * @file include/utf8/regex.h
 */

#ifndef UTF8_REGEX_H
#define UTF8_REGEX_H

#include <stdbool.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

/**
 * @brief Compiles a UTF-8 regex pattern and creates a PCRE2 match data object.
 *
 * @param pattern Null-terminated UTF-8 regex pattern.
 * @param code    Output: set to compiled PCRE2 code object on success.
 * @param match   Output: set to PCRE2 match data object on success.
 * @return        true on success, false on error (code and match set to NULL).
 *
 * Caller must free both objects when done.
 */
bool utf8_regex_compile(const uint8_t* pattern, pcre2_code** code, pcre2_match_data** match);

/**
 * @brief Frees PCRE2 regex code and match data objects.
 *
 * Safe to call with NULL arguments.
 *
 * @param code  Compiled PCRE2 code object (may be NULL).
 * @param match PCRE2 match data object (may be NULL).
 */
void utf8_regex_free(pcre2_code* code, pcre2_match_data* match);

#endif  // UTF8_REGEX_H
