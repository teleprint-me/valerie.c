/**
 * @file src/utf8/regex.c
 */

#include "regex.h"

bool utf8_regex_compile(const uint8_t* pattern, pcre2_code** code, pcre2_match_data** match) {
    if (!pattern || !code || !match) {
        return false;
    }

    int error_code;
    PCRE2_SIZE error_offset;
    PCRE2_UCHAR8 error_message[256];

    *code = pcre2_compile(
        (PCRE2_SPTR) pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &error_code,
        &error_offset,
        NULL
    );
    if (!*code) {
        pcre2_get_error_message(error_code, error_message, sizeof(error_message));
        return false;
    }

    *match = pcre2_match_data_create_from_pattern(*code, NULL);
    if (!*match) {
        pcre2_code_free(*code);
        *code = NULL;
        return false;
    }

    return true;
}

void utf8_regex_free(pcre2_code* code, pcre2_match_data* match) {
    if (match) {
        pcre2_match_data_free(match);
    }

    if (code) {
        pcre2_code_free(code);
    }
}
