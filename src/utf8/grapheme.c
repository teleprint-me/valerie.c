/**
 * @file src/utf8/grapheme.c
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

#include "posix.h"  // IWYU pragma: keep
#include "memory.h"
#include "utf8/codepoint.h"
#include "utf8/grapheme-data.h"
#include "utf8/grapheme.h"
#include <stdint.h>
#include <string.h>  // only for mem*() functions
#include <stdio.h>

/**
 * @note The database is automated and then compiled from source into a
 *       lookup table and then tested to see if a codepoint falls between
 *       a lo and hi range. If a classification is matched, it is returned.
 *       If a classification is unmatched, undefined is returned instead.
 *       Some classifications require:
 *          - Unit: (0020)
 *          - Tuple: (0023 FE0F)
 *          - Range: (0000..0008)
 *       Others are individual codepoints. e.g.
 *          - Identifying whitespace.
 *          - Whether a joiner is available as a specific case.
 *          - Etc.
 * @todo Attempt to enable O(1) lookup times.
 * @note Collation is required to apply sorting (UTS 10).
 */
UTF8GraphemeClass utf8_gcb_class(uint32_t cp) {
    for (size_t i = 0; i < UTF8_GRAPHEME_SIZE; i++) {
        if (cp >= graphemes[i].lo && cp <= graphemes[i].hi) {
            return graphemes[i].cls;
        }
    }

    // Default
    return GCB_UNDEFINED;
}

bool utf8_gcb_is_break(UTF8GraphemeBuffer* gb, int32_t cp) {
    UTF8GraphemeClass prev = utf8_gcb_class(gb->cp[0]);
    UTF8GraphemeClass curr = utf8_gcb_class(cp);

    // GB3: CR × LF
    if (prev == GCB_CR && curr == GCB_LF) {
        return false;
    }

    // GB4: (Control | CR | LF) ÷
    if (prev == GCB_CR || prev == GCB_LF || prev == GCB_CONTROL) {
        return true;
    }

    // GB5: ÷ (Control | CR | LF)
    if (curr == GCB_CR || curr == GCB_LF || curr == GCB_CONTROL) {
        return true;
    }

    // GB9: × Extend
    if (curr == GCB_EXTEND) {
        return false;
    }

    // GB9a: × ZWJ
    if (curr == GCB_ZWJ || prev == GCB_ZWJ) {
        return false;
    }

    // GB12/13: Do not break between pairs of Regional_Indicator
    if (prev == GCB_REGIONAL_INDICATOR && curr == GCB_REGIONAL_INDICATOR) {
        // Count how many previous consecutive RIs, including prev
        size_t ri_count = 0;
        for (size_t i = 1; i < gb->count; i++) {
            UTF8GraphemeClass c = utf8_gcb_class(gb->cp[i]);
            if (c == GCB_REGIONAL_INDICATOR) {
                ri_count++;
            } else {
                break;
            }
        }
        // Glue as pairs (even count = glue, odd = break)
        if (ri_count % 2 == 0) {
            return false;  // glue two RIs as a flag
        }
        return true;  // break between flags
    }

    // GB999: break everywhere else
    return true;
}

// Insert new codepoint at front (shift right)
void utf8_gcb_buffer_push(UTF8GraphemeBuffer* gb, uint32_t cp) {
    if (gb->count < UTF8_GCB_COUNT) {
        gb->count++;
    }

    // Shift all right (buf[0] = newest)
    for (size_t i = gb->count - 1; i > 0; i--) {
        gb->cp[i] = gb->cp[i - 1];
    }
    gb->cp[0] = cp;
}

int64_t utf8_gcb_count(const char* src) {
    if (!src) {
        return -1;
    }

    if (!*src) {
        return 0;
    }

    const uint8_t* stream = (const uint8_t*) src;

    int64_t count = 0;
    UTF8GraphemeBuffer gb = {0};
    bool first = true;

    while (*stream) {
        int8_t width = utf8_cp_width(stream);
        if (-1 == width || !utf8_cp_is_valid(stream)) {
            return -1;
        }

        uint32_t cp = utf8_cp_decode(stream);
        if (first || utf8_gcb_is_break(&gb, cp)) {
            first = false;
            count++;
        }

        utf8_gcb_buffer_push(&gb, cp);
        stream += width;
    }

    return count;
}

UTF8GraphemeIter utf8_gcb_iter(const char* start) {
    return (UTF8GraphemeIter) {
        .current = start,
        .first = true,
    };
}

const char* utf8_gcb_iter_next(UTF8GraphemeIter* it) {
    if (!it || !it->current || !*it->current) {
        return NULL;
    }

    const uint8_t* stream = (const uint8_t*) it->current;
    size_t cluster_len = 0;
    size_t offset = 0;

    memset(it->buffer, 0, UTF8_GCB_SIZE);

    while (stream[offset]) {
        int8_t width = utf8_cp_width(&stream[offset]);
        if (width < 1) {
            break;  // invalid byte
        }

        uint32_t cp = utf8_cp_decode(&stream[offset]);
        if (offset != 0 && utf8_gcb_is_break(&it->gb, cp)) {
            break;
        }

        if (cluster_len + width < UTF8_GCB_SIZE) {
            memcpy(&it->buffer[cluster_len], &stream[offset], width);
            cluster_len += width;
        } else {
            break;  // overflow and truncate
        }

        utf8_gcb_buffer_push(&it->gb, cp);
        offset += width;
    }

    if (cluster_len == 0) {
        return NULL;
    }

    it->buffer[cluster_len] = '\0';
    it->current += offset;

    // end of string
    if (!*it->current) {
        it->current = NULL;
    }

    return it->buffer;
}

char** utf8_gcb_split(const char* src, size_t* capacity) {
    if (!src || !*src || !capacity) {
        return NULL;
    }

    const uint8_t* stream = (const uint8_t*) src;

    // Get the literal byte length
    size_t len = 0;
    while (stream[len]) {
        len++;
    }

    *capacity = 0;
    char** parts = memory_alloc(sizeof(char*), alignof(char*));

    UTF8GraphemeBuffer gb = {0};

    size_t cluster_start = 0;
    for (size_t i = 0; i < len;) {
        int8_t width = utf8_cp_width(&stream[i]);
        if (width < 1) {
            break;
        }

        uint32_t curr_cp = utf8_cp_decode(&stream[i]);

        if (i != 0 && utf8_gcb_is_break(&gb, curr_cp)) {
            // End of previous cluster
            size_t cluster_len = i - cluster_start;
            char* cluster = memory_alloc(cluster_len + 1, alignof(char));
            memcpy(cluster, &stream[cluster_start], cluster_len);
            cluster[cluster_len] = '\0';

            parts = memory_realloc(
                parts, sizeof(char*) * (*capacity), sizeof(char*) * (*capacity + 1), alignof(char*)
            );
            parts[(*capacity)++] = cluster;
            cluster_start = i;
        }

        utf8_gcb_buffer_push(&gb, curr_cp);
        i += width;
    }

    // Copy final cluster
    if (cluster_start < len) {
        size_t cluster_len = len - cluster_start;
        char* cluster = memory_alloc(cluster_len + 1, alignof(char));
        memcpy(cluster, &stream[cluster_start], cluster_len);
        cluster[cluster_len] = '\0';

        parts = memory_realloc(
            parts, sizeof(char*) * (*capacity), sizeof(char*) * (*capacity + 1), alignof(char*)
        );
        parts[(*capacity)++] = cluster;
    }

    return parts;
}

void utf8_gcb_split_free(char** parts, size_t capacity) {
    utf8_cp_split_free((uint8_t**) parts, capacity);
}

void utf8_gcb_split_dump(char** parts, size_t capacity) {
    for (size_t i = 0; i < capacity; i++) {
        const uint8_t* cluster = (const uint8_t*) parts[i];
        printf("%s ", cluster);

        // First pass: Print codepoints
        const uint8_t* c = cluster;
        int total_len = 0;

        while (*c) {
            int8_t width = utf8_cp_width(c);
            if (width < 1) {
                break;
            }
            printf("[U+%04X | %d] ", utf8_cp_decode(c), width);
            c += width;
            total_len += width;
        }

        // Second pass: Print widths
        printf("[%d bytes]\n", total_len);
    }
}
