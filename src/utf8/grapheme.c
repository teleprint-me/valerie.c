/**
 * @file examples/grapheme.c
 * @ref https://www.unicode.org/reports/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/
 * @ref https://www.unicode.org/Public/UCD/latest/ucd/emoji/
 */

#include "posix.h"  // IWYU pragma: keep
#include "memory.h"
#include "utf8/byte.h"
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
 */
GraphemeClass utf8_gcb_class(uint32_t cp) {
    for (size_t i = 0; i < GRAPHEME_SIZE; i++) {
        if (cp >= graphemes[i].lo && cp <= graphemes[i].hi) {
            return graphemes[i].cls;
        }
    }

    // Default
    return GCB_UNDEFINED;
}

bool utf8_gcb_is_break(GraphemeBuffer* gb, int32_t cp) {
    GraphemeClass prev = utf8_gcb_class(gb->cp[0]);
    GraphemeClass curr = utf8_gcb_class(cp);

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

    // GB9: × Spacing Mark
    if (curr == GCB_SPACINGMARK) {
        return false;
    }

    // GB9: × Prepend
    if (curr == GCB_PREPEND || prev == GCB_PREPEND) {
        return false;
    }

    if (curr == GCB_DIACRITIC) {
        return false;
    }

    // GB9a: × ZWJ
    if (curr == GCB_ZWJ || prev == GCB_ZWJ) {
        return false;
    }

    // GB11: EP_1 × ZWJ × EP_2 × ZWJ × ... × EP_N
    // Only apply GB11 if current is Extended_Pictographic
    if (curr == GCB_EXTENDED_PICTOGRAPHIC) {
        // Scan back through gb->cp[1]..gb->cp[GRAPHEME_BUF_MAX-1]
        for (size_t i = 1; i < gb->count; i++) {
            GraphemeClass c = utf8_gcb_class(gb->cp[i]);
            if (c == GCB_ZWJ || c == GCB_EXTEND) {
                continue;
            }
            if (c == GCB_EXTENDED_PICTOGRAPHIC) {
                return false;
            }
            break;
        }
    }

    // GB12/GB13: Do not break between emoji indicators (includes regional)
    // Only apply if prev and current are GCB_EMOJI
    if (prev == GCB_EMOJI && curr == GCB_EMOJI) {
        // Scan back through gb->cp[1]..gb->cp[GRAPHEME_BUF_MAX-1]
        size_t ri = 0;  // count consecutive pairs
        for (size_t i = 1; i < gb->count; i++) {
            // emojis may be a literal, component, or presentation
            GraphemeClass c = utf8_gcb_class(gb->cp[i]);
            if (c == GCB_EMOJI || c == GCB_EMOJI_PRESENTATION || c == GCB_EMOJI_COMPONENT) {
                ri++;
            } else {
                break;
            }
        }
        if (ri % 2 != 0) {
            return true;  // break on odd count (starting new flag)
        }
        return false;  // glue as a pair
    }

    // GB999: break everywhere else
    return true;
}

// Insert new codepoint at front (shift right)
void utf8_gcb_buffer_push(GraphemeBuffer* gb, uint32_t cp) {
    if (gb->count < GRAPHEME_BUFFER_MAX) {
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
    GraphemeBuffer gb = {0};
    bool first = true;

    while (*stream) {
        if (!utf8_byte_is_valid(stream)) {
            return -1;
        }

        int8_t width = utf8_byte_width(stream);
        if (1 > width) {
            return -1;
        }

        uint32_t cp = utf8_byte_decode(stream);
        if (first || utf8_gcb_is_break(&gb, cp)) {
            first = false;
            count++;
        }

        utf8_gcb_buffer_push(&gb, cp);
        stream += width;
    }

    return count;
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

    GraphemeBuffer gb = {0};

    size_t cluster_start = 0;
    for (size_t i = 0; i < len;) {
        int8_t width = utf8_byte_width(&stream[i]);
        if (width < 1) {
            break;
        }

        uint32_t curr_cp = utf8_byte_decode(&stream[i]);

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
    utf8_byte_split_free((uint8_t**) parts, capacity);
}

void utf8_gcb_split_dump(char** parts, size_t capacity) {
    for (uint32_t i = 0; i < capacity; i++) {
        const uint8_t* cluster = (const uint8_t*) parts[i];
        int8_t w = utf8_byte_width(cluster);
        printf("%s | U+%04X | width: %d\n", cluster, utf8_byte_decode(cluster), w);

        while(*cluster) {
            int8_t width = utf8_byte_width(cluster);
            printf("    U+%04X | width: %d\n", utf8_byte_decode(cluster), width);
            cluster += width;
        }
    }
}
