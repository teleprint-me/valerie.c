// @file examples/split.c
// @brief driver for smoke testing utf8 byte ops
#include "memory.h"
#include "utf8/byte.h"
#include <stdio.h>

bool utf8_byte_split_dump(uint8_t** parts, uint8_t n_parts) {
    for (uint64_t i = 0; i < n_parts; i++) {
        int64_t n_bytes = utf8_byte_count(parts[i]);
        if (-1 == n_bytes) {
            return false;
        }

        printf("part='%s', bytes=", parts[i]);  // should be one byte per entry
        for (int64_t j = 0; j < n_bytes; j++) {
            printf("%04x ", parts[i][j]);  // should be one byte per entry
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
    return true;
}

bool utf8_byte_join_dump(uint8_t* src) {
    int64_t n_bytes = utf8_byte_count(src);
    if (-1 == n_bytes) {
        return false;
    }

    printf("src='%s', bytes=", src);
    for (int64_t i = 0; i < n_bytes; i++) {
        printf("%04x ", src[i]);  // should be one byte per entry
    }
    printf("\n");
    fflush(stdout);
    return true;
}

int main(void) {
    const uint8_t text[] = "Hello, world!";

    uint64_t n_parts = 0;
    uint8_t** parts = utf8_byte_split(text, &n_parts);
    if (!parts) {
        fprintf(stderr, "Failed to split input text!\n");
        return 1;
    }

    if (!utf8_byte_split_dump(parts, n_parts)) {
        fprintf(stderr, "Failed to dump split text!\n");
        utf8_byte_split_free(parts, n_parts);
        return 1;
    }

    uint8_t* joined = utf8_byte_join(parts, n_parts, (const uint8_t*) " ");
    if (!joined) {
        fprintf(stderr, "Failed to join split text!\n");
        utf8_byte_split_free(parts, n_parts);
        return 1;
    }

    if (!utf8_byte_join_dump(joined)) {
        fprintf(stderr, "Failed to dump joined text!\n");
        memory_free(joined);
        utf8_byte_split_free(parts, n_parts);
        return 1;
    }

    memory_free(joined);
    utf8_byte_split_free(parts, n_parts);
    return 0;
}
