/**
 * @file src/utf8/byte.c
 */

#include "memory.h"
#include "utf8/byte.h"
#include <stddef.h>
#include <string.h>

int64_t utf8_byte_count(const uint8_t* start) {
    if (!start) {
        return -1;  // Invalid pointer
    }

    int64_t count = 0;
    while (start[count]) {
        count++;
    }

    return count;
}

uint8_t* utf8_byte_copy(const uint8_t* start) {
    if (!start) {
        return NULL;
    }

    int64_t count = utf8_byte_count(start);
    if (-1 == count) {
        return NULL;
    }

    uint8_t* buffer = memory_alloc((count + 1) * sizeof(char), alignof(char));
    if (!buffer) {
        return NULL;
    }

    if (count > 0) {
        memcpy(buffer, start, count);
    }
    buffer[count] = '\0';

    return buffer;
}
