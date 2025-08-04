/**
 * @file src/utf8/byte.c
 */

#include "utf8/byte.h"

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
