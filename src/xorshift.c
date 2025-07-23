/**
 * @file src/xorshift.c
 */

#include "xorshift.h"

uint32_t xorshift_int32(uint64_t* state) {
    *state ^= *state >> 12; // shift right, then flip
    *state ^= *state << 25; // shift left, then flip
    *state ^= *state >> 27; // shift right, then flip
    return (*state * 0x2545F4914F6CDD1Dull) >> 32; // scale, then drop 32-bits
}

float xorshift_float(uint64_t* state) {
    return (xorshift_int32(state) >> 8) / 16777216.0f;
}
