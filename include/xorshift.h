/**
 * @file include/xorshift.h
 */

#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <stdint.h>

/**
 * xorshift rng: generate next step in sequence [0, 2^64).
 * @ref https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
 */
uint32_t xorshift_int32(uint64_t* state);

/**
 * xorshift rng: normalize rng state [0, 1).
 */
float xorshift_float(uint64_t* state);

#endif // XORSHIFT_H
