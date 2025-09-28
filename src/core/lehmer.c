/**
 * @file      lehmer.c
 * @brief     Thread-local Lehmer Random Number Generator (LCG, Park-Miller)
 * @copyright Copyright © 2023 Austin Berrio
 *
 * A minimal, fast, and thread-safe Lehmer (Park-Miller) pseudo-random number
 * generator using 32-bit safe arithmetic and thread-local storage.
 *
 * Based on:
 *  - "Random Number Generators: Good Ones Are Hard to Find" by Park & Miller (1988)
 *    @ref https://dl.acm.org/doi/10.1145/63039.63042
 *  - @ref https://www.cs.wm.edu/~va/software/park/park.html
 *
 * @note The RNG state is local to each thread using `thread_local`, which avoids
 * synchronization overhead. Functions are reentrant per-thread.
 *
 * @warning Not suitable for cryptographic purposes.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include <string.h>
#include <math.h>

#include "core/lehmer.h"

/**
 * Global State
 */

thread_local LehmerState lehmer_state = {
    .seed = LEHMER_SEED,
    .norm = 0.0,
};

/**
 * Private Functions
 */

static inline void lehmer_mod(void) {
    const int64_t q = LEHMER_MODULUS / LEHMER_MULTIPLIER;
    const int64_t r = LEHMER_MODULUS % LEHMER_MULTIPLIER;

    int64_t hi = lehmer_state.seed / q;
    int64_t lo = lehmer_state.seed % q;

    int64_t t = LEHMER_MULTIPLIER * lo - r * hi;
    lehmer_state.seed = (t > 0) ? t : t + LEHMER_MODULUS;
}

static inline void lehmer_norm(void) {
    lehmer_state.norm = (double) lehmer_state.seed / (double) LEHMER_MODULUS;
}

/**
 * Public Functions
 */

void lehmer_init(int64_t seed) {
    lehmer_state.seed = (seed > 0) ? seed : LEHMER_SEED;
}

int64_t lehmer_int64(void) {
    lehmer_mod();
    return lehmer_state.seed;
}

int32_t lehmer_int32(void) {
    return (int32_t) lehmer_int64();
}

double lehmer_double(void) {
    lehmer_mod();
    lehmer_norm();
    return lehmer_state.norm;
}

float lehmer_float(void) {
    return (float) lehmer_double();
}

// Xavier/Glorot uniform
float lehmer_xavier(size_t in, size_t out) {
    float a = sqrtf(6.0f / (in + out));
    float ud = 2.0f * lehmer_float() - 1.0f;
    return ud * a;
}

// Box-Muller normal
float xorshift_muller(size_t in, size_t out) {
    float u1 = lehmer_float();
    if (u1 < 1e-7f) {
        u1 = 1e-7f;  // avoid 0
    }
    float u2 = lehmer_float();
    if (u2 < 1e-7f) {
        u2 = 1e-7f;  // avoid 0
    }
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float) M_PI * u2);
    float stddev = sqrtf(2.0f / (in + out));
    return z0 * stddev;
}

// Fisher–Yates shuffle
bool xorshift_yates(void* base, size_t n, size_t size) {
    if (!base || n < 2) {
        return false;  // redundant
    }

    uint8_t* arr = (uint8_t*) base;
    uint8_t* tmp = (uint8_t*) malloc(size);
    if (!tmp) {
        return false;  // malloc failed
    }

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = lehmer_int32() % (i + 1);
        /// @note use memmove for safe overlapping memory
        memmove(tmp, arr + i * size, size);
        memmove(arr + i * size, arr + j * size, size);
        memmove(arr + j * size, tmp, size);
    }

    free(tmp);
    return true;
}
