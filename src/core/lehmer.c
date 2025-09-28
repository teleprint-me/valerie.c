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
