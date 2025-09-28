/**
 * @file      lehmer.h
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

#ifndef LEHMER_H
#define LEHMER_H

#include <stdint.h>

/**
 * @def _Thread_local
 * @brief Compatibility definition for compilers lacking C11 `_Thread_local`
 */
#if defined(__GNUC__) && !defined(_Thread_local)
    #define _Thread_local __thread
#endif

/**
 * @def thread_local
 * @brief Portable alias for C11/C23 thread-local storage keyword
 */
#ifndef thread_local
    #define thread_local _Thread_local
#endif

/**
 * @def LEHMER_MODULUS
 * @brief Mersenne prime modulus (2^31 - 1)
 */
#define LEHMER_MODULUS 2147483647

/**
 * @def LEHMER_MULTIPLIER
 * @brief Park-Miller multiplier
 */
#define LEHMER_MULTIPLIER 48271

/**
 * @def LEHMER_SEED
 * @brief Default seed value
 */
#define LEHMER_SEED 123456789

/**
 * @struct LehmerState
 * @brief Represents the internal state of the Lehmer RNG
 *
 * Each thread has its own instance via `thread_local`.
 */
typedef struct LehmerState {
    int64_t seed; /**< Current raw integer seed/state */
    double norm; /**< Normalized output in [0.0, 1.0) */
} LehmerState;

/**
 * @var lehmer_state
 * @brief Thread-local instance of the Lehmer RNG state
 *
 * Accessible if you want to inspect or modify state manually.
 */
extern thread_local LehmerState lehmer_state;

/**
 * @brief Initialize or reseed the current thread's RNG
 *
 * @param seed A positive integer seed (must be in [1, LEHMER_MODULUS-1]).
 *             If zero or negative, `LEHMER_SEED` will be used.
 */
void lehmer_init(int64_t seed);

/**
 * @brief Generate the next random 64-bit integer in the sequence
 * @return Random int64_t in the range [1, LEHMER_MODULUS - 1]
 */
int64_t lehmer_int64(void);

/**
 * @brief Generate the next random 32-bit integer in the sequence
 * @return Random int32_t in the range [1, LEHMER_MODULUS - 1] truncated
 */
int32_t lehmer_int32(void);

/**
 * @brief Generate a normalized random number in [0.0, 1.0)
 * @return A double-precision float in the range [0.0, 1.0)
 */
double lehmer_double(void);

/**
 * @brief Generate a normalized random number in [0.0, 1.0)
 * @return A single-precision float in the range [0.0, 1.0)
 */
float lehmer_float(void);

#endif  // LEHMER_H
