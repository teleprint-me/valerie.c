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

#include <stdbool.h>

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
    long seed; /**< Current raw integer seed/state */
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
void lehmer_init(long seed);

/**
 * @brief Generate the next random 64-bit integer in the sequence
 * @return Random long in the range [1, LEHMER_MODULUS - 1]
 */
long lehmer_int64(void);

/**
 * @brief Generate the next random 32-bit integer in the sequence
 * @return Random int in the range [1, LEHMER_MODULUS - 1] truncated
 */
int lehmer_int32(void);

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

/**
 * @brief Generate a Xavier-Bengio scaled uniform distribution [-a, a].
 *
 * Uses the formula: a = sqrt(6 / (in + out))
 * Output is uniform in [-a, a].
 *
 * @param out Output dimension (fan-out)
 * @param in  Input dimension (fan-in)
 * @return 32-bit float uniform in [-a, a]
 */
float lehmer_xavier(unsigned out, unsigned in);

/**
 * @brief Generate a Xavier-Bengio scaled normal distribution using Box-Muller transform.
 *
 * Standard normal (mean=0, variance=1), scaled by sqrt(2 / (in + out)).
 *
 * @param out Output dimension (fan-out)
 * @param in  Input dimension (fan-in)
 * @return 32-bit float from N(0, stddev^2)
 */
float lehmer_muller(unsigned out, unsigned in);

/**
 * @brief In-place Fisher–Yates shuffle of an array.
 *
 * Shuffles a buffer of @p n elements, each of size @p size bytes.
 *
 * @param base Pointer to buffer to shuffle
 * @param n    Number of elements in buffer
 * @param size Size in bytes of each element
 * @return true on success, false on failure
 */
bool lehmer_yates(void* base, unsigned n, unsigned size);

#endif  // LEHMER_H
