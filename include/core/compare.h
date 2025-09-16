/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/compare.h
 *
 * @brief Compare floating-point numbers with a given tolerance in pure C
 *
 * @note see 1.2: Epsilon-Delta Definition of a Limit for details
 * https://math.libretexts.org/Bookshelves/Calculus/Calculus_3e_(Apex)
 */

#ifndef COMPARE_H
#define COMPARE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Math Constants */
#define PI 3.14159265358979323846f
#define SQRT_2 1.41421356237309504880f
#define SQRT_2_PI 0.79788456080286535588f

/** Epsilon */
#define EPSILON_DOUBLE 1e-15
#define EPSILON_SINGLE 1e-7f

/** Inline type-safe functions */
static inline int min_int(int a, int b) { return a < b ? a : b; }
static inline unsigned int min_uint(unsigned int a, unsigned int b) { return a < b ? a : b; }
static inline long min_long(long a, long b) { return a < b ? a : b; }
static inline unsigned long min_ulong(unsigned long a, unsigned long b) { return a < b ? a : b; }
static inline float min_float(float a, float b) { return a < b ? a : b; }
static inline double min_double(double a, double b) { return a < b ? a : b; }

static inline int max_int(int a, int b) { return a > b ? a : b; }
static inline unsigned int max_uint(unsigned int a, unsigned int b) { return a > b ? a : b; }
static inline long max_long(long a, long b) { return a > b ? a : b; }
static inline unsigned long max_ulong(unsigned long a, unsigned long b) { return a > b ? a : b; }
static inline float max_float(float a, float b) { return a > b ? a : b; }
static inline double max_double(double a, double b) { return a > b ? a : b; }

/** Type-generic dispatch macros using ISO C11 _Generic */
#define MIN(a, b) \
    _Generic((a), \
        int: min_int, \
        unsigned int: min_uint, \
        long: min_long, \
        unsigned long: min_ulong, \
        float: min_float, \
        double: min_double, \
        default: min_ulong \
    )(a, b)

#define MAX(a, b) \
    _Generic((a), \
        int: max_int, \
        unsigned int: max_uint, \
        long: max_long, \
        unsigned long: max_ulong, \
        float: max_float, \
        double: max_double, \
        default: max_ulong \
    )(a, b)

#define CLAMP(x, lo, hi) MAX((lo), MIN((x), (hi)))
#define MINMAX(x, a, b) CLAMP((x), MIN((a), (b)), MAX((a), (b)))
#define MIDPOINT(a, b) (((a) + (b)) / 2)

/**
 * @brief Determine if two double-precision floating-point numbers are close
 *        within a specified tolerance.
 *
 * @param a           The first floating-point number.
 * @param b           The second floating-point number.
 * @param significand The number of significant digits to consider (must be
 *                    in the range 1 to 15 inclusive). This determines the
 *                    absolute tolerance.
 *
 * @return            True if the numbers are close within the specified
 *                    tolerance, false otherwise.
 *
 * @note The significand is clamped if it is out of range.
 * @note EPSILON_DOUBLE affects relative tolerance.
 */
bool is_close_double(double a, double b, size_t significand);

/**
 * @brief Determine if two single-precision floating-point numbers are close
 *        within a specified tolerance.
 *
 * @param a           The first floating-point number.
 * @param b           The second floating-point number.
 * @param significand The number of significant digits to consider (must be
 *                    in the range 1 to 7 inclusive). This determines the
 *                    absolute tolerance.
 *
 * @return            True if the numbers are close within the specified
 *                    tolerance, false otherwise.
 *
 * @note The significand is clamped if it is out of range.
 * @note EPSILON_SINGLE affects relative tolerance.
 */
bool is_close_float(float a, float b, size_t significand);

#endif // COMPARE_H
