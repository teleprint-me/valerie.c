/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/compare.c
 *
 * @brief Compare floating-point numbers with a given tolerance in pure C
 *
 * @note The 64-bit implementation is reverse compatible with the 32-bit
 * implementation. Whatever is true for the 64-bit implementation must then be
 * true for the 32-bit implementation.
 */


#include <stdlib.h>
#include <math.h>

#include "core/compare.h"

// Pre-computed lookup table
static const double tolerance_table[16]
    = {1.0,
       0.1,
       0.01,
       0.001,
       0.0001,
       0.00001,
       0.000001,
       0.0000001,
       0.00000001,
       0.000000001,
       0.0000000001,
       0.00000000001,
       0.000000000001,
       0.0000000000001,
       0.00000000000001,
       0.000000000000001};

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
 * @note DOUBLE_EPSILON affects relative tolerance.
 */
bool is_close_double(double a, double b, size_t significand) {
    if (a == b) {
        return true;
    }

    if (isinf(a) || isinf(b) || isnan(a) || isnan(b)) {
        return false;
    }

    significand = CLAMP(significand, 1, 15);

    double absolute_tolerance = tolerance_table[significand];
    double relative_tolerance = EPSILON_DOUBLE * fmax(fabs(a), fabs(b));
    double difference = fabs(a - b);

    return difference <= fmax(relative_tolerance, absolute_tolerance);
}

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
 * @note SINGLE_EPSILON affects relative tolerance.
 */
bool is_close_float(float a, float b, size_t significand) {
    if (a == b) {
        return true;
    }

    if (isinf(a) || isinf(b) || isnan(a) || isnan(b)) {
        return false;
    }

    significand = CLAMP(significand, 1, 7);

    float absolute_tolerance = (float) tolerance_table[significand];
    float relative_tolerance = ((float) EPSILON_SINGLE) * fmaxf(fabsf(a), fabsf(b));
    float difference = fabsf(a - b);

    return difference <= fmaxf(relative_tolerance, absolute_tolerance);
}
