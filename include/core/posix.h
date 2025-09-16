/**
 * @file core/posix.h
 * @brief Fallback utilities for language and compiler compatibility (POSIX/C99/C11)
 *
 * This header provides:
 * - Boolean type compatibility for pre-C99
 * - Fallbacks for thread-local storage
 * - Alignment utilities for C11 and GCC/Clang
 *
 * Only defines features if they are not already available.
 */

#ifndef POSIX_H
#define POSIX_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h> // POSIX threading is preferred

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name Language Compatibility Fallbacks
 * @{
 */

// C99 Boolean support
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #include <stdbool.h>
#else
    #ifndef __bool_true_false_are_defined
        #define bool _Bool
        #define true 1
        #define false 0
        #define __bool_true_false_are_defined 1
    #endif
#endif

// Fallback for _Thread_local (introduced in C11)
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
    #ifndef _Thread_local
        #if defined(__GNUC__) || defined(__clang__)
            #define _Thread_local __thread
        #endif
    #endif

    // Standard alias for thread-local storage
    #ifndef thread_local
        #define thread_local _Thread_local
    #endif
#endif

/** @} */

/**
 * @name Compiler Feature Fallbacks
 * @{
 */

// C11 alignment support
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
#else
    #ifndef alignas
        #define alignas _Alignas
    #endif
    #ifndef alignof
        #define alignof _Alignof
    #endif
#endif

// Common max alignment constant
#ifndef MAX_ALIGN
    #define MAX_ALIGN alignof(max_align_t)
#endif

// Struct-level default alignment attribute (GCC/Clang)
#if !defined(MAX_ALIGN_ATTR) && (defined(__GNUC__) || defined(__clang__))
    #define MAX_ALIGN_ATTR __attribute__((aligned(MAX_ALIGN)))
#endif

/** @} */

#ifdef __cplusplus
}
#endif

#endif // POSIX_H
