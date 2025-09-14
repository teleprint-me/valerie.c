/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/test.h
 * @brief Minimal customizable unit testing framework for C.
 *
 * Provides structures and functions to define, run, and manage unit tests
 * with flexible test logic and callback hooks.
 */

#ifndef UNIT_TEST_H
#define UNIT_TEST_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>
#include <stdint.h>

/**
 * @name Forward Declarations
 * @{
 */

/**
 * @brief Structure representing a test unit.
 *
 * This structure is used to define individual unit tests,
 * including before, run, and after hooks.
 */
typedef struct TestUnit TestUnit;

/**
 * @brief Structure representing a test group.
 *
 * This structure is used to group multiple unit tests together,
 * allowing for easy execution and management.
 */
typedef struct TestGroup TestGroup;

/**
 * @brief Structure representing a test suite.
 *
 * This structure is used to manage a collection of test groups,
 * allowing for easy execution.
 */
typedef struct TestSuite TestSuite;

/** @} */

/**
 * @name Macros
 * @{
 */

/**
 * @brief Asserts a condition in a test and logs an error if false.
 *
 * If the condition is false, logs the formatted error message and
 * returns 1 to indicate test failure.
 *
 * @param condition Condition to assert.
 * @param format printf-style format string for the error message.
 * @param ... Arguments for the format string.
 */
#define ASSERT(condition, format, ...) \
    do { \
        if (!(condition)) { \
            LOG_ERROR(format, ##__VA_ARGS__); \
            return 1; \
        } \
    } while (0)

/** @} */

/**
 * @name Function Pointers
 * @{
 */

/**
 * @brief Function pointer for individual unit tests.
 *
 * Functions implementing a logical unit test should take a pointer to a TestUnitHook
 * and return 0 for success, non-zero for failure.
 */
typedef int (*TestUnitHook)(TestUnit* unit);

/**
 * @brief Function pointer for group-level setup/teardown hooks.
 *
 * Called before or after all unit tests in the group, with the group as context.
 * Returns 0 on success, non-zero on failure.
 */
typedef int (*TestGroupHook)(TestGroup* group);

/**
 * @brief Function pointer for suite-level test functions.
 *
 * Test suites run a series of tests and return 0 on success, non-zero otherwise.
 */
typedef int (*TestSuiteHook)(void);

/** @} */

/**
 * @name Structures
 * @{
 */

/**
 * @brief Represents a individual unit test.
 */
typedef struct TestUnit {
    const void* data; /**< User-defined parameters. */
    size_t index; /**< Index number of the test unit (1-based). */
    int8_t result; /**< Result of the test unit (0 = success, 1 = failure). */
} TestUnit;

/**
 * @brief Context for running a group of tests.
 */
typedef struct TestGroup {
    const char* name; /**< Group name. */
    const void* shared;
    TestUnit* units; /**< Array of unit tests. */
    size_t count; /**< Number of unit tests. */

    TestUnitHook run; /**< Hook to run a single test. */
    TestUnitHook before_each; /**< Hook to run before each test. */
    TestUnitHook after_each; /**< Hook to run after each test. */
    TestGroupHook before_all; /**< Hook to run before all tests. */
    TestGroupHook after_all; /**< Hook to run after all tests. */
} TestGroup;

/**
 * @brief Represents a named test suite.
 */
typedef struct TestSuite {
    const char* name; /**< Name of the test suite. */
    TestSuiteHook run; /**< Function to run the test suite. */
} TestSuite;

/** @} */

/**
 * @name Public Functions
 * @{
 */

/**
 * @brief Runs a group of unit tests.
 *
 * Executes the provided TestUnitHook function on each TestUnit.
 * Optionally invokes a callback before and or after each test.
 * Logs results and returns 0 if all tests pass, 1 otherwise.
 *
 * @param group Pointer to the TestGroup defining the tests.
 * @return 0 if all tests pass, 1 if any fail, -1 on invalid input.
 */
int test_group_run(TestGroup* group);

/**
 * @brief Runs a named test suite.
 *
 * Logs start and completion status of the suite.
 *
 * @param suite Pointer to the TestSuite defining the tests.
 * @return 0 on success, non-zero on failure.
 */
int test_suite_run(TestSuite* suite);

/** @} */

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // UNIT_TEST_H
