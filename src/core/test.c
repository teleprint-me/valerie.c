/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/test.c
 * @brief Minimal customizable unit testing framework for C.
 *
 * Provides structures and functions to define, run, and manage unit tests
 * with flexible test logic and callback hooks.
 */

#include "core/logger.h"
#include "core/test.h"

int test_group_run(TestGroup* group) {
    if (!group || !group->name || !group->units || group->count == 0 || !group->run) {
        LOG_ERROR("[TestGroup] Invalid parameters.");
        return -1;
    }

    LOG_INFO("[TestGroup] [RUN] %s: Number of tests: %zu", group->name, group->count);

    size_t failures = 0;

    if (group->before_all && group->before_all(group) != 0) {
        LOG_ERROR("[TestGroup] [FAIL] %s: Group setup failed.", group->name);
        return 1;
    }

    for (size_t i = 0; i < group->count; i++) {
        int result = 0;

        TestUnit* unit = &group->units[i];
        unit->index = i + 1;

        if (group->before_each) {  // group->before_each()
            result |= group->before_each(unit);
        }

        result |= group->run(unit);

        if (group->after_each) {  // group->after_each()
            result |= group->after_each(unit);
        }

        if (result != 0) {
            failures++;
            LOG_ERROR("[TestGroup] [FAIL] %s: Test case %zu failed.", group->name, unit->index);
        }
    }

    if (group->after_all && group->after_all(group) != 0) {
        LOG_ERROR("[TestGroup] [FAIL] %s: Group teardown failed.", group->name);
        return 1;
    }

    size_t passed = group->count - failures;
    LOG_INFO("[TestGroup] [RESULT] %s: %zu/%zu tests passed", group->name, passed, group->count);

    return (failures) > 0 ? 1 : 0;
}

int test_suite_run(TestSuite* suite) {
    if (!suite || !suite->name || !suite->run) {
        LOG_ERROR("[TestSuite] Invalid parameters.");
        return -1;
    }

    LOG_INFO("[TestSuite] [RUN] %s", suite->name);

    int result = suite->run();
    if (result == 0) {
        LOG_INFO("[TestSuite] [PASS] %s", suite->name);
    } else {
        LOG_ERROR("[TestSuite] [FAIL] %s", suite->name);
    }
    return result;
}
