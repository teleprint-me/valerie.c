/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/test.c
 * @brief Minimal customizable unit testing framework for C.
 *
 * Provides structures and functions to define, run, and manage unit tests
 * with flexible test logic and callback hooks.
 */

#include "core/test.h"

int test_group_run(TestGroup* group) {
    if (!group || !group->name || !group->units || group->count == 0 || !group->run) {
        fprintf(stderr, "[TestGroup] Invalid parameters.\n");
        return -1;
    }

    fprintf(stdout, "[TestGroup] [RUN] %s: Number of tests: %zu\n", group->name, group->count);

    size_t failures = 0;

    if (group->before_all && group->before_all(group) != 0) {
        fprintf(stderr, "[TestGroup] [FAIL] %s: Group setup failed.\n", group->name);
        return 1;
    }

    for (size_t i = 0; i < group->count; i++) {
        int result = 0;

        TestUnit* unit = &group->units[i];
        unit->index = i + 1;

        if (group->before_each) { // group->before_each()
            result |= group->before_each(unit);
        }

        result |= group->run(unit);

        if (group->after_each) { // group->after_each()
            result |= group->after_each(unit);
        }

        if (result != 0) {
            failures++;
            fprintf(stderr, "[TestGroup] [FAIL] %s: Test case %zu failed.\n", group->name, unit->index);
        }
    }

    if (group->after_all && group->after_all(group) != 0) {
        fprintf(stderr, "[TestGroup] [FAIL] %s: Group teardown failed.\n", group->name);
        return 1;
    }

    size_t passed = group->count - failures;
    fprintf(stdout, "[TestGroup] [RESULT] %s: %zu/%zu tests passed\n", group->name, passed, group->count);

    return (failures) > 0 ? 1 : 0;
}

int test_suite_run(TestSuite* suite) {
    if (!suite || !suite->name || !suite->run) {
        fprintf(stderr, "[TestSuite] Invalid parameters.\n");
        return -1;
    }

    fprintf(stdout, "[TestSuite] [RUN] %s\n", suite->name);

    int result = suite->run();
    if (result == 0) {
        fprintf(stdout, "[TestSuite] [PASS] %s\n", suite->name);
    } else {
        fprintf(stderr, "[TestSuite] [FAIL] %s\n", suite->name);
    }
    return result;
}
