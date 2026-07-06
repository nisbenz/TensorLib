#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;
static int current_failed = 0;

#define TEST(name) static void name(void)

#define RUN_TEST(name) do { \
    tests_run++; \
    current_failed = 0; \
    printf("  %-42s", #name); \
    name(); \
    if (!current_failed) { tests_passed++; printf("PASS\n"); } \
} while (0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL (line %d: %s)\n", __LINE__, #cond); \
        current_failed = 1; \
        return; \
    } \
} while (0)

#define ASSERT_EQ_INT(a, b) do { \
    long _a = (long)(a), _b = (long)(b); \
    if (_a != _b) { \
        printf("FAIL (line %d: %s != %s -> %ld vs %ld)\n", __LINE__, #a, #b, _a, _b); \
        current_failed = 1; \
        return; \
    } \
} while (0)

#define ASSERT_EQ_FLOAT(a, b) do { \
    float _a = (float)(a), _b = (float)(b); \
    if (fabsf(_a - _b) > 1e-5f) { \
        printf("FAIL (line %d: %s != %s -> %f vs %f)\n", __LINE__, #a, #b, _a, _b); \
        current_failed = 1; \
        return; \
    } \
} while (0)

#define ASSERT_NULL(p)     ASSERT_TRUE((p) == NULL)
#define ASSERT_NOT_NULL(p) ASSERT_TRUE((p) != NULL)

#define TEST_SUITE_SUMMARY() \
    printf("\n%d/%d tests passed\n", tests_passed, tests_run); \
    return tests_passed == tests_run ? 0 : 1;

#endif
