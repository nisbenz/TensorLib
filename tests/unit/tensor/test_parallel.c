#define _POSIX_C_SOURCE 200809L
#include "../../../src/tensor/parallel.h"
#include "test_common.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int configured;

static int expected(int threads)
{
#ifdef _OPENMP
    return threads;
#else
    (void)threads;
    return 1;
#endif
}

static int resolve(tensorlib_parallel_kind kind, long long work, int tasks)
{
    return tensorlib_parallel_threads_for(kind, work, 100, tasks);
}

TEST(test_boundaries_and_inheritance)
{
    const int caps[] = {3, 4, 5, 1, 6, 2, 4, 7, 1, 3};
    const int limits[] = {20, 30, 40, 10, 20, 4096, 30, 40, 40, 20};
    for (int index = 0; index < TENSORLIB_PARALLEL_COUNT; ++index) {
        tensorlib_parallel_kind kind = (tensorlib_parallel_kind)index;
        int cap = configured ? caps[index] : 8;
        int limit = configured ? limits[index] : 100;
        ASSERT_EQ_INT(resolve(kind, limit - 1, 0), 1);
        ASSERT_EQ_INT(resolve(kind, limit, 0), expected(cap));
        ASSERT_EQ_INT(resolve(kind, limit, 2), expected(cap < 2 ? cap : 2));
    }
#ifdef _OPENMP
    omp_set_num_threads(2);
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_PACK_RHS, 10000, 0), 2);
    omp_set_num_threads(8);
#endif
}

TEST(test_nested_work_stays_serial)
{
    int wrong = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(2) reduction(+:wrong)
#endif
    {
        if (resolve(TENSORLIB_PARALLEL_MATMUL, 10000, 0) != 1) ++wrong;
    }
    ASSERT_EQ_INT(wrong, 0);
}

/* Report initialization is allowed inside an OpenMP team; kernel resolution
 * there deliberately returns serial before initialization. */
TEST(test_concurrent_first_initialization)
{
    int wrong = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(8) reduction(+:wrong)
#endif
    {
        FILE* stream = tmpfile();
        if (stream == NULL) {
            ++wrong;
        } else {
            char report[2048];
            tensorlib_parallel_report(stream);
            rewind(stream);
            size_t count = fread(report, 1, sizeof(report) - 1, stream);
            report[count] = '\0';
#ifdef _OPENMP
            if (strstr(report, "MEMORY: threads=0 min_elements=0") == NULL)
                ++wrong;
#else
            if (strstr(report, "serial") == NULL) ++wrong;
#endif
            fclose(stream);
        }
    }
    ASSERT_EQ_INT(wrong, 0);
}

TEST(test_invalid_settings_inherit)
{
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_CLONE, 19, 0), 1);
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_CLONE, 20, 0), expected(3));
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_PACK_RHS, 20, 0), expected(3));
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_COMPUTE, 99, 0), 1);
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_COMPUTE, 100, 0), expected(8));
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_GELU, 100, 0), expected(8));
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_ADAMW, 20, 0), expected(3));
}

TEST(test_large_valid_limits)
{
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_MEMORY, LLONG_MAX - 1, 0), 1);
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_MEMORY, LLONG_MAX, 0), expected(8));
}

TEST(test_environment_is_cached)
{
#ifdef _WIN32
    ASSERT_EQ_INT(_putenv_s("TENSORLIB_MEMORY_THREADS", "1"), 0);
#else
    ASSERT_EQ_INT(setenv("TENSORLIB_MEMORY_THREADS", "1", 1), 0);
#endif
    ASSERT_EQ_INT(resolve(TENSORLIB_PARALLEL_MEMORY, 100, 0), expected(8));
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "defaults";
    configured = strcmp(mode, "configured") == 0;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#endif
    if (strcmp(mode, "invalid") == 0) {
        RUN_TEST(test_invalid_settings_inherit);
    } else if (strcmp(mode, "large") == 0) {
        RUN_TEST(test_large_valid_limits);
    } else {
        if (strcmp(mode, "concurrent") == 0)
            RUN_TEST(test_concurrent_first_initialization);
        RUN_TEST(test_boundaries_and_inheritance);
        if (strcmp(mode, "cache") == 0) RUN_TEST(test_environment_is_cached);
    }
    RUN_TEST(test_nested_work_stays_serial);
    TEST_SUITE_SUMMARY();
}
