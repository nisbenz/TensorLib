#define _POSIX_C_SOURCE 200809L
#include "../../../src/tensor/parallel.h"
#include "test_common.h"
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

int main(int argc, char** argv)
{
    configured = argc > 1 && strcmp(argv[1], "configured") == 0;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#endif
    RUN_TEST(test_boundaries_and_inheritance);
    RUN_TEST(test_nested_work_stays_serial);
    TEST_SUITE_SUMMARY();
}
