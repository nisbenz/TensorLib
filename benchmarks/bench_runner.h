#ifndef TENSORLIB_BENCH_RUNNER_H
#define TENSORLIB_BENCH_RUNNER_H

#include <stdio.h>

#include "bench_harness.h"

#define BENCH_MAX_THREADS 16

typedef struct {
    bench_profile profile;
    const char* suite;
    int threads[BENCH_MAX_THREADS];
    int thread_count;
} bench_options;

int bench_run_suites(const bench_options* options, FILE* csv);

#endif
