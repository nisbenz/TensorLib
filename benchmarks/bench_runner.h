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

typedef struct {
    const char* suite;
    const char* name;
    const char* shape;
    const char* layout;
    const char* metric;
    double units_per_call;
    int parallel;
    bench_operation operation;
    void* context;
} bench_case;

int bench_run_suites(const bench_options* options, FILE* csv);
int bench_execute_case(const bench_options* options,
                       FILE* csv,
                       const bench_case* benchmark,
                       int requested_threads,
                       bench_measurement* result);

/* Writes a measurement already collected by a custom benchmark. */
int bench_record_measurement(const bench_options* options,
                             FILE* csv,
                             const bench_case* benchmark,
                             int requested_threads,
                             int measured_threads,
                             const bench_measurement* result);

#endif
