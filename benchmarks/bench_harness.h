#ifndef TENSORLIB_BENCH_HARNESS_H
#define TENSORLIB_BENCH_HARNESS_H

#include <stddef.h>

typedef int (*bench_operation)(void* context, double* checksum);
typedef void (*bench_measure_reset)(void* context);

typedef struct {
    const char* profile;
    int sample_count;
    double minimum_sample_seconds;
    double warmup_seconds;
} bench_profile;

typedef struct {
    double median_seconds;
    double p95_seconds;
    double checksum;
    int iterations_per_sample;
} bench_measurement;

double bench_now_seconds(void);
bench_profile bench_profile_named(const char* name);
int bench_measure(bench_operation operation,
                  void* context,
                  const bench_profile* profile,
                  bench_measurement* result);
int bench_measure_with_reset(bench_operation operation,
                             void* context,
                             const bench_profile* profile,
                             bench_measure_reset reset,
                             bench_measurement* result);

#endif
