#define _POSIX_C_SOURCE 200809L

#include "bench_harness.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

double bench_now_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

bench_profile bench_profile_named(const char* name)
{
    bench_profile profile;
    if (name != NULL && strcmp(name, "full") == 0) {
        profile.profile = "full";
        profile.sample_count = 15;
        profile.minimum_sample_seconds = 0.1;
        profile.warmup_seconds = 0.25;
    } else if (name != NULL && strcmp(name, "smoke") == 0) {
        profile.profile = "smoke";
        profile.sample_count = 1;
        profile.minimum_sample_seconds = 0.0;
        profile.warmup_seconds = 0.0;
    } else {
        profile.profile = "quick";
        profile.sample_count = 5;
        profile.minimum_sample_seconds = 0.02;
        profile.warmup_seconds = 0.02;
    }
    return profile;
}

static int compare_double(const void* left, const void* right)
{
    double a = *(const double*)left;
    double b = *(const double*)right;
    return (a > b) - (a < b);
}

static int run_iterations(bench_operation operation,
                          void* context,
                          int iterations,
                          double* checksum,
                          double* seconds)
{
    double start = bench_now_seconds();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (operation(context, checksum) != 0) return 1;
    }
    *seconds = bench_now_seconds() - start;
    return 0;
}

int bench_measure(bench_operation operation,
                  void* context,
                  const bench_profile* profile,
                  bench_measurement* result)
{
    double* samples;
    double checksum = 0.0;
    double elapsed = 0.0;
    double warmup_elapsed = 0.0;
    int iterations = 1;

    if (operation == NULL || profile == NULL || result == NULL ||
        profile->sample_count <= 0) return 1;
    while (warmup_elapsed < profile->warmup_seconds) {
        if (run_iterations(operation, context, 1, &checksum, &elapsed) != 0) {
            return 1;
        }
        warmup_elapsed += elapsed;
    }
    if (run_iterations(operation, context, 1, &checksum, &elapsed) != 0) {
        return 1;
    }
    if (elapsed > 0.0 && elapsed < profile->minimum_sample_seconds) {
        double target = profile->minimum_sample_seconds / elapsed;
        iterations = (int)target + 1;
        if (iterations > 1000000) iterations = 1000000;
    }

    samples = (double*)malloc((size_t)profile->sample_count * sizeof(*samples));
    if (samples == NULL) return 1;
    checksum = 0.0;
    for (int sample = 0; sample < profile->sample_count; ++sample) {
        if (run_iterations(operation, context, iterations,
                           &checksum, &elapsed) != 0) {
            free(samples);
            return 1;
        }
        samples[sample] = elapsed / (double)iterations;
    }
    qsort(samples, (size_t)profile->sample_count,
          sizeof(*samples), compare_double);
    result->median_seconds = samples[profile->sample_count / 2];
    result->p95_seconds = samples[(95 * profile->sample_count - 1) / 100];
    result->checksum = checksum;
    result->iterations_per_sample = iterations;
    free(samples);
    return 0;
}
