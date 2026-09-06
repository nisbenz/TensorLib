#include "bench_runner.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <tensorlib/tensor_matmul.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int bench_configure_threads(int requested)
{
#ifdef _OPENMP
    int actual = 1;
    omp_set_dynamic(0);
    omp_set_num_threads(requested);
#pragma omp parallel
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    return actual;
#else
    (void)requested;
    return 1;
#endif
}

#ifndef _OPENMP
static void write_skipped(FILE* csv,
                          const bench_options* options,
                          const bench_case* benchmark,
                          int requested)
{
    printf("  %-24s threads=%d skipped (OpenMP disabled)\n",
           benchmark->name, requested);
    if (csv != NULL) {
        fprintf(csv, "%s,%s,%s,%s,%s,%d,1,0,0,%s,0,0,0,skipped\n",
                benchmark->suite, benchmark->name, benchmark->shape,
                benchmark->layout, options->profile.profile, requested,
                benchmark->metric);
    }
}
#endif

int bench_record_measurement(const bench_options* options,
                             FILE* csv,
                             const bench_case* benchmark,
                             int requested_threads,
                             int measured_threads,
                             const bench_measurement* result)
{
    double value;

    if (options == NULL || benchmark == NULL || result == NULL ||
        !isfinite(result->median_seconds) ||
        !isfinite(result->p95_seconds) ||
        !isfinite(result->checksum)) return 1;
    value = benchmark->units_per_call > 0.0
          ? benchmark->units_per_call / result->median_seconds
          : result->median_seconds * 1000.0;
    printf("  %-24s threads=%-3d median=%9.3f ms p95=%9.3f ms "
           "%s=%9.3f\n",
           benchmark->name, measured_threads, result->median_seconds * 1000.0,
           result->p95_seconds * 1000.0, benchmark->metric, value);
    if (csv != NULL) {
        fprintf(csv, "%s,%s,%s,%s,%s,%d,%d,%.9g,%.9g,%s,%.9g,%d,%.9g,ok\n",
                benchmark->suite, benchmark->name, benchmark->shape,
                benchmark->layout, options->profile.profile,
                requested_threads, measured_threads, result->median_seconds,
                result->p95_seconds, benchmark->metric, value,
                result->iterations_per_sample, result->checksum);
    }
    return 0;
}

int bench_record_scalar(const bench_options* options,
                        FILE* csv,
                        const char* suite,
                        const char* name,
                        const char* shape,
                        const char* layout,
                        const char* metric,
                        int requested_threads,
                        int measured_threads,
                        double value)
{
    if (options == NULL || suite == NULL || name == NULL ||
        !isfinite(value)) return 1;
    printf("  %-24s threads=%-3d %s=%9.3f\n",
           name, measured_threads, metric, value);
    if (csv != NULL) {
        fprintf(csv, "%s,%s,%s,%s,%s,%d,%d,0,0,%s,%.9g,1,%.9g,ok\n",
                suite, name, shape, layout, options->profile.profile,
                requested_threads, measured_threads, metric, value, value);
    }
    return 0;
}

int bench_execute_case(const bench_options* options,
                       FILE* csv,
                       const bench_case* benchmark,
                       int requested_threads,
                       bench_measurement* result)
{
    if (options == NULL || benchmark == NULL || result == NULL) return 1;
#ifndef _OPENMP
    if (requested_threads > 1) {
        write_skipped(csv, options, benchmark, requested_threads);
        return 2;
    }
#endif
    int measured_threads = bench_configure_threads(requested_threads);
    if (bench_measure_with_reset(benchmark->operation, benchmark->context,
                                 &options->profile, benchmark->reset, result) != 0 ||
        !isfinite(result->checksum)) {
        fprintf(stderr, "Benchmark failed: %s/%s\n",
                benchmark->suite, benchmark->name);
        return 1;
    }
    return bench_record_measurement(options, csv, benchmark,
                                    requested_threads, measured_threads, result);
}

static int suite_selected(const char* requested, const char* suite)
{
    return strcmp(requested, "all") == 0 || strcmp(requested, suite) == 0;
}

int bench_run_kernel_suite(const bench_options* options, FILE* csv);
int bench_run_autograd_suite(const bench_options* options, FILE* csv);
int bench_run_nn_suite(const bench_options* options, FILE* csv);
int bench_run_backward_matrix_suite(const bench_options* options, FILE* csv);
int bench_run_scaling_suite(const bench_options* options, FILE* csv);

int bench_run_suites(const bench_options* options, FILE* csv)
{
    int status = 0;
    printf("SIMD matmul: %s\n\n",
           matmul_avx2_available() ? "AVX2/FMA" : "portable scalar");
    if (suite_selected(options->suite, "kernels")) {
        status |= bench_run_kernel_suite(options, csv);
    }
    if (suite_selected(options->suite, "autograd")) {
        status |= bench_run_autograd_suite(options, csv);
    }
    if (suite_selected(options->suite, "nn")) {
        status |= bench_run_nn_suite(options, csv);
        status |= bench_run_backward_matrix_suite(options, csv);
    }
    if (suite_selected(options->suite, "scaling")) {
        status |= bench_run_scaling_suite(options, csv);
    }
    return status;
}
