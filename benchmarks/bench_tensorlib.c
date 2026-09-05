#include "bench_runner.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void usage(const char* program)
{
    printf("Usage: %s [--profile quick|full] [--suite NAME] "
           "[--threads LIST] [--csv PATH] [--smoke]\n", program);
    printf("Suites: kernels, autograd, nn, scaling, all\n");
}

static int available_threads(void)
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

static void default_threads(bench_options* options)
{
    int maximum = available_threads();
    int value = 1;
    options->thread_count = 0;
    while (value < maximum && options->thread_count < BENCH_MAX_THREADS - 1) {
        options->threads[options->thread_count++] = value;
        value *= 2;
    }
    options->threads[options->thread_count++] = maximum;
}

static int parse_threads(const char* text, bench_options* options)
{
    const char* current = text;
    options->thread_count = 0;
    while (*current != '\0') {
        char* end = NULL;
        long value;
        errno = 0;
        value = strtol(current, &end, 10);
        if (errno != 0 || end == current || value < 1 ||
            value > 1024 || options->thread_count == BENCH_MAX_THREADS) {
            return 1;
        }
        options->threads[options->thread_count++] = (int)value;
        if (*end == '\0') break;
        if (*end != ',') return 1;
        current = end + 1;
    }
    return options->thread_count == 0;
}

static int valid_suite(const char* suite)
{
    return strcmp(suite, "all") == 0 || strcmp(suite, "kernels") == 0 ||
           strcmp(suite, "autograd") == 0 || strcmp(suite, "nn") == 0 ||
           strcmp(suite, "scaling") == 0;
}

static const char* compiler_name(void)
{
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

static void print_environment(const bench_options* options)
{
    printf("TensorLib CPU Benchmarks\n");
    printf("  profile: %s\n", options->profile.profile);
    printf("  compiler: %s\n", compiler_name());
    printf("  logical processors: %d\n", available_threads());
#ifdef _OPENMP
    printf("  OpenMP: enabled (%d)\n", _OPENMP);
#else
    printf("  OpenMP: disabled\n");
#endif
    printf("  requested thread ladder:");
    for (int index = 0; index < options->thread_count; ++index) {
        printf(" %d", options->threads[index]);
    }
    printf("\n\n");
}

int main(int argc, char** argv)
{
    bench_options options;
    const char* csv_path = NULL;
    FILE* csv = NULL;

    options.profile = bench_profile_named("quick");
    options.suite = "all";
    default_threads(&options);
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--help") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (strcmp(argv[index], "--smoke") == 0) {
            options.profile = bench_profile_named("smoke");
        } else if (strcmp(argv[index], "--profile") == 0 && index + 1 < argc) {
            const char* name = argv[++index];
            if (strcmp(name, "quick") != 0 && strcmp(name, "full") != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            options.profile = bench_profile_named(name);
        } else if (strcmp(argv[index], "--suite") == 0 && index + 1 < argc) {
            options.suite = argv[++index];
            if (!valid_suite(options.suite)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[index], "--threads") == 0 && index + 1 < argc) {
            if (parse_threads(argv[++index], &options) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[index], "--csv") == 0 && index + 1 < argc) {
            csv_path = argv[++index];
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (csv_path != NULL) {
        csv = fopen(csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "Could not open CSV output '%s'.\n", csv_path);
            return EXIT_FAILURE;
        }
        fprintf(csv, "suite,case,shape,layout,profile,requested_threads,"
                     "actual_threads,median_seconds,p95_seconds,metric,value,"
                     "iterations,checksum,status\n");
    }
    print_environment(&options);
    int status = bench_run_suites(&options, csv);
    if (csv != NULL && fclose(csv) != 0) status = 1;
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
