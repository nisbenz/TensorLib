#include "bench_harness.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>

extern void openblas_set_num_threads(int num_threads);

typedef struct {
    int batch;
    int rows;
    int inner;
    int columns;
    float* a;
    float* b;
} blas_context;

static int blas_operation(void* opaque, double* checksum)
{
    blas_context* context = (blas_context*)opaque;
    size_t output_count = (size_t)context->batch * context->rows * context->columns;
    float* output = (float*)malloc(output_count * sizeof(*output));
    if (output == NULL) return 1;
    for (int batch = 0; batch < context->batch; ++batch) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    context->rows, context->columns, context->inner,
                    1.0f,
                    context->a + (size_t)batch * context->rows * context->inner,
                    context->inner,
                    context->b + (size_t)batch * context->inner * context->columns,
                    context->columns, 0.0f,
                    output + (size_t)batch * context->rows * context->columns,
                    context->columns);
    }
    *checksum += output[0] + output[output_count - 1];
    free(output);
    return 0;
}

static int run_case(const bench_profile* profile,
                    FILE* csv,
                    const char* name,
                    const char* shape,
                    int threads,
                    int batch,
                    int rows,
                    int inner,
                    int columns)
{
    blas_context context = {batch, rows, inner, columns, NULL, NULL};
    size_t a_count = (size_t)batch * rows * inner;
    size_t b_count = (size_t)batch * inner * columns;
    bench_measurement result;
    double gflops = 2.0 * batch * rows * inner * columns / 1e9;

    context.a = (float*)malloc(a_count * sizeof(*context.a));
    context.b = (float*)malloc(b_count * sizeof(*context.b));
    if (context.a == NULL || context.b == NULL) {
        free(context.b);
        free(context.a);
        return 1;
    }
    for (size_t index = 0; index < a_count; ++index) {
        context.a[index] = 0.01f * (float)((int)(index % 29) - 14);
    }
    for (size_t index = 0; index < b_count; ++index) {
        context.b[index] = 0.02f * (float)((int)(index % 31) - 15);
    }
    openblas_set_num_threads(threads);
    if (bench_measure(blas_operation, &context, profile, &result) != 0) {
        free(context.b);
        free(context.a);
        return 1;
    }
    printf("  %-20s threads=%-3d median=%9.3f ms GFLOP/s=%9.3f\n",
           name, threads, result.median_seconds * 1000.0,
           gflops / result.median_seconds);
    if (csv != NULL) {
        fprintf(csv, "openblas,%s,%s,contiguous,%s,%d,%d,%.9g,%.9g,"
                     "GFLOP/s,%.9g,%d,%.9g,ok\n",
                name, shape, profile->profile, threads, threads,
                result.median_seconds, result.p95_seconds,
                gflops / result.median_seconds,
                result.iterations_per_sample, result.checksum);
    }
    free(context.b);
    free(context.a);
    return 0;
}

int main(int argc, char** argv)
{
    bench_profile profile = bench_profile_named("quick");
    const char* csv_path = NULL;
    int threads = 1;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--profile") == 0 && index + 1 < argc) {
            profile = bench_profile_named(argv[++index]);
        } else if (strcmp(argv[index], "--threads") == 0 && index + 1 < argc) {
            threads = atoi(argv[++index]);
        } else if (strcmp(argv[index], "--csv") == 0 && index + 1 < argc) {
            csv_path = argv[++index];
        } else if (strcmp(argv[index], "--smoke") == 0) {
            profile = bench_profile_named("smoke");
        } else {
            fprintf(stderr, "Usage: %s [--profile quick|full] "
                    "[--threads N] [--csv PATH] [--smoke]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (threads < 1) return EXIT_FAILURE;
    FILE* csv = csv_path == NULL ? NULL : fopen(csv_path, "w");
    if (csv_path != NULL && csv == NULL) return EXIT_FAILURE;
    if (csv != NULL) {
        fprintf(csv, "suite,case,shape,layout,profile,requested_threads,"
                     "actual_threads,median_seconds,p95_seconds,metric,value,"
                     "iterations,checksum,status\n");
    }
    int smoke = strcmp(profile.profile, "smoke") == 0;
    int full = strcmp(profile.profile, "full") == 0;
    int square = smoke ? 32 : (full ? 1024 : 256);
    int tokens = smoke ? 8 : (full ? 512 : 128);
    int status = 0;
    printf("OpenBLAS matched matmul baseline\n");
    status |= run_case(&profile, csv, "matmul_square", "[MxK]x[KxN]",
                       threads, 1, square, square, square);
    status |= run_case(&profile, csv, "matmul_qkv", "[BTxC]x[Cx3C]",
                       threads, 1, tokens, 192, 576);
    status |= run_case(&profile, csv, "matmul_attention",
                       "[BHxTxD]x[BHxDxT]", threads,
                       smoke ? 2 : 12, smoke ? 16 : 128,
                       smoke ? 8 : 64, smoke ? 16 : 128);
    if (csv != NULL && fclose(csv) != 0) status = 1;
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
