#include "bench_runner.h"

#include <string.h>

#include <tensorlib/tensor.h>

typedef enum { SCALE_ADD, SCALE_GELU, SCALE_MATMUL } scaling_kind;

typedef struct {
    scaling_kind kind;
    tensor* a;
    tensor* b;
} scaling_context;

static void fill_tensor(tensor* value, float scale)
{
    for (int index = 0; index < tensor_numel(value); ++index) {
        value->storage->data[index] =
            scale * (float)((index % 29) - 14);
    }
}

static int scaling_operation(void* opaque, double* checksum)
{
    scaling_context* context = (scaling_context*)opaque;
    tensor* output;
    if (context->kind == SCALE_ADD) output = t_add(context->a, context->b);
    else if (context->kind == SCALE_GELU) output = t_gelu(context->a);
    else output = t_matmul(context->a, context->b);
    if (output == NULL) return 1;
    *checksum += output->storage->data[output->offset];
    t_free(output);
    return 0;
}

static void write_scaling_row(FILE* csv,
                              const bench_options* options,
                              const bench_case* benchmark,
                              int threads,
                              double speedup,
                              double efficiency)
{
    if (csv == NULL) return;
    fprintf(csv, "%s,%s,%s,%s,%s,%d,%d,0,0,speedup,%.9g,0,%.9g,derived\n",
            benchmark->suite, benchmark->name, benchmark->shape,
            benchmark->layout, options->profile.profile, threads, threads,
            speedup, efficiency);
}

static int run_scaling_case(const bench_options* options,
                            FILE* csv,
                            scaling_kind kind,
                            const char* name,
                            const char* shape,
                            const char* metric,
                            double units,
                            int a_ndim,
                            const int* a_dims,
                            int b_ndim,
                            const int* b_dims)
{
    scaling_context context;
    bench_case benchmark = {
        "scaling", name, shape, "contiguous", metric, units, 1,
        scaling_operation, &context
    };
    double baseline = 0.0;
    int status = 0;

    memset(&context, 0, sizeof(context));
    context.kind = kind;
    context.a = t_alloc(a_ndim, a_dims);
    context.b = b_ndim < 0 ? NULL : t_alloc(b_ndim, b_dims);
    if (context.a == NULL || (b_ndim >= 0 && context.b == NULL)) {
        t_free(context.b);
        t_free(context.a);
        return 1;
    }
    fill_tensor(context.a, 0.01f);
    if (context.b != NULL) fill_tensor(context.b, 0.02f);
    printf(" %s\n", name);
    for (int index = 0; index < options->thread_count; ++index) {
        bench_measurement result;
        int threads = options->threads[index];
        int result_status = bench_execute_case(
            options, csv, &benchmark, threads, &result);
        if (result_status == 1) {
            status = 1;
            continue;
        }
        if (result_status == 2) continue;
        if (baseline == 0.0) baseline = result.median_seconds;
        double speedup = baseline / result.median_seconds;
        double efficiency = speedup / (double)threads;
        printf("    speedup=%6.2fx efficiency=%5.1f%%\n",
               speedup, efficiency * 100.0);
        write_scaling_row(csv, options, &benchmark, threads,
                          speedup, efficiency);
    }
    t_free(context.b);
    t_free(context.a);
    return status;
}

int bench_run_scaling_suite(const bench_options* options, FILE* csv)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int full = strcmp(options->profile.profile, "full") == 0;
    int elements = smoke ? 4096 : 8 * 1024 * 1024;
    int size = smoke ? 32 : (full ? 1536 : 768);
    int vector_dims[1] = {elements};
    int matrix_dims[2] = {size, size};
    int status = 0;

    printf("Thread-scaling suite\n");
    status |= run_scaling_case(options, csv, SCALE_ADD, "add_large",
        "[N]", "GB/s", 12.0 * elements / 1e9,
        1, vector_dims, 1, vector_dims);
    status |= run_scaling_case(options, csv, SCALE_GELU, "gelu_large",
        "[N]", "GB/s", 8.0 * elements / 1e9,
        1, vector_dims, -1, NULL);
    status |= run_scaling_case(options, csv, SCALE_MATMUL, "matmul_square",
        "[M,K]x[K,N]", "GFLOP/s", 2.0 * size * size * size / 1e9,
        2, matrix_dims, 2, matrix_dims);
    printf("\n");
    return status;
}
