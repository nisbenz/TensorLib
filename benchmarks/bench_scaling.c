#include "bench_runner.h"

#include <string.h>

#include <tensorlib/tensor.h>

typedef enum {
    SCALE_ADD, SCALE_GELU, SCALE_MATMUL, SCALE_CLONE, SCALE_PACKED, SCALE_PACK
} scaling_kind;

typedef struct {
    scaling_kind kind;
    tensor* a;
    tensor* b;
    tensor_matmul_packed_rhs* packed;
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
    if (context->kind == SCALE_PACK) {
        tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(context->a);
        if (packed == NULL) return 1;
        *checksum += context->a->storage->data[context->a->offset];
        t_free_matmul_packed_rhs(packed);
        return 0;
    }
    if (context->kind == SCALE_ADD) output = t_add(context->a, context->b);
    else if (context->kind == SCALE_GELU) output = t_gelu(context->a);
    else if (context->kind == SCALE_CLONE) output = t_clone(context->a);
    else if (context->kind == SCALE_PACKED)
        output = t_matmul_packed_rhs(context->a, context->packed);
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
                            const char* suite,
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
        suite, name, shape, "contiguous", metric, units, 1,
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
    if (kind == SCALE_PACKED) {
        context.packed = t_pack_matmul_rhs(context.b);
        if (context.packed == NULL) {
            t_free(context.b);
            t_free(context.a);
            return 1;
        }
    }
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
    t_free_matmul_packed_rhs(context.packed);
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
    int command_batch1_a[2] = {256, 512};
    int command_batch1_b[2] = {512, 1536};
    int command_batch16_a[2] = {4096, 512};
    int command_batch16_b[2] = {512, 1536};
    int command_dweight_a[2] = {512, 4096};
    int command_dweight_b[2] = {4096, 2048};
    int command_attention_a[3] = {128, 256, 64};
    int command_attention_b[3] = {128, 64, 256};
    int status = 0;
    int bench_run_decoder_scaling(const bench_options*, FILE*);

    printf("Thread-scaling suite\n");
    status |= run_scaling_case(options, csv, "scaling", SCALE_ADD, "add_large",
        "[N]", "GB/s", 12.0 * elements / 1e9,
        1, vector_dims, 1, vector_dims);
    status |= run_scaling_case(options, csv, "scaling", SCALE_GELU, "gelu_large",
        "[N]", "GB/s", 8.0 * elements / 1e9,
        1, vector_dims, -1, NULL);
    status |= run_scaling_case(options, csv, "scaling", SCALE_MATMUL, "matmul_square",
        "[MxK]x[KxN]", "GFLOP/s", 2.0 * size * size * size / 1e9,
        2, matrix_dims, 2, matrix_dims);
    if (full) {
        status |= run_scaling_case(options, csv, "scaling", SCALE_MATMUL,
            "command_qkv_batch1", "[256x512]x[512x1536]", "GFLOP/s",
            2.0 * 256 * 512 * 1536 / 1e9,
            2, command_batch1_a, 2, command_batch1_b);
        status |= run_scaling_case(options, csv, "scaling", SCALE_MATMUL,
            "command_qkv_batch16", "[4096x512]x[512x1536]", "GFLOP/s",
            2.0 * 4096 * 512 * 1536 / 1e9,
            2, command_batch16_a, 2, command_batch16_b);
        status |= run_scaling_case(options, csv, "scaling", SCALE_MATMUL,
            "command_mlp_dweight", "[512x4096]x[4096x2048]", "GFLOP/s",
            2.0 * 512 * 4096 * 2048 / 1e9,
            2, command_dweight_a, 2, command_dweight_b);
        status |= run_scaling_case(options, csv, "scaling", SCALE_MATMUL,
            "command_attention", "[128x256x64]x[128x64x256]", "GFLOP/s",
            2.0 * 128 * 256 * 64 * 256 / 1e9,
            3, command_attention_a, 3, command_attention_b);
    }
    status |= bench_run_decoder_scaling(options, csv);
    printf("\n");
    return status;
}

/* Sweep sizes around the serial/parallel crossover without changing kernels. */
int bench_run_policy_suite(const bench_options* options, FILE* csv)
{
    const int counts[] = {256, 4096, 65536, 262144, 1048576};
    const int sizes[] = {32, 64, 128, 256};
    const scaling_kind vector_kinds[] = {SCALE_ADD, SCALE_GELU, SCALE_CLONE};
    const char* vector_names[] = {"add", "gelu", "clone"};
    const scaling_kind matrix_kinds[] = {SCALE_MATMUL, SCALE_PACKED, SCALE_PACK};
    const char* matrix_names[] = {"matmul", "packed_matmul", "pack_rhs"};
    int status = 0;
    printf("Thread-policy crossover suite\n");
    for (int index = 0; index < 5; ++index) {
        int dims[] = {counts[index]};
        char shape[48];
        snprintf(shape, sizeof(shape), "[%d]", dims[0]);
        for (int kind = 0; kind < 3; ++kind) {
            char name[48];
            snprintf(name, sizeof(name), "%s_%d", vector_names[kind], dims[0]);
            status |= run_scaling_case(options, csv, "policy", vector_kinds[kind],
                name, shape, "GB/s", (kind == 0 ? 12.0 : 8.0) * dims[0] / 1e9,
                1, dims, kind == 0 ? 1 : -1, dims);
        }
    }
    for (int index = 0; index < 4; ++index) {
        int size = sizes[index];
        int dims[] = {size, size};
        char shape[64];
        snprintf(shape, sizeof(shape), "[%dx%d]x[%dx%d]", size, size, size, size);
        for (int kind = 0; kind < 3; ++kind) {
            char name[48];
            snprintf(name, sizeof(name), "%s_%d", matrix_names[kind], size);
            status |= run_scaling_case(options, csv, "policy", matrix_kinds[kind],
                name, kind == 2 ? "RHS square" : shape,
                kind == 2 ? "GB/s" : "GFLOP/s",
                (kind == 2 ? 8.0 * size * size : 2.0 * size * size * size) / 1e9,
                2, dims, kind == 2 ? -1 : 2, dims);
        }
    }
    int batch_dims[] = {8, 64, 64};
    status |= run_scaling_case(options, csv, "policy", SCALE_MATMUL,
        "batched_matmul_64", "[8x64x64]x[8x64x64]", "GFLOP/s",
        2.0 * 8 * 64 * 64 * 64 / 1e9, 3, batch_dims, 3, batch_dims);
    return status;
}
