#include "bench_runner.h"

#include <stdlib.h>
#include <string.h>

#include <tensorlib/tensor.h>

typedef enum {
    KERNEL_ALLOC,
    KERNEL_CLONE,
    KERNEL_CONTIGUOUS,
    KERNEL_TRANSPOSE,
    KERNEL_ADD,
    KERNEL_MUL,
    KERNEL_GELU,
    KERNEL_SUM,
    KERNEL_MAX,
    KERNEL_GATHER,
    KERNEL_MATMUL,
    KERNEL_PACKED_MATMUL
} kernel_kind;

typedef struct {
    kernel_kind kind;
    int ndim;
    int dims[4];
    tensor* a;
    tensor* b;
    tensor* a_owner;
    tensor_matmul_packed_rhs* packed;
    int reduction_axis;
} kernel_context;

static void fill_tensor(tensor* value, float scale)
{
    int count = tensor_numel(value);
    for (int index = 0; index < count; ++index) {
        value->storage->data[index] =
            scale * (float)((index % 31) - 15);
    }
}

static int kernel_init(kernel_context* context,
                       kernel_kind kind,
                       int ndim,
                       const int* dims,
                       int b_ndim,
                       const int* b_dims)
{
    memset(context, 0, sizeof(*context));
    context->kind = kind;
    context->ndim = ndim;
    for (int index = 0; index < ndim; ++index) context->dims[index] = dims[index];
    if (kind != KERNEL_ALLOC) {
        context->a = t_alloc(ndim, dims);
        if (context->a == NULL) return 1;
        fill_tensor(context->a, 0.01f);
    }
    if (b_ndim >= 0) {
        context->b = t_alloc(b_ndim, b_dims);
        if (context->b == NULL) return 1;
        fill_tensor(context->b, 0.02f);
    }
    if (kind == KERNEL_GATHER) {
        int vocabulary = dims[0];
        int count = tensor_numel(context->b);
        for (int index = 0; index < count; ++index) {
            context->b->storage->data[index] = (float)(index % vocabulary);
        }
    }
    if (kind == KERNEL_PACKED_MATMUL) {
        context->packed = t_pack_matmul_rhs(context->b);
        if (context->packed == NULL) return 1;
    }
    return 0;
}

static int kernel_make_transposed(kernel_context* context,
                                  kernel_kind kind,
                                  int rows,
                                  int columns)
{
    int base_dims[2] = {columns, rows};
    if (kernel_init(context, kind, 2, base_dims, -1, NULL) != 0) return 1;
    context->a_owner = context->a;
    context->a = t_transpose(context->a_owner, 0, 1);
    return context->a == NULL;
}

static void kernel_destroy(kernel_context* context)
{
    t_free_matmul_packed_rhs(context->packed);
    t_free(context->b);
    t_free(context->a);
    t_free(context->a_owner);
    memset(context, 0, sizeof(*context));
}

static void consume_tensor(tensor* value, double* checksum)
{
    int count = tensor_numel(value);
    *checksum += value->storage->data[value->offset];
    if (count > 1 && is_contiguous(value)) {
        *checksum += value->storage->data[value->offset + count - 1];
    }
}

static int kernel_operation(void* opaque, double* checksum)
{
    kernel_context* context = (kernel_context*)opaque;
    tensor* output = NULL;

    switch (context->kind) {
        case KERNEL_ALLOC:
            output = t_alloc(context->ndim, context->dims);
            break;
        case KERNEL_CLONE:
            output = t_clone(context->a);
            break;
        case KERNEL_CONTIGUOUS:
            output = t_contiguous(context->a);
            break;
        case KERNEL_TRANSPOSE:
            output = t_transpose(context->a, 0, 1);
            break;
        case KERNEL_ADD:
            output = t_add(context->a, context->b);
            break;
        case KERNEL_MUL:
            output = t_mul(context->a, context->b);
            break;
        case KERNEL_GELU:
            output = t_gelu(context->a);
            break;
        case KERNEL_SUM:
            output = t_sum(context->a, context->reduction_axis);
            break;
        case KERNEL_MAX:
            output = t_max(context->a, context->reduction_axis);
            break;
        case KERNEL_GATHER:
            output = t_gather_rows(context->a, context->b);
            break;
        case KERNEL_MATMUL:
            output = t_matmul(context->a, context->b);
            break;
        case KERNEL_PACKED_MATMUL:
            output = t_matmul_packed_rhs(context->a, context->packed);
            break;
    }
    if (output == NULL) return 1;
    consume_tensor(output, checksum);
    t_free(output);
    return 0;
}

static int run_kernel(const bench_options* options,
                      FILE* csv,
                      bench_case* benchmark,
                      kernel_context* context,
                      int threads)
{
    bench_measurement result;
    benchmark->operation = kernel_operation;
    benchmark->context = context;
    int status = bench_execute_case(options, csv, benchmark, threads, &result);
    kernel_destroy(context);
    return status == 1;
}

static int run_case(const bench_options* options,
                    FILE* csv,
                    kernel_kind kind,
                    const char* name,
                    const char* shape,
                    const char* layout,
                    const char* metric,
                    double units,
                    int ndim,
                    const int* dims,
                    int b_ndim,
                    const int* b_dims,
                    int reduction_axis)
{
    kernel_context context;
    bench_case benchmark = {
        "kernels", name, shape, layout, metric, units, 0, NULL, NULL
    };
    if (kernel_init(&context, kind, ndim, dims, b_ndim, b_dims) != 0) {
        kernel_destroy(&context);
        return 1;
    }
    context.reduction_axis = reduction_axis;
    return run_kernel(options, csv, &benchmark, &context, 1);
}

int bench_run_kernel_suite(const bench_options* options, FILE* csv)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int full = strcmp(options->profile.profile, "full") == 0;
    int elements = smoke ? 4096 : 8 * 1024 * 1024;
    int square = smoke ? 32 : (full ? 1024 : 256);
    int vector_dims[1] = {elements};
    int matrix_dims[2] = {smoke ? 32 : 1024, smoke ? 32 : 1024};
    int broadcast_dims[3] = {smoke ? 2 : 32, smoke ? 8 : 128,
                             smoke ? 16 : 512};
    int channel_dims[1] = {broadcast_dims[2]};
    int gather_table[2] = {smoke ? 64 : 16384, smoke ? 16 : 256};
    int gather_indices[2] = {smoke ? 2 : 16, smoke ? 8 : 128};
    int square_a[2] = {square, square};
    int square_b[2] = {square, square};
    int qkv_a[2] = {full ? 512 : (smoke ? 8 : 128), 192};
    int qkv_b[2] = {192, 576};
    int attention_a[3] = {smoke ? 2 : 12, smoke ? 16 : 128,
                           smoke ? 8 : 64};
    int attention_b[3] = {attention_a[0], attention_a[2], attention_a[1]};
    int status = 0;

    printf("Kernel suite (public API cost, including output allocation)\n");
    status |= run_case(options, csv, KERNEL_ALLOC, "allocate_free",
                       "[N]", "contiguous", "ms/call", 0.0,
                       1, vector_dims, -1, NULL, 0);
    status |= run_case(options, csv, KERNEL_CLONE, "clone",
                       "[N]", "contiguous", "GB/s", 8.0 * elements / 1e9,
                       1, vector_dims, -1, NULL, 0);
    {
        kernel_context context;
        bench_case benchmark = {"kernels", "transpose_view", "[R,C]",
            "zero-copy", "ms/call", 0.0, 0, NULL, NULL};
        if (kernel_init(&context, KERNEL_TRANSPOSE, 2, matrix_dims,
                        -1, NULL) != 0) status = 1;
        else status |= run_kernel(options, csv, &benchmark, &context, 1);
    }
    {
        kernel_context context;
        bench_case benchmark = {"kernels", "contiguous_copy", "[R,C]",
            "transposed", "GB/s",
            8.0 * matrix_dims[0] * matrix_dims[1] / 1e9, 0, NULL, NULL};
        if (kernel_make_transposed(&context, KERNEL_CONTIGUOUS,
                                   matrix_dims[0], matrix_dims[1]) != 0) status = 1;
        else status |= run_kernel(options, csv, &benchmark, &context, 1);
    }
    status |= run_case(options, csv, KERNEL_ADD, "add_contiguous", "[N]",
                       "contiguous", "GB/s", 12.0 * elements / 1e9,
                       1, vector_dims, 1, vector_dims, 0);
    status |= run_case(options, csv, KERNEL_MUL, "multiply_contiguous", "[N]",
                       "contiguous", "GB/s", 12.0 * elements / 1e9,
                       1, vector_dims, 1, vector_dims, 0);
    status |= run_case(options, csv, KERNEL_ADD, "add_channel_broadcast",
                       "[B,T,C]+[C]", "broadcast", "GB/s",
                       8.0 * broadcast_dims[0] * broadcast_dims[1] *
                       broadcast_dims[2] / 1e9,
                       3, broadcast_dims, 1, channel_dims, 0);
    status |= run_case(options, csv, KERNEL_GELU, "gelu", "[N]",
                       "contiguous", "GB/s", 8.0 * elements / 1e9,
                       1, vector_dims, -1, NULL, 0);
    status |= run_case(options, csv, KERNEL_SUM, "sum_trailing", "[R,C]",
                       "contiguous;serial", "GB/s",
                       4.0 * matrix_dims[0] * matrix_dims[1] / 1e9,
                       2, matrix_dims, -1, NULL, 1);
    status |= run_case(options, csv, KERNEL_MAX, "max_non_trailing", "[R,C]",
                       "contiguous;serial", "GB/s",
                       4.0 * matrix_dims[0] * matrix_dims[1] / 1e9,
                       2, matrix_dims, -1, NULL, 0);
    status |= run_case(options, csv, KERNEL_GATHER, "embedding_gather",
                       "[V,C]@[B,T]", "row-gather;serial", "GB/s",
                       4.0 * gather_indices[0] * gather_indices[1] *
                       gather_table[1] / 1e9,
                       2, gather_table, 2, gather_indices, 0);
    status |= run_case(options, csv, KERNEL_MATMUL, "matmul_square", "[M,K]x[K,N]",
                       "contiguous", "GFLOP/s",
                       2.0 * square * square * square / 1e9,
                       2, square_a, 2, square_b, 0);
    status |= run_case(options, csv, KERNEL_PACKED_MATMUL, "matmul_square_packed",
                       "[M,K]x[K,N]", "packed-rhs", "GFLOP/s",
                       2.0 * square * square * square / 1e9,
                       2, square_a, 2, square_b, 0);
    status |= run_case(options, csv, KERNEL_MATMUL, "matmul_qkv",
                       "[BT,C]x[C,3C]", "contiguous", "GFLOP/s",
                       2.0 * qkv_a[0] * qkv_a[1] * qkv_b[1] / 1e9,
                       2, qkv_a, 2, qkv_b, 0);
    status |= run_case(options, csv, KERNEL_MATMUL, "matmul_attention",
                       "[BH,T,D]x[BH,D,T]", "batched", "GFLOP/s",
                       2.0 * attention_a[0] * attention_a[1] * attention_a[2] *
                       attention_b[2] / 1e9,
                       3, attention_a, 3, attention_b, 0);
    printf("\n");
    return status;
}
