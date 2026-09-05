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
