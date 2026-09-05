#include "bench_runner.h"

#include <string.h>

#include <tensorlib/autograd.h>

typedef struct {
    ag_tensor* input;
    ag_tensor* weight;
    int backward;
} autograd_context;

static ag_tensor* make_leaf(int ndim, const int* dims, float scale)
{
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int index = 0; index < tensor_numel(raw); ++index) {
        raw->storage->data[index] = scale * (float)((index % 23) - 11);
    }
    return ag_from_owned_tensor(raw, 1);
}

static int autograd_operation(void* opaque, double* checksum)
{
    autograd_context* context = (autograd_context*)opaque;
    ag_tensor* product = NULL;
    ag_tensor* activated = NULL;
    ag_tensor* rows = NULL;
    ag_tensor* loss = NULL;
    int status = 1;

    product = ag_mul(context->input, context->weight);
    if (product == NULL) goto cleanup;
    activated = ag_gelu(product);
    if (activated == NULL) goto cleanup;
    rows = ag_mean(activated, 1, 0);
    if (rows == NULL) goto cleanup;
    loss = ag_mean(rows, 0, 0);
    if (loss == NULL) goto cleanup;
    if (context->backward && ag_backward(loss) != 0) goto cleanup;
    *checksum += loss->value->storage->data[loss->value->offset];
    if (context->backward) {
        if (context->input->grad == NULL || context->weight->grad == NULL) {
            goto cleanup;
        }
        *checksum += context->weight->grad->storage->data[0];
        ag_zero_grad(context->input);
        ag_zero_grad(context->weight);
    }
    status = 0;

cleanup:
    ag_tensor_release(loss);
    ag_tensor_release(rows);
    ag_tensor_release(activated);
    ag_tensor_release(product);
    return status;
}

static int run_autograd_case(const bench_options* options,
                             FILE* csv,
                             const char* name,
                             int rows,
                             int width,
                             int backward)
{
    int input_dims[2] = {rows, width};
    int weight_dims[1] = {width};
    autograd_context context;
    bench_measurement result;
    bench_case benchmark = {
        "autograd", name, "[rowsxwidth]", "channel-broadcast",
        "Melem/s", (double)rows * width / 1.0e6, 0,
        autograd_operation, &context
    };
    memset(&context, 0, sizeof(context));
    context.input = make_leaf(2, input_dims, 0.01f);
    context.weight = make_leaf(1, weight_dims, 0.02f);
    context.backward = backward;
    if (context.input == NULL || context.weight == NULL) {
        ag_tensor_release(context.weight);
        ag_tensor_release(context.input);
        return 1;
    }
    int status = bench_execute_case(options, csv, &benchmark, 1, &result);
    ag_tensor_release(context.weight);
    ag_tensor_release(context.input);
    return status == 1;
}

int bench_run_autograd_suite(const bench_options* options, FILE* csv)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int full = strcmp(options->profile.profile, "full") == 0;
    int rows = smoke ? 4 : (full ? 4096 : 1024);
    int width = smoke ? 8 : 512;
    int status = 0;

    printf("Autograd suite (dynamic graph lifecycle included)\n");
    status |= run_autograd_case(options, csv, "composed_forward",
                                rows, width, 0);
    status |= run_autograd_case(options, csv, "composed_forward_backward",
                                rows, width, 1);
    printf("\n");
    return status;
}
