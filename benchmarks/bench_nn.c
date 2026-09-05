#include "bench_runner.h"

#include <string.h>

#include <tensorlib/nn.h>

typedef ag_tensor* (*model_forward)(const void* model, const ag_tensor* input);

typedef struct {
    void* model;
    nn_module* module;
    model_forward forward;
    ag_tensor* input;
    tensor* targets;
    nn_sgd* sgd;
    nn_adamw* adamw;
    int train;
} nn_bench_context;

static ag_tensor* make_input(int ndim, const int* dims, int token_ids)
{
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int index = 0; index < tensor_numel(raw); ++index) {
        raw->storage->data[index] = token_ids
            ? (float)(index % 256)
            : 0.01f * (float)((index % 31) - 15);
    }
    return ag_from_owned_tensor(raw, 0);
}

static tensor* make_targets(int count, int classes)
{
    int dims[1] = {count};
    tensor* targets = t_alloc(1, dims);
    if (targets == NULL) return NULL;
    for (int index = 0; index < count; ++index) {
        targets->storage->data[index] = (float)(index % classes);
    }
    return targets;
}

static int nn_operation(void* opaque, double* checksum)
{
    nn_bench_context* context = (nn_bench_context*)opaque;
    ag_tensor* output = context->forward(context->model, context->input);
    ag_tensor* loss = NULL;
    int status = 1;

    if (output == NULL) goto cleanup;
    if (context->train) {
        loss = nn_cross_entropy(output, context->targets);
        if (loss == NULL || ag_backward(loss) != 0) goto cleanup;
        if (context->sgd != NULL) {
            if (nn_sgd_step(context->sgd) != 0) goto cleanup;
            nn_sgd_zero_grad(context->sgd);
        } else {
            if (context->adamw == NULL || nn_adamw_step(context->adamw) != 0) {
                goto cleanup;
            }
            nn_adamw_zero_grad(context->adamw);
        }
        *checksum += loss->value->storage->data[loss->value->offset];
    } else {
        *checksum += output->value->storage->data[output->value->offset];
    }
    status = 0;

cleanup:
    ag_tensor_release(loss);
    ag_tensor_release(output);
    return status;
}

static int run_nn_case(const bench_options* options,
                       FILE* csv,
                       const char* name,
                       const char* shape,
                       const char* layout,
                       double items,
                       int threads,
                       nn_bench_context* context,
                       bench_measurement* result)
{
    bench_case benchmark = {
        "nn", name, shape, layout, "items/s", items, 1,
        nn_operation, context
    };
    return bench_execute_case(options, csv, &benchmark, threads, result);
}

static void destroy_context(nn_bench_context* context)
{
    nn_sgd_destroy(context->sgd);
    nn_adamw_destroy(context->adamw);
    t_free(context->targets);
    ag_tensor_release(context->input);
    if (context->module != NULL) context->module->destroy(context->module);
    memset(context, 0, sizeof(*context));
}

static ag_tensor* linear_forward(const void* model, const ag_tensor* input)
{
    return nn_linear_forward((const nn_linear*)model, input);
}

static ag_tensor* layer_norm_forward(const void* model, const ag_tensor* input)
{
    return nn_layer_norm_forward((const nn_layer_norm*)model, input);
}

static ag_tensor* attention_forward(const void* model, const ag_tensor* input)
{
    return nn_multihead_attention_forward(
        (const nn_multihead_attention*)model, input);
}

static ag_tensor* block_forward(const void* model, const ag_tensor* input)
{
    return nn_decoder_block_forward((const nn_decoder_block*)model, input);
}

static ag_tensor* mlp_forward(const void* model, const ag_tensor* input)
{
    return nn_mlp_forward((const nn_mlp*)model, input);
}

static ag_tensor* decoder_forward(const void* model, const ag_tensor* input)
{
    return nn_decoder_forward((const nn_decoder*)model, input);
}
