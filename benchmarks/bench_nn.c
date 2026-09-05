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
                       const char* metric,
                       double items,
                       int threads,
                       nn_bench_context* context,
                       bench_measurement* result)
{
    bench_case benchmark = {
        "nn", name, shape, layout, metric, items, 1,
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

typedef enum {
    COMPONENT_LINEAR,
    COMPONENT_LAYER_NORM,
    COMPONENT_ATTENTION,
    COMPONENT_BLOCK
} component_kind;

static int run_component(const bench_options* options,
                         FILE* csv,
                         component_kind kind,
                         int batch,
                         int time,
                         int channels)
{
    int dims[3] = {batch, time, channels};
    int linear_dims[2] = {batch * time, channels};
    nn_bench_context context;
    bench_measurement result;
    nn_rng rng;
    const char* name = NULL;
    const char* shape = "[B,T,C]";

    memset(&context, 0, sizeof(context));
    nn_rng_seed(&rng, UINT64_C(0xB34C4));
    if (kind == COMPONENT_LINEAR) {
        nn_linear* model = nn_linear_create("bench_linear", channels,
            channels * 3, 1, NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO, &rng);
        context.model = model;
        context.module = model == NULL ? NULL : &model->base;
        context.forward = linear_forward;
        context.input = make_input(2, linear_dims, 0);
        name = "linear_qkv";
        shape = "[BT,C]->[BT,3C]";
    } else if (kind == COMPONENT_LAYER_NORM) {
        nn_layer_norm* model = nn_layer_norm_create(
            "bench_norm", channels, 1e-5f, 1);
        context.model = model;
        context.module = model == NULL ? NULL : &model->base;
        context.forward = layer_norm_forward;
        context.input = make_input(3, dims, 0);
        name = "layer_norm";
    } else if (kind == COMPONENT_ATTENTION) {
        nn_multihead_attention* model = nn_multihead_attention_create(
            "bench_attention", channels, 6, 0.0f, &rng);
        context.model = model;
        context.module = model == NULL ? NULL : &model->base;
        context.forward = attention_forward;
        context.input = make_input(3, dims, 0);
        name = "causal_attention";
    } else {
        nn_decoder_block* model = nn_decoder_block_create(
            "bench_block", channels, 6, 0.0f, 1e-5f, &rng);
        context.model = model;
        context.module = model == NULL ? NULL : &model->base;
        context.forward = block_forward;
        context.input = make_input(3, dims, 0);
        name = "decoder_block";
    }
    if (context.module == NULL || context.input == NULL) {
        destroy_context(&context);
        return 1;
    }
    nn_module_set_training(context.module, 0);
    int status = run_nn_case(options, csv, name, shape,
                             "forward;graph-build", "tokens/s",
                             (double)(batch * time),
                             1, &context, &result);
    destroy_context(&context);
    return status == 1;
}

static int setup_mlp(nn_bench_context* context, int batch, int train)
{
    int hidden[1] = {128};
    nn_activation activations[2] = {
        nn_activation_relu(), nn_activation_custom("identity", NULL, NULL)
    };
    nn_mlp_config config = {
        784, hidden, 1, 10, activations, 1,
        NN_INIT_HE_NORMAL, NN_INIT_ZERO
    };
    int input_dims[2] = {batch, 784};
    nn_rng rng;
    nn_mlp* model;

    memset(context, 0, sizeof(*context));
    nn_rng_seed(&rng, UINT64_C(0x4D4E495354));
    model = nn_mlp_create("bench_mlp", &config, &rng);
    context->model = model;
    context->module = model == NULL ? NULL : &model->base;
    context->forward = mlp_forward;
    context->input = make_input(2, input_dims, 0);
    context->train = train;
    if (train) {
        context->targets = make_targets(batch, 10);
        context->sgd = model == NULL ? NULL : nn_sgd_create(&model->base, 0.05f);
    }
    return context->module == NULL || context->input == NULL ||
           (train && (context->targets == NULL || context->sgd == NULL));
}

static int run_mlp(const bench_options* options, FILE* csv, int batch, int train)
{
    nn_bench_context context;
    bench_measurement result;
    if (setup_mlp(&context, batch, train) != 0) {
        destroy_context(&context);
        return 1;
    }
    int status = run_nn_case(options, csv,
        train ? "mnist_mlp_train_step" : "mnist_mlp_forward",
        "[B,784]->[B,10]", train ? "forward+loss+backward+sgd" :
        "forward;graph-build", "samples/s", (double)batch,
        1, &context, &result);
    destroy_context(&context);
    return status == 1;
}

static tensor* make_token_targets(int batch, int time, int classes)
{
    int dims[2] = {batch, time};
    tensor* targets = t_alloc(2, dims);
    if (targets == NULL) return NULL;
    for (int index = 0; index < batch * time; ++index) {
        targets->storage->data[index] = (float)(index % classes);
    }
    return targets;
}

static int setup_decoder(nn_bench_context* context,
                         int batch,
                         int time,
                         int channels,
                         int layers,
                         int train)
{
    nn_decoder_config config = {
        256, time, channels, 6, layers, train ? 0.1f : 0.0f, 1e-5f
    };
    int input_dims[2] = {batch, time};
    nn_rng rng;
    nn_decoder* model;

    memset(context, 0, sizeof(*context));
    nn_rng_seed(&rng, UINT64_C(0x71594C4D));
    model = nn_decoder_create("bench_decoder", &config, &rng);
    context->model = model;
    context->module = model == NULL ? NULL : &model->base;
    context->forward = decoder_forward;
    context->input = make_input(2, input_dims, 1);
    context->train = train;
    if (train) {
        nn_adamw_config optimizer_config = nn_adamw_default_config();
        optimizer_config.max_grad_norm = 1.0f;
        context->targets = make_token_targets(batch, time, 256);
        context->adamw = model == NULL ? NULL :
            nn_adamw_create(&model->base, &optimizer_config);
    }
    if (context->module != NULL && !train) {
        nn_module_set_training(context->module, 0);
    }
    return context->module == NULL || context->input == NULL ||
           (train && (context->targets == NULL || context->adamw == NULL));
}

static int run_decoder_case(const bench_options* options,
                            FILE* csv,
                            int batch,
                            int time,
                            int channels,
                            int layers,
                            int train,
                            int threads)
{
    nn_bench_context context;
    bench_measurement result;
    if (setup_decoder(&context, batch, time, channels, layers, train) != 0) {
        destroy_context(&context);
        return 1;
    }
    int status = run_nn_case(options, csv,
        train ? "tiny_lm_train_step" : "tiny_lm_forward",
        "[B,128]->[B,128,256]",
        train ? "forward+loss+backward+adamw" : "forward;graph-build",
        "tokens/s", (double)(batch * time), threads, &context, &result);
    destroy_context(&context);
    return status == 1;
}

static int run_decoder_cases(const bench_options* options, FILE* csv,
                             int batch, int threads)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int time = smoke ? 8 : 128;
    int channels = smoke ? 24 : 192;
    int layers = smoke ? 1 : 4;
    int status = run_decoder_case(options, csv, batch, time, channels,
                                  layers, 0, threads);
    status |= run_decoder_case(options, csv, batch, time, channels,
                               layers, 1, threads);
    return status;
}

int bench_run_nn_suite(const bench_options* options, FILE* csv)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int full = strcmp(options->profile.profile, "full") == 0;
    int batch = smoke ? 1 : (full ? 4 : 2);
    int time = smoke ? 8 : 128;
    int channels = smoke ? 24 : 192;
    int status = 0;

    printf("Neural-network suite (eager float32)\n");
    status |= run_component(options, csv, COMPONENT_LINEAR,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_LAYER_NORM,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_ATTENTION,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_BLOCK,
                            batch, time, channels);
    status |= run_mlp(options, csv, smoke ? 2 : 64, 0);
    status |= run_mlp(options, csv, smoke ? 2 : 64, 1);
    status |= run_decoder_cases(options, csv, batch, 1);
    printf("\n");
    return status;
}
