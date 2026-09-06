#include "bench_runner.h"

#include <stdlib.h>
#include <string.h>

#include <tensorlib/nn.h>
#include <tensorlib/autograd_internal.h>
#include "../src/tensor/tensor_alloc_internal.h"

typedef ag_tensor* (*model_forward)(const void* model, const ag_tensor* input);

typedef struct {
    void* model;
    nn_module* module;
    model_forward forward;
    ag_tensor* input;
    tensor* targets;
    nn_sgd* sgd;
    nn_adamw* adamw;
    nn_rng rng;
    int train;
    int component_backward;
    double* phase_samples[6];
    int phase_sample_count;
    int phase_sample_capacity;
    tensor_alloc_stats allocation_stats;
    ag_backward_stats backward_stats;
    size_t allocation_baseline_bytes;
} nn_bench_context;

enum {
    PHASE_ZERO_GRAD,
    PHASE_FORWARD,
    PHASE_LOSS,
    PHASE_BACKWARD,
    PHASE_ADAMW,
    PHASE_GRAPH_RELEASE,
    PHASE_COUNT
};

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
    ag_tensor* output;
    ag_tensor* loss = NULL;
    int status = 1;
    double started;
    double phase_times[PHASE_COUNT] = {0};

    if (context->train) {
        started = bench_now_seconds();
        if (context->sgd != NULL) {
            nn_sgd_zero_grad(context->sgd);
        } else if (context->adamw != NULL) {
            nn_adamw_zero_grad(context->adamw);
        }
        phase_times[PHASE_ZERO_GRAD] = bench_now_seconds() - started;
    }
    if (context->component_backward) nn_module_zero_grad(context->module);

    started = bench_now_seconds();
    output = context->forward(context->model, context->input);
    if (context->train) {
        phase_times[PHASE_FORWARD] = bench_now_seconds() - started;
    }
    if (output == NULL) goto cleanup;
    if (context->train) {
        started = bench_now_seconds();
        loss = nn_cross_entropy(output, context->targets);
        phase_times[PHASE_LOSS] = bench_now_seconds() - started;
        if (loss == NULL) goto cleanup;
        started = bench_now_seconds();
        if (ag_backward(loss) != 0) goto cleanup;
        phase_times[PHASE_BACKWARD] = bench_now_seconds() - started;
        started = bench_now_seconds();
        if (context->sgd != NULL) {
            if (nn_sgd_step(context->sgd) != 0) goto cleanup;
        } else {
            if (context->adamw == NULL || nn_adamw_step(context->adamw) != 0) {
                goto cleanup;
            }
        }
        phase_times[PHASE_ADAMW] = bench_now_seconds() - started;
        *checksum += loss->value->storage->data[loss->value->offset];
    } else if (context->component_backward) {
        tensor* seed = t_alloc(output->value->ndim, output->value->dims);
        if (seed == NULL) goto cleanup;
        for (int index = 0; index < tensor_numel(seed); ++index) {
            seed->storage->data[index] = 1.0f;
        }
        if (ag_backward_with_grad(output, seed) != 0) {
            t_free(seed);
            goto cleanup;
        }
        t_free(seed);
        *checksum += output->value->storage->data[output->value->offset];
    } else {
        *checksum += output->value->storage->data[output->value->offset];
    }
    status = 0;

cleanup:
    if (context->train) started = bench_now_seconds();
    ag_tensor_release(loss);
    ag_tensor_release(output);
    if (context->component_backward) nn_module_zero_grad(context->module);
    if (context->train) {
        phase_times[PHASE_GRAPH_RELEASE] = bench_now_seconds() - started;
        if (context->phase_sample_count < context->phase_sample_capacity) {
            for (int phase = 0; phase < PHASE_COUNT; ++phase) {
                context->phase_samples[phase][context->phase_sample_count] =
                    phase_times[phase];
            }
            ++context->phase_sample_count;
        }
    }
    return status;
}

static void reset_phase_samples(void* opaque)
{
    nn_bench_context* context = (nn_bench_context*)opaque;
    context->phase_sample_count = 0;
    tensor_alloc_stats_reset_counters();
}

static void report_allocation_stats(const bench_options* options,
                                    FILE* csv,
                                    int requested_threads,
                                    int measured_threads,
                                    const nn_bench_context* context)
{
    double calls = context->phase_sample_count > 0
                 ? (double)context->phase_sample_count : 1.0;
    const tensor_alloc_stats* stats = &context->allocation_stats;
    bench_record_scalar(options, csv, "nn_phase", "tiny_lm_allocations",
                        "[Bx128]->[Bx128x256]", "isolated-phase",
                        "alloc/call", requested_threads, measured_threads,
                        (double)stats->allocations / calls);
    bench_record_scalar(options, csv, "nn_phase", "tiny_lm_allocated_bytes",
                        "[Bx128]->[Bx128x256]", "isolated-phase",
                        "bytes/call", requested_threads, measured_threads,
                        (double)stats->allocated_bytes / calls);
    bench_record_scalar(options, csv, "nn_phase", "tiny_lm_peak_live_bytes",
                        "[Bx128]->[Bx128x256]", "isolated-phase",
                        "bytes", requested_threads, measured_threads,
                        (double)(stats->peak_live_bytes >=
                                 context->allocation_baseline_bytes
                             ? stats->peak_live_bytes -
                               context->allocation_baseline_bytes : 0));
}

static void report_backward_stats(const bench_options* options,
                                  FILE* csv,
                                  int requested_threads,
                                  int measured_threads,
                                  const nn_bench_context* context)
{
    static const char* names[AG_BACKWARD_OP_COUNT] = {
        "add", "sub", "mul", "div", "neg", "exp", "log", "pow",
        "sqrt", "relu", "sigmoid", "tanh", "gelu", "matmul", "sum",
        "mean", "max", "reshape", "transpose", "slice", "expand",
        "gather_rows", "mul_scalar", "div_scalar", "layer_norm",
        "softmax", "log_softmax", "cross_entropy"
    };
    const ag_backward_stats* stats = &context->backward_stats;
    for (int operation = 0; operation < AG_BACKWARD_OP_COUNT; ++operation) {
        if (stats->operation_calls[operation] == 0) continue;
        char name[64];
        snprintf(name, sizeof(name), "backward_op_%s", names[operation]);
        bench_record_scalar(options, csv, "nn_backward", name,
            "[Bx128]->[Bx128x256]", "profiled-single-call", "ms/call",
            requested_threads, measured_threads,
            stats->operation_seconds[operation] * 1000.0);
    }
    static const char* engine_names[] = {
        "backward_graph_traversal", "backward_shape_reduction",
        "backward_gradient_accumulation", "backward_persistent_merge"
    };
    const double engine_values[] = {
        stats->traversal_seconds, stats->reduction_seconds,
        stats->accumulation_seconds, stats->merge_seconds
    };
    for (int index = 0; index < 4; ++index) {
        bench_record_scalar(options, csv, "nn_backward", engine_names[index],
            "[Bx128]->[Bx128x256]", "profiled-single-call", "ms/call",
            requested_threads, measured_threads,
            engine_values[index] * 1000.0);
    }
}

static int compare_double(const void* left, const void* right)
{
    double a = *(const double*)left;
    double b = *(const double*)right;
    return (a > b) - (a < b);
}

static void report_training_phases(const bench_options* options,
                                   FILE* csv,
                                   int requested_threads,
                                   int measured_threads,
                                   nn_bench_context* context)
{
    static const char* names[PHASE_COUNT] = {
        "tiny_lm_zero_grad", "tiny_lm_forward_phase", "tiny_lm_loss",
        "tiny_lm_backward", "tiny_lm_adamw", "tiny_lm_graph_release"
    };
    if (context->phase_sample_count <= 0) return;
    for (int phase = 0; phase < PHASE_COUNT; ++phase) {
        bench_case benchmark = {
            "nn_phase", names[phase], "[Bx128]->[Bx128x256]",
            "isolated-phase", "ms/call", 0.0, 0, NULL, NULL
        };
        bench_measurement result;
        qsort(context->phase_samples[phase],
              (size_t)context->phase_sample_count, sizeof(double), compare_double);
        result.median_seconds =
            context->phase_samples[phase][context->phase_sample_count / 2];
        result.p95_seconds = context->phase_samples[phase][
            (95 * context->phase_sample_count - 1) / 100];
        result.checksum = result.median_seconds;
        result.iterations_per_sample = 1;
        benchmark.context = context;
        bench_record_measurement(options, csv, &benchmark,
                                 requested_threads, measured_threads, &result);
    }
}

static int run_nn_case(const bench_options* options,
                       FILE* csv,
                       const char* suite,
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
        suite, name, shape, layout, metric, items, 1,
        nn_operation, context
    };
    benchmark.reset = context->train ? reset_phase_samples : NULL;
    return bench_execute_case(options, csv, &benchmark, threads, result);
}

static void destroy_context(nn_bench_context* context)
{
    for (int phase = 0; phase < PHASE_COUNT; ++phase) {
        free(context->phase_samples[phase]);
    }
    nn_sgd_destroy(context->sgd);
    nn_adamw_destroy(context->adamw);
    t_free(context->targets);
    ag_tensor_release(context->input);
    if (context->module != NULL) context->module->destroy(context->module);
    memset(context, 0, sizeof(*context));
}

static int allocate_phase_samples(nn_bench_context* context, int sample_count)
{
    int capacity = sample_count * 4;
    if (capacity < 64) capacity = 64;
    context->phase_sample_capacity = capacity;
    for (int phase = 0; phase < PHASE_COUNT; ++phase) {
        context->phase_samples[phase] =
            (double*)calloc((size_t)capacity, sizeof(double));
        if (context->phase_samples[phase] == NULL) return 1;
    }
    return 0;
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
                         int backward,
                         int batch,
                         int time,
                         int channels)
{
    int dims[3] = {batch, time, channels};
    int linear_dims[2] = {batch * time, channels};
    nn_bench_context context;
    bench_measurement result;
    const char* name = NULL;
    const char* shape = "[BxTxC]";
    char backward_name[64];

    memset(&context, 0, sizeof(context));
    nn_rng_seed(&context.rng, UINT64_C(0xB34C4));
    if (kind == COMPONENT_LINEAR) {
        nn_linear* model = nn_linear_create("bench_linear", channels,
            channels * 3, 1, NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO,
            &context.rng);
        context.model = model;
        context.module = model == NULL ? NULL : &model->base;
        context.forward = linear_forward;
        context.input = make_input(2, linear_dims, 0);
        name = "linear_qkv";
        shape = "[BTxC]->[BTx3C]";
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
            "bench_attention", channels, 6, 0.0f, &context.rng);
        context.model = model;
        context.module = model == NULL ? NULL : &model->base;
        context.forward = attention_forward;
        context.input = make_input(3, dims, 0);
        name = "causal_attention";
    } else {
        nn_decoder_block* model = nn_decoder_block_create(
            "bench_block", channels, 6, 0.0f, 1e-5f, &context.rng);
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
    context.component_backward = backward;
    if (backward) context.input->requires_grad = 1;
    nn_module_set_training(context.module, 0);
    if (backward) {
        snprintf(backward_name, sizeof(backward_name), "%s_backward", name);
        name = backward_name;
    }
    int status = run_nn_case(options, csv, "nn", name, shape,
                             backward ? "forward+backward" : "forward;graph-build",
                             "tokens/s",
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
    nn_mlp* model;

    memset(context, 0, sizeof(*context));
    nn_rng_seed(&context->rng, UINT64_C(0x4D4E495354));
    model = nn_mlp_create("bench_mlp", &config, &context->rng);
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
    int status = run_nn_case(options, csv, "nn",
        train ? "mnist_mlp_train_step" : "mnist_mlp_forward",
        "[Bx784]->[Bx10]", train ? "forward+loss+backward+sgd" :
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
    nn_decoder* model;

    memset(context, 0, sizeof(*context));
    nn_rng_seed(&context->rng, UINT64_C(0x71594C4D));
    model = nn_decoder_create("bench_decoder", &config, &context->rng);
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
                            const char* suite,
                            int batch,
                            int time,
                            int channels,
                            int layers,
                            int train,
                            int threads,
                            bench_measurement* measurement)
{
    nn_bench_context context;
    bench_measurement local_result;
    bench_measurement* result = measurement == NULL ? &local_result : measurement;
    if (train) {
        tensor_alloc_stats_enable(1);
        tensor_alloc_stats_reset();
    }
    if (setup_decoder(&context, batch, time, channels, layers, train) != 0) {
        if (train) tensor_alloc_stats_enable(0);
        destroy_context(&context);
        return 1;
    }
    if (train && allocate_phase_samples(&context, options->profile.sample_count) != 0) {
        destroy_context(&context);
        return 1;
    }
    if (train) {
        tensor_alloc_stats_read(&context.allocation_stats);
        context.allocation_baseline_bytes =
            context.allocation_stats.live_bytes;
        tensor_alloc_stats_reset_counters();
    }
    int status = run_nn_case(options, csv, suite,
        train ? "tiny_lm_train_step" : "tiny_lm_forward",
        "[Bx128]->[Bx128x256]",
        train ? "forward+loss+backward+adamw" : "forward;graph-build",
        "tokens/s", (double)(batch * time), threads, &context, result);
    if (status == 0 && train && strcmp(suite, "nn") == 0) {
        int measured_threads = bench_configure_threads(threads);
        tensor_alloc_stats_read(&context.allocation_stats);
        report_training_phases(options, csv, threads, measured_threads, &context);
        report_allocation_stats(options, csv, threads, measured_threads, &context);
        ag_backward_stats_reset();
        ag_backward_stats_enable(1);
        double checksum = 0.0;
        int profile_status = nn_operation(&context, &checksum);
        ag_backward_stats_enable(0);
        ag_backward_stats_read(&context.backward_stats);
        if (profile_status != 0) status = 1;
        if (status == 0) {
            report_backward_stats(options, csv, threads, measured_threads,
                                  &context);
        }
    }
    destroy_context(&context);
    if (train) {
        tensor_alloc_stats_read(&context.allocation_stats);
        if (context.allocation_stats.live_bytes != 0) status = 1;
        tensor_alloc_stats_enable(0);
    }
    return status;
}

static int run_decoder_cases(const bench_options* options, FILE* csv, int batch)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int time = smoke ? 8 : 128;
    int channels = smoke ? 24 : 192;
    int layers = smoke ? 1 : 4;
    int forward_status = run_decoder_case(options, csv, "nn", batch, time,
        channels, layers, 0, options->threads[0], NULL);
    int status = forward_status == 1;
    for (int index = 0; index < options->thread_count; ++index) {
        int train_status = run_decoder_case(options, csv, "nn", batch, time,
            channels, layers, 1, options->threads[index], NULL);
        if (train_status == 1) status = 1;
    }
    return status;
}

int bench_run_decoder_scaling(const bench_options* options, FILE* csv)
{
    int smoke = strcmp(options->profile.profile, "smoke") == 0;
    int batch = smoke ? 1 : 2;
    int time = smoke ? 8 : 128;
    int channels = smoke ? 24 : 192;
    int layers = smoke ? 1 : 4;
    int status = 0;

    for (int train = 0; train <= 1; ++train) {
        double baseline = 0.0;
        printf(" tiny_lm_%s\n", train ? "train_step" : "forward");
        for (int index = 0; index < options->thread_count; ++index) {
            bench_measurement result;
            int threads = options->threads[index];
            int case_status = run_decoder_case(options, csv, "scaling",
                batch, time, channels, layers, train, threads, &result);
            if (case_status == 1) {
                status = 1;
                continue;
            }
            if (case_status == 2) {
                continue;
            }
            if (baseline == 0.0) baseline = result.median_seconds;
            double speedup = baseline / result.median_seconds;
            printf("    speedup=%6.2fx efficiency=%5.1f%%\n",
                   speedup, speedup * 100.0 / (double)threads);
        }
    }
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
    status |= run_component(options, csv, COMPONENT_LINEAR, 0,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_LINEAR, 1,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_LAYER_NORM, 0,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_LAYER_NORM, 1,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_ATTENTION, 0,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_ATTENTION, 1,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_BLOCK, 0,
                            batch, time, channels);
    status |= run_component(options, csv, COMPONENT_BLOCK, 1,
                            batch, time, channels);
    status |= run_mlp(options, csv, smoke ? 2 : 64, 0);
    status |= run_mlp(options, csv, smoke ? 2 : 64, 1);
    status |= run_decoder_cases(options, csv, batch);
    printf("\n");
    return status;
}
