#include "bench_runner.h"

#include <stdio.h>
#include <string.h>

#include <tensorlib/tensor.h>
#include "../src/tensor/tensor_matmul_internal.h"

typedef enum {
    LINEAR_DINPUT,
    LINEAR_DWEIGHT,
    LINEAR_DWEIGHT_DIRECT,
    LINEAR_DBIAS,
    LINEAR_PACK
} linear_gradient_kind;

typedef struct {
    linear_gradient_kind kind;
    tensor* input;
    tensor* output_gradient;
    tensor* weight;
    tensor_matmul_packed_rhs* packed_weight;
} linear_gradient_context;

typedef struct {
    const char* name;
    int in_features;
    int out_features;
} linear_shape;

typedef struct {
    int probability_value;
    tensor* left;
    tensor* right;
    tensor* output_gradient;
} attention_gradient_context;

static void fill_tensor(tensor* value, float scale)
{
    int count = tensor_numel(value);
    for (int index = 0; index < count; ++index) {
        value->storage->data[index] = scale * (float)((index % 31) - 15);
    }
}

static void destroy_linear_context(linear_gradient_context* context)
{
    t_free_matmul_packed_rhs(context->packed_weight);
    t_free(context->weight);
    t_free(context->output_gradient);
    t_free(context->input);
    memset(context, 0, sizeof(*context));
}

static int init_linear_context(linear_gradient_context* context,
                               linear_gradient_kind kind,
                               int in_features,
                               int out_features)
{
    int input_dims[3] = {4, 128, in_features};
    int output_dims[3] = {4, 128, out_features};
    int weight_dims[2] = {out_features, in_features};
    memset(context, 0, sizeof(*context));
    context->kind = kind;
    context->input = t_alloc(3, input_dims);
    context->output_gradient = t_alloc(3, output_dims);
    context->weight = t_alloc(2, weight_dims);
    if (context->input == NULL || context->output_gradient == NULL ||
        context->weight == NULL) return 1;
    fill_tensor(context->input, 0.001f);
    fill_tensor(context->output_gradient, 0.002f);
    fill_tensor(context->weight, 0.003f);
    if (kind == LINEAR_DINPUT) {
        context->packed_weight = t_pack_matmul_rhs(context->weight);
        if (context->packed_weight == NULL) return 1;
    }
    return 0;
}

static int linear_gradient_operation(void* opaque, double* checksum)
{
    linear_gradient_context* context = (linear_gradient_context*)opaque;
    tensor* first = NULL;
    tensor* second = NULL;
    tensor* output = NULL;
    if (context->kind == LINEAR_DINPUT) {
        output = t_matmul_packed_rhs(context->output_gradient,
                                     context->packed_weight);
    } else if (context->kind == LINEAR_DWEIGHT) {
        first = t_transpose(context->input, 1, 2);
        second = first == NULL ? NULL :
                 t_matmul(first, context->output_gradient);
        output = second == NULL ? NULL : t_sum(second, 0);
    } else if (context->kind == LINEAR_DWEIGHT_DIRECT) {
        first = t_transpose(context->weight, 0, 1);
        output = first == NULL ? NULL : tensor_matmul_backward_rhs(
            context->input, context->output_gradient, first);
    } else if (context->kind == LINEAR_DBIAS) {
        first = t_sum(context->output_gradient, 0);
        output = first == NULL ? NULL : t_sum(first, 0);
    } else {
        tensor_matmul_packed_rhs* packed =
            t_pack_matmul_rhs(context->weight);
        if (packed == NULL) return 1;
        *checksum += 1.0;
        t_free_matmul_packed_rhs(packed);
        return 0;
    }
    if (output == NULL) {
        t_free(second);
        t_free(first);
        return 1;
    }
    *checksum += output->storage->data[output->offset];
    t_free(output);
    t_free(second);
    t_free(first);
    return 0;
}

static int run_linear_gradient(const bench_options* options,
                               FILE* csv,
                               const linear_shape* shape,
                               linear_gradient_kind kind,
                               const char* suffix,
                               const char* layout)
{
    linear_gradient_context context;
    bench_measurement result;
    char name[64];
    char dimensions[64];
    snprintf(name, sizeof(name), "%s_%s", shape->name, suffix);
    snprintf(dimensions, sizeof(dimensions), "[4x128x%d]->[4x128x%d]",
             shape->in_features, shape->out_features);
    if (init_linear_context(&context, kind, shape->in_features,
                            shape->out_features) != 0) {
        destroy_linear_context(&context);
        return 1;
    }
    bench_case benchmark = {
        "nn_backward", name, dimensions, layout, "ms/call", 0.0, 1,
        linear_gradient_operation, &context, NULL
    };
    int status = bench_execute_case(options, csv, &benchmark,
                                    options->threads[0], &result);
    destroy_linear_context(&context);
    return status == 1;
}

static void destroy_attention_context(attention_gradient_context* context)
{
    t_free(context->output_gradient);
    t_free(context->right);
    t_free(context->left);
    memset(context, 0, sizeof(*context));
}

static int init_attention_context(attention_gradient_context* context,
                                  int probability_value)
{
    int square_dims[3] = {24, 128, 128};
    int head_dims[3] = {24, 128, 32};
    memset(context, 0, sizeof(*context));
    context->probability_value = probability_value;
    context->left = t_alloc(3, probability_value ? square_dims : head_dims);
    context->right = t_alloc(3, head_dims);
    context->output_gradient =
        t_alloc(3, probability_value ? head_dims : square_dims);
    if (context->left == NULL || context->right == NULL ||
        context->output_gradient == NULL) return 1;
    fill_tensor(context->left, 0.001f);
    fill_tensor(context->right, 0.002f);
    fill_tensor(context->output_gradient, 0.003f);
    return 0;
}

static int attention_gradient_operation(void* opaque, double* checksum)
{
    attention_gradient_context* context =
        (attention_gradient_context*)opaque;
    tensor* left_transpose = NULL;
    tensor* right_transpose = NULL;
    tensor* left_gradient = NULL;
    tensor* right_gradient = NULL;
    if (context->probability_value) {
        right_transpose = t_transpose(context->right, 1, 2);
        left_transpose = t_transpose(context->left, 1, 2);
        left_gradient = right_transpose == NULL ? NULL :
            t_matmul(context->output_gradient, right_transpose);
        right_gradient = left_transpose == NULL ? NULL :
            t_matmul(left_transpose, context->output_gradient);
    } else {
        left_transpose = t_transpose(context->output_gradient, 1, 2);
        left_gradient = t_matmul(context->output_gradient, context->right);
        right_gradient = left_transpose == NULL ? NULL :
            t_matmul(left_transpose, context->left);
    }
    if (left_gradient == NULL || right_gradient == NULL) {
        t_free(right_gradient); t_free(left_gradient);
        t_free(right_transpose); t_free(left_transpose);
        return 1;
    }
    *checksum += left_gradient->storage->data[left_gradient->offset] +
                 right_gradient->storage->data[right_gradient->offset];
    t_free(right_gradient); t_free(left_gradient);
    t_free(right_transpose); t_free(left_transpose);
    return 0;
}

static int run_attention_gradient(const bench_options* options,
                                  FILE* csv,
                                  int probability_value)
{
    attention_gradient_context context;
    bench_measurement result;
    if (init_attention_context(&context, probability_value) != 0) {
        destroy_attention_context(&context);
        return 1;
    }
    bench_case benchmark = {
        "nn_backward",
        probability_value ? "attention_probability_value_backward" :
                            "attention_qk_transpose_backward",
        probability_value ? "[24x128x128]@[24x128x32]" :
                            "[24x128x32]@[24x32x128]",
        "two-gradient-matmuls", "ms/call", 0.0, 1,
        attention_gradient_operation, &context, NULL
    };
    int status = bench_execute_case(options, csv, &benchmark,
                                    options->threads[0], &result);
    destroy_attention_context(&context);
    return status == 1;
}

int bench_run_backward_matrix_suite(const bench_options* options, FILE* csv)
{
    static const linear_shape shapes[] = {
        {"linear_qkv", 192, 576},
        {"linear_attention_output", 192, 192},
        {"linear_mlp_expand", 192, 768},
        {"linear_mlp_project", 768, 192},
        {"linear_vocabulary", 192, 256}
    };
    int status = 0;
    printf("Backward matrix components\n");
    for (size_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]); ++index) {
        status |= run_linear_gradient(options, csv, &shapes[index],
                                      LINEAR_DINPUT, "dinput",
                                      "packed-weight");
        status |= run_linear_gradient(options, csv, &shapes[index],
                                      LINEAR_DWEIGHT, "dweight",
                                      "batched-matmul+reduce");
        status |= run_linear_gradient(options, csv, &shapes[index],
                                      LINEAR_DWEIGHT_DIRECT, "dweight_direct",
                                      "flattened-transpose-left");
        status |= run_linear_gradient(options, csv, &shapes[index],
                                      LINEAR_DBIAS, "dbias",
                                      "two-axis-reduce");
        status |= run_linear_gradient(options, csv, &shapes[index],
                                      LINEAR_PACK, "pack", "packing-only");
    }
    status |= run_attention_gradient(options, csv, 0);
    status |= run_attention_gradient(options, csv, 1);
    printf("\n");
    return status;
}
