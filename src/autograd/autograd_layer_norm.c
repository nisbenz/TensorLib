#include <math.h>
#include <stdlib.h>

#include "../../include/tensorlib/autograd_internal.h"

typedef struct {
    int rows;
    int width;
    float* means;
    float* inverse_stds;
} layer_norm_context;

static void free_layer_norm_context(void* opaque)
{
    layer_norm_context* context = (layer_norm_context*)opaque;
    if (context == NULL) return;
    free(context->means);
    free(context->inverse_stds);
    free(context);
}

static int row_base(const tensor* value, int row, int width)
{
    if (is_contiguous((tensor*)value)) return value->offset + row * width;
    int base = value->offset;
    int remaining = row;
    for (int axis = value->ndim - 2; axis >= 0; --axis) {
        int coordinate = remaining % value->dims[axis];
        remaining /= value->dims[axis];
        base += coordinate * value->strides[axis];
    }
    return base;
}

static int backward_layer_norm(const ag_node* node,
                               const tensor* output_gradient,
                               tensor** input_gradients)
{
    layer_norm_context* context;
    tensor* input;
    tensor* weight;
    int width;
    float width_f;
    tensor* input_gradient = NULL;
    tensor* weight_gradient = NULL;
    tensor* bias_gradient = NULL;

    if (node == NULL || input_gradients == NULL ||
        !tensor_has_valid_metadata(output_gradient)) return 1;
    context = (layer_norm_context*)node->context;
    if (context == NULL) return 1;
    input = node->inputs[0]->value;
    weight = node->input_count == 3 ? node->inputs[1]->value : NULL;
    width = context->width;
    width_f = (float)width;
    if (node->inputs[0]->requires_grad) {
        input_gradient = t_alloc(input->ndim, input->dims);
        if (input_gradient == NULL) goto fail;
    }
    if (node->input_count == 3 && node->inputs[1]->requires_grad) {
        weight_gradient = t_alloc(1, &width);
        if (weight_gradient == NULL) goto fail;
    }
    if (node->input_count == 3 && node->inputs[2]->requires_grad) {
        bias_gradient = t_alloc(1, &width);
        if (bias_gradient == NULL) goto fail;
    }
    if (weight_gradient != NULL) {
        for (int k = 0; k < width; ++k) weight_gradient->storage->data[k] = 0.0f;
    }
    if (bias_gradient != NULL) {
        for (int k = 0; k < width; ++k) bias_gradient->storage->data[k] = 0.0f;
    }

    for (int row = 0; row < context->rows; ++row) {
        int input_base = row_base(input, row, width);
        int gradient_base = row_base(output_gradient, row, width);
        float sum = 0.0f;
        float weighted_sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            float normalized = (input->storage->data[input_base + k *
                              input->strides[input->ndim - 1]] - context->means[row]) *
                               context->inverse_stds[row];
            float upstream = output_gradient->storage->data[gradient_base + k *
                              output_gradient->strides[output_gradient->ndim - 1]];
            float scale = weight == NULL ? 1.0f : weight->storage->data[k];
            sum += upstream * scale;
            weighted_sum += upstream * scale * normalized;
            if (weight_gradient != NULL) {
                weight_gradient->storage->data[k] += upstream * normalized;
            }
            if (bias_gradient != NULL) bias_gradient->storage->data[k] += upstream;
        }
        if (input_gradient != NULL) {
            for (int k = 0; k < width; ++k) {
                float normalized = (input->storage->data[input_base + k *
                                  input->strides[input->ndim - 1]] - context->means[row]) *
                                   context->inverse_stds[row];
                float upstream = output_gradient->storage->data[gradient_base + k *
                                  output_gradient->strides[output_gradient->ndim - 1]];
                float scale = weight == NULL ? 1.0f : weight->storage->data[k];
                input_gradient->storage->data[row * width + k] =
                    context->inverse_stds[row] *
                    (upstream * scale - sum / width_f -
                     normalized * weighted_sum / width_f);
            }
        }
    }
    input_gradients[0] = input_gradient;
    if (node->input_count == 3) {
        input_gradients[1] = weight_gradient;
        input_gradients[2] = bias_gradient;
    }
    return 0;

fail:
    t_free(input_gradient);
    t_free(weight_gradient);
    t_free(bias_gradient);
    return 1;
}

ag_tensor* ag_layer_norm(const ag_tensor* input,
                         const ag_tensor* weight,
                         const ag_tensor* bias,
                         float epsilon)
{
    tensor* output = NULL;
    layer_norm_context* context = NULL;
    ag_tensor* inputs[3];
    int width;
    float width_f;

    if (input == NULL || !tensor_has_valid_metadata(input->value) ||
        input->value->ndim < 1 || !isfinite(epsilon) || epsilon <= 0.0f) return NULL;
    width = input->value->dims[input->value->ndim - 1];
    width_f = (float)width;
    if ((weight == NULL) != (bias == NULL) ||
        (weight != NULL && (!tensor_has_valid_metadata(weight->value) ||
                            !tensor_has_valid_metadata(bias->value) ||
                            weight->value->ndim != 1 || bias->value->ndim != 1 ||
                            weight->value->dims[0] != width || bias->value->dims[0] != width))) {
        return NULL;
    }
    output = t_alloc(input->value->ndim, input->value->dims);
    context = (layer_norm_context*)calloc(1, sizeof(*context));
    if (output == NULL || context == NULL) goto fail;
    context->width = width;
    context->rows = tensor_numel(input->value) / width;
    context->means = (float*)calloc((size_t)context->rows, sizeof(float));
    context->inverse_stds = (float*)calloc((size_t)context->rows, sizeof(float));
    if (context->means == NULL || context->inverse_stds == NULL) goto fail;

    for (int row = 0; row < context->rows; ++row) {
        int base = row_base(input->value, row, width);
        float mean = 0.0f;
        float variance = 0.0f;
        for (int k = 0; k < width; ++k) mean += input->value->storage->data[
            base + k * input->value->strides[input->value->ndim - 1]];
        mean /= width_f;
        for (int k = 0; k < width; ++k) {
            float delta = input->value->storage->data[
                base + k * input->value->strides[input->value->ndim - 1]] - mean;
            variance += delta * delta;
        }
        context->means[row] = mean;
        context->inverse_stds[row] = 1.0f / sqrtf(variance / width_f + epsilon);
        for (int k = 0; k < width; ++k) {
            float normalized = (input->value->storage->data[
                base + k * input->value->strides[input->value->ndim - 1]] - mean) *
                context->inverse_stds[row];
            output->storage->data[row * width + k] =
                normalized * (weight == NULL ? 1.0f : weight->value->storage->data[k]) +
                (bias == NULL ? 0.0f : bias->value->storage->data[k]);
        }
    }
    inputs[0] = (ag_tensor*)input;
    if (weight == NULL) return ag_make_result(output, AG_OP_LAYER_NORM, 1,
                                               inputs, backward_layer_norm,
                                               context, free_layer_norm_context);
    inputs[1] = (ag_tensor*)weight;
    inputs[2] = (ag_tensor*)bias;
    return ag_make_result(output, AG_OP_LAYER_NORM, 3, inputs,
                          backward_layer_norm, context, free_layer_norm_context);

fail:
    t_free(output);
    free_layer_norm_context(context);
    return NULL;
}
