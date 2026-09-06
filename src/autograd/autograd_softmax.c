#include <math.h>
#include <stdlib.h>

#include "../../include/tensorlib/autograd_internal.h"
#include "../tensor/parallel.h"

#define TENSORLIB_SOFTMAX_MIN_PARALLEL_ELEMENTS (1 << 16)

typedef struct {
    int rows;
    int width;
    int log_softmax;
    int causal;
} softmax_context;

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

static int allowed_width(const softmax_context* context, int row)
{
    return context->causal ? row % context->width + 1 : context->width;
}

static int backward_softmax(const ag_node* node,
                            const tensor* output_gradient,
                            tensor** input_gradients)
{
    softmax_context* context;
    tensor* output;
    tensor* gradient;

    if (node == NULL || input_gradients == NULL ||
        !tensor_has_valid_metadata(output_gradient)) return 1;
    context = (softmax_context*)node->context;
    if (context == NULL || !node->inputs[0]->requires_grad) return 0;
    output = node->output->value;
    gradient = t_alloc(output->ndim, output->dims);
    if (gradient == NULL) return 1;

    int threads = tensorlib_parallel_threads(
        (long long)context->rows * context->width,
        TENSORLIB_SOFTMAX_MIN_PARALLEL_ELEMENTS, context->rows);
#ifdef _OPENMP
#pragma omp parallel for if(threads > 1) schedule(static) num_threads(threads)
#endif
    for (int row = 0; row < context->rows; ++row) {
        int width = allowed_width(context, row);
        int output_base = row_base(output, row, context->width);
        int upstream_base = row_base(output_gradient, row, context->width);
        float total = 0.0f;
        if (context->log_softmax) {
            for (int k = 0; k < width; ++k) {
                total += output_gradient->storage->data[upstream_base + k *
                    output_gradient->strides[output_gradient->ndim - 1]];
            }
        } else {
            for (int k = 0; k < width; ++k) {
                total += output_gradient->storage->data[upstream_base + k *
                    output_gradient->strides[output_gradient->ndim - 1]] *
                    output->storage->data[output_base + k];
            }
        }
        for (int k = 0; k < context->width; ++k) {
            float upstream = output_gradient->storage->data[upstream_base + k *
                output_gradient->strides[output_gradient->ndim - 1]];
            float probability = context->log_softmax
                              ? expf(output->storage->data[output_base + k])
                              : output->storage->data[output_base + k];
            gradient->storage->data[row * context->width + k] = k < width
                ? (context->log_softmax
                    ? upstream - probability * total
                    : probability * (upstream - total))
                : 0.0f;
        }
    }
    input_gradients[0] = gradient;
    return 0;
}

ag_tensor* ag_softmax_last_dim(const ag_tensor* input,
                               int log_softmax,
                               int causal)
{
    tensor* output;
    softmax_context* context;
    int width;

    if (input == NULL || !tensor_has_valid_metadata(input->value) ||
        input->value->ndim < 1) return NULL;
    width = input->value->dims[input->value->ndim - 1];
    if (causal && (input->value->ndim < 2 ||
                   input->value->dims[input->value->ndim - 2] != width)) return NULL;
    output = t_alloc(input->value->ndim, input->value->dims);
    context = (softmax_context*)calloc(1, sizeof(*context));
    if (output == NULL || context == NULL) goto fail;
    context->width = width;
    context->rows = tensor_numel(input->value) / width;
    context->log_softmax = log_softmax != 0;
    context->causal = causal != 0;

    for (int row = 0; row < context->rows; ++row) {
        int input_base = row_base(input->value, row, width);
        int output_base = row * width;
        int count = allowed_width(context, row);
        float maximum = -INFINITY;
        float sum = 0.0f;
        for (int k = 0; k < count; ++k) {
            float value = input->value->storage->data[input_base + k *
                input->value->strides[input->value->ndim - 1]];
            if (isnan(value)) maximum = value;
            else if (value > maximum) maximum = value;
        }
        for (int k = 0; k < count; ++k) {
            float exponential = expf(input->value->storage->data[input_base + k *
                input->value->strides[input->value->ndim - 1]] - maximum);
            output->storage->data[output_base + k] = exponential;
            sum += exponential;
        }
        float log_sum = logf(sum);
        for (int k = 0; k < width; ++k) {
            if (k >= count) output->storage->data[output_base + k] =
                context->log_softmax ? -INFINITY : 0.0f;
            else if (context->log_softmax) output->storage->data[output_base + k] =
                input->value->storage->data[input_base + k *
                    input->value->strides[input->value->ndim - 1]] - maximum - log_sum;
            else output->storage->data[output_base + k] /= sum;
        }
    }
    {
        ag_tensor* inputs[1] = {(ag_tensor*)input};
        return ag_make_result(output,
                              context->log_softmax ? AG_OP_LOG_SOFTMAX : AG_OP_SOFTMAX,
                              1, inputs, backward_softmax, context, free);
    }

fail:
    t_free(output);
    free(context);
    return NULL;
}
