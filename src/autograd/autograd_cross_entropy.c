#include <math.h>
#include <stdlib.h>

#include "../../include/tensorlib/autograd_internal.h"
#include "../tensor/parallel.h"

#define TENSORLIB_CROSS_ENTROPY_MIN_PARALLEL_ELEMENTS (1 << 16)

typedef struct {
    int rows;
    int classes;
    int* targets;
} cross_entropy_context;

static void free_cross_entropy_context(void* opaque)
{
    cross_entropy_context* context = (cross_entropy_context*)opaque;
    if (context == NULL) return;
    free(context->targets);
    free(context);
}

static int flat_index(const tensor* value, int flat)
{
    int index = value->offset;
    int remaining = flat;
    for (int axis = value->ndim - 1; axis >= 0; --axis) {
        int coordinate = remaining % value->dims[axis];
        remaining /= value->dims[axis];
        index += coordinate * value->strides[axis];
    }
    return index;
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

static int backward_cross_entropy(const ag_node* node,
                                  const tensor* output_gradient,
                                  tensor** input_gradients)
{
    cross_entropy_context* context;
    tensor* input;
    tensor* gradient;
    float upstream;

    if (node == NULL || input_gradients == NULL ||
        !tensor_has_valid_metadata(output_gradient)) return 1;
    context = (cross_entropy_context*)node->context;
    if (context == NULL || !node->inputs[0]->requires_grad) return 0;
    input = node->inputs[0]->value;
    gradient = t_alloc(input->ndim, input->dims);
    if (gradient == NULL) return 1;
    upstream = output_gradient->storage->data[output_gradient->offset] /
               (float)context->rows;
    int threads = tensorlib_parallel_threads(
        (long long)context->rows * context->classes,
        TENSORLIB_CROSS_ENTROPY_MIN_PARALLEL_ELEMENTS, context->rows);
#ifdef _OPENMP
#pragma omp parallel for if(threads > 1) schedule(static) num_threads(threads)
#endif
    for (int row = 0; row < context->rows; ++row) {
        int base = row_base(input, row, context->classes);
        float maximum = -INFINITY;
        float sum = 0.0f;
        for (int k = 0; k < context->classes; ++k) {
            float value = input->storage->data[base + k *
                input->strides[input->ndim - 1]];
            if (isnan(value)) maximum = value;
            else if (value > maximum) maximum = value;
        }
        for (int k = 0; k < context->classes; ++k) {
            sum += expf(input->storage->data[base + k *
                input->strides[input->ndim - 1]] - maximum);
        }
        for (int k = 0; k < context->classes; ++k) {
            float probability = expf(input->storage->data[base + k *
                input->strides[input->ndim - 1]] - maximum) / sum;
            gradient->storage->data[row * context->classes + k] = upstream *
                (probability - (k == context->targets[row] ? 1.0f : 0.0f));
        }
    }
    input_gradients[0] = gradient;
    return 0;
}

ag_tensor* ag_cross_entropy(const ag_tensor* logits, const tensor* targets)
{
    tensor* output = NULL;
    cross_entropy_context* context = NULL;
    int classes;
    int rows;

    if (logits == NULL || !tensor_has_valid_metadata(logits->value) ||
        !tensor_has_valid_metadata(targets) || logits->value->ndim < 1 ||
        targets->ndim != logits->value->ndim - 1) return NULL;
    for (int axis = 0; axis < targets->ndim; ++axis) {
        if (targets->dims[axis] != logits->value->dims[axis]) return NULL;
    }
    classes = logits->value->dims[logits->value->ndim - 1];
    rows = tensor_numel((tensor*)targets);
    context = (cross_entropy_context*)calloc(1, sizeof(*context));
    output = t_alloc(0, NULL);
    if (context == NULL || output == NULL) goto fail;
    context->rows = rows;
    context->classes = classes;
    context->targets = (int*)malloc((size_t)rows * sizeof(*context->targets));
    if (context->targets == NULL) goto fail;
    for (int row = 0; row < rows; ++row) {
        float target = targets->storage->data[flat_index(targets, row)];
        if (!isfinite(target) || floorf(target) != target || target < 0.0f ||
            target >= (float)classes) goto fail;
        context->targets[row] = (int)target;
    }

    float loss = 0.0f;
    for (int row = 0; row < rows; ++row) {
        int base = row_base(logits->value, row, classes);
        float maximum = -INFINITY;
        float sum = 0.0f;
        for (int k = 0; k < classes; ++k) {
            float value = logits->value->storage->data[base + k *
                logits->value->strides[logits->value->ndim - 1]];
            if (isnan(value)) maximum = value;
            else if (value > maximum) maximum = value;
        }
        for (int k = 0; k < classes; ++k) {
            sum += expf(logits->value->storage->data[base + k *
                logits->value->strides[logits->value->ndim - 1]] - maximum);
        }
        loss -= logits->value->storage->data[base +
            context->targets[row] * logits->value->strides[
                logits->value->ndim - 1]] - maximum - logf(sum);
    }
    output->storage->data[0] = loss / (float)rows;
    {
        ag_tensor* inputs[1] = {(ag_tensor*)logits};
        return ag_make_result(output, AG_OP_CROSS_ENTROPY, 1, inputs,
                              backward_cross_entropy, context,
                              free_cross_entropy_context);
    }

fail:
    t_free(output);
    free_cross_entropy_context(context);
    return NULL;
}
