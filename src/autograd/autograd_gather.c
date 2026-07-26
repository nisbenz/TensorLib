#include <stdlib.h>

#include "../../include/tensorlib/autograd_internal.h"

typedef struct {
    tensor* indices;
} gather_context;

static int tensor_flat_index(const tensor* value, int flat)
{
    int index = value->offset;
    int remaining = flat;

    for (int dim = value->ndim - 1; dim >= 0; --dim) {
        int coordinate = remaining % value->dims[dim];
        remaining /= value->dims[dim];
        index += coordinate * value->strides[dim];
    }
    return index;
}

static void free_gather_context(void* raw_context)
{
    gather_context* context = (gather_context*)raw_context;

    if (context == NULL) return;
    t_free(context->indices);
    free(context);
}

static int backward_gather_rows(const ag_node* node,
                                const tensor* output_gradient,
                                tensor** input_gradients)
{
    const gather_context* context = (const gather_context*)node->context;
    const tensor* table = node->inputs[0]->value;
    tensor* gradient;
    int index_count;
    int width;

    if (context == NULL || !tensor_has_valid_metadata(context->indices) ||
        !tensor_has_valid_metadata(output_gradient) ||
        !tensor_has_valid_metadata(table)) {
        return 1;
    }
    gradient = t_alloc(table->ndim, table->dims);
    if (gradient == NULL) return 1;
    for (int i = 0; i < tensor_numel(gradient); ++i) {
        gradient->storage->data[i] = 0.0f;
    }

    index_count = tensor_numel(context->indices);
    width = table->dims[1];
    for (int item = 0; item < index_count; ++item) {
        int row = (int)context->indices->storage->data[
            tensor_flat_index(context->indices, item)];
        for (int column = 0; column < width; ++column) {
            int output_index = tensor_flat_index(
                output_gradient, item * width + column);
            gradient->storage->data[row * width + column] +=
                output_gradient->storage->data[output_index];
        }
    }
    input_gradients[0] = gradient;
    return 0;
}

ag_tensor* ag_gather_rows(const ag_tensor* table, const tensor* indices)
{
    gather_context* context;
    tensor* output;
    ag_tensor* inputs[1];

    if (table == NULL || !tensor_has_valid_metadata(table->value) ||
        !tensor_has_valid_metadata(indices)) {
        return NULL;
    }
    output = t_gather_rows(table->value, (tensor*)indices);
    if (output == NULL) return NULL;
    context = (gather_context*)calloc(1, sizeof(*context));
    if (context == NULL) {
        t_free(output);
        return NULL;
    }
    context->indices = t_clone((tensor*)indices);
    if (context->indices == NULL) {
        free(context);
        t_free(output);
        return NULL;
    }
    inputs[0] = (ag_tensor*)table;
    return ag_make_result(output, AG_OP_GATHER_ROWS, 1, inputs,
                          backward_gather_rows, context,
                          free_gather_context);
}
