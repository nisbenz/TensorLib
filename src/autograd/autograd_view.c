#include <stdlib.h>
#include <string.h>

#include "./../../include/tensorlib/autograd_internal.h"

typedef struct {
    int dim0;
    int dim1;
} transpose_context;

typedef struct {
    int dim;
    int start;
} slice_context;

static int backward_reshape(const ag_node* node,
                            const tensor* output_gradient,
                            tensor** input_gradients) {
    if (node == NULL || node->input_count != 1 || input_gradients == NULL ||
        !tensor_has_valid_metadata(output_gradient)) return 1;
    if (!node->inputs[0]->requires_grad) return 0;
    tensor* input = node->inputs[0]->value;
    input_gradients[0] = t_reshape((tensor*)output_gradient, input->ndim, input->dims);
    return input_gradients[0] == NULL;
}

static int backward_transpose(const ag_node* node,
                              const tensor* output_gradient,
                              tensor** input_gradients) {
    if (node == NULL || node->input_count != 1 || input_gradients == NULL ||
        node->context == NULL || !tensor_has_valid_metadata(output_gradient)) return 1;
    if (!node->inputs[0]->requires_grad) return 0;
    transpose_context* context = (transpose_context*)node->context;
    input_gradients[0] = t_transpose((tensor*)output_gradient, context->dim0, context->dim1);
    return input_gradients[0] == NULL;
}

static int backward_slice(const ag_node* node,
                          const tensor* output_gradient,
                          tensor** input_gradients) {
    if (node == NULL || node->input_count != 1 || input_gradients == NULL ||
        node->context == NULL || !tensor_has_valid_metadata(output_gradient)) return 1;
    if (!node->inputs[0]->requires_grad) return 0;

    tensor* input = node->inputs[0]->value;
    slice_context* context = (slice_context*)node->context;
    tensor* gradient = ag_full_like(input, 0.0f);
    if (gradient == NULL) return 1;
    if (is_contiguous((tensor*)output_gradient)) {
        size_t outer = 1;
        size_t inner = 1;
        for (int axis = 0; axis < context->dim; ++axis) {
            outer *= (size_t)input->dims[axis];
        }
        for (int axis = context->dim + 1; axis < input->ndim; ++axis) {
            inner *= (size_t)input->dims[axis];
        }
        size_t slice = (size_t)output_gradient->dims[context->dim];
        for (size_t block = 0; block < outer; ++block) {
            size_t source = (size_t)output_gradient->offset +
                            block * slice * inner;
            size_t destination = block * (size_t)input->dims[context->dim] *
                                 inner + (size_t)context->start * inner;
            memcpy(gradient->storage->data + destination,
                   output_gradient->storage->data + source,
                   slice * inner * sizeof(float));
        }
        input_gradients[0] = gradient;
        return 0;
    }
    int* output_coords = (int*)calloc((size_t)output_gradient->ndim, sizeof(int));
    int* input_coords = (int*)calloc((size_t)input->ndim, sizeof(int));
    if (output_coords == NULL || input_coords == NULL) {
        t_free(gradient); free(output_coords); free(input_coords);
        return 1;
    }

    for (int i = 0; i < tensor_numel((tensor*)output_gradient); ++i) {
        for (int axis = 0; axis < input->ndim; ++axis) input_coords[axis] = output_coords[axis];
        input_coords[context->dim] += context->start;
        int source = get_flat_index_nd((tensor*)output_gradient, output_coords);
        int destination = get_flat_index_nd(gradient, input_coords);
        gradient->storage->data[destination] = output_gradient->storage->data[source];
        advance_coords(output_coords, output_gradient->dims, output_gradient->ndim);
    }
    free(output_coords);
    free(input_coords);
    input_gradients[0] = gradient;
    return 0;
}

static int backward_expand(const ag_node* node,
                           const tensor* output_gradient,
                           tensor** input_gradients) {
    if (node == NULL || node->input_count != 1 || input_gradients == NULL ||
        !tensor_has_valid_metadata(output_gradient)) return 1;
    if (!node->inputs[0]->requires_grad) return 0;
    input_gradients[0] = t_clone((tensor*)output_gradient);
    return input_gradients[0] == NULL;
}

ag_tensor* ag_reshape(const ag_tensor* value, int new_ndim, const int* new_dims) {
    if (value == NULL) return NULL;
    tensor* output = t_reshape(value->value, new_ndim, (int*)new_dims);
    ag_tensor* inputs[1] = {(ag_tensor*)value};
    return ag_make_result(output, AG_OP_RESHAPE, 1, inputs,
                          backward_reshape, NULL, NULL);
}

ag_tensor* ag_transpose(const ag_tensor* value, int dim0, int dim1) {
    if (value == NULL) return NULL;
    tensor* output = t_transpose(value->value, dim0, dim1);
    if (output == NULL) return NULL;
    transpose_context* context = (transpose_context*)malloc(sizeof(*context));
    if (context == NULL) { t_free(output); return NULL; }
    context->dim0 = dim0;
    context->dim1 = dim1;
    ag_tensor* inputs[1] = {(ag_tensor*)value};
    return ag_make_result(output, AG_OP_TRANSPOSE, 1, inputs,
                          backward_transpose, context, free);
}

ag_tensor* ag_slice(const ag_tensor* value, int dim, int start, int end) {
    if (value == NULL) return NULL;
    tensor* output = t_slice(value->value, dim, start, end);
    if (output == NULL) return NULL;
    slice_context* context = (slice_context*)malloc(sizeof(*context));
    if (context == NULL) { t_free(output); return NULL; }
    context->dim = dim;
    context->start = start;
    ag_tensor* inputs[1] = {(ag_tensor*)value};
    return ag_make_result(output, AG_OP_SLICE, 1, inputs,
                          backward_slice, context, free);
}

ag_tensor* ag_expand(const ag_tensor* value, int new_ndim, const int* new_dims) {
    if (value == NULL) return NULL;
    tensor* output = t_expand(value->value, new_ndim, new_dims);
    ag_tensor* inputs[1] = {(ag_tensor*)value};
    return ag_make_result(output, AG_OP_EXPAND, 1, inputs,
                          backward_expand, NULL, NULL);
}
