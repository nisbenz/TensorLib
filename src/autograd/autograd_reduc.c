#include <math.h>
#include <stdlib.h>

#include "./../../include/tensorlib/autograd_internal.h"

typedef struct {
    int dim;
    int keepdim;
} reduction_context;

static tensor* expand_reduction_gradient(const ag_node* node,
                                         const tensor* output_gradient) {
    reduction_context* context = (reduction_context*)node->context;
    tensor* shaped = context->keepdim
                   ? t_clone((tensor*)output_gradient)
                   : t_unsqueeze((tensor*)output_gradient, context->dim);
    if (shaped == NULL) return NULL;
    tensor* expanded = t_expand(shaped, node->inputs[0]->value->ndim,
                                node->inputs[0]->value->dims);
    t_free(shaped);
    return expanded;
}

static int backward_sum(const ag_node* node,
                        const tensor* output_gradient,
                        tensor** input_gradients) {
    if (node == NULL || node->input_count != 1 || node->context == NULL ||
        input_gradients == NULL || !tensor_has_valid_metadata(output_gradient)) return 1;
    if (!node->inputs[0]->requires_grad) return 0;
    input_gradients[0] = expand_reduction_gradient(node, output_gradient);
    return input_gradients[0] == NULL;
}

static int backward_mean(const ag_node* node,
                         const tensor* output_gradient,
                         tensor** input_gradients) {
    if (backward_sum(node, output_gradient, input_gradients) != 0) return 1;
    if (input_gradients[0] == NULL) return 0;
    reduction_context* context = (reduction_context*)node->context;
    tensor* scaled = t_div_scalar(input_gradients[0],
                                  (float)node->inputs[0]->value->dims[context->dim]);
    t_free(input_gradients[0]);
    input_gradients[0] = scaled;
    return scaled == NULL;
}

static void map_output_to_input(const reduction_context* context,
                                const int* output_coords,
                                int input_ndim,
                                int* input_coords) {
    for (int input_axis = 0, output_axis = 0; input_axis < input_ndim; ++input_axis) {
        if (input_axis == context->dim) {
            input_coords[input_axis] = 0;
            if (context->keepdim) output_axis++;
        } else {
            input_coords[input_axis] = output_coords[output_axis++];
        }
    }
}

static int backward_max(const ag_node* node,
                        const tensor* output_gradient,
                        tensor** input_gradients) {
    if (node == NULL || node->input_count != 1 || node->context == NULL ||
        input_gradients == NULL || !tensor_has_valid_metadata(output_gradient)) return 1;
    if (!node->inputs[0]->requires_grad) return 0;

    reduction_context* context = (reduction_context*)node->context;
    tensor* input = node->inputs[0]->value;
    tensor* output = node->output->value;
    tensor* gradient = ag_full_like(input, 0.0f);
    int* output_coords = output->ndim > 0
                       ? (int*)calloc((size_t)output->ndim, sizeof(int)) : NULL;
    int* input_coords = (int*)calloc((size_t)input->ndim, sizeof(int));
    if (gradient == NULL || (output->ndim > 0 && output_coords == NULL) || input_coords == NULL) {
        t_free(gradient); free(output_coords); free(input_coords);
        return 1;
    }

    for (int output_element = 0; output_element < tensor_numel(output); ++output_element) {
        map_output_to_input(context, output_coords, input->ndim, input_coords);
        int output_index = get_flat_index_nd(output, output_coords);
        int upstream_index = get_flat_index_nd((tensor*)output_gradient, output_coords);
        float maximum = output->storage->data[output_index];
        float upstream = output_gradient->storage->data[upstream_index];
        int ties = 0;

        if (!isnan(maximum)) {
            for (int reduced = 0; reduced < input->dims[context->dim]; ++reduced) {
                input_coords[context->dim] = reduced;
                int input_index = get_flat_index_nd(input, input_coords);
                if (input->storage->data[input_index] == maximum) ties++;
            }
        }

        for (int reduced = 0; reduced < input->dims[context->dim]; ++reduced) {
            input_coords[context->dim] = reduced;
            int input_index = get_flat_index_nd(input, input_coords);
            int gradient_index = get_flat_index_nd(gradient, input_coords);
            if (isnan(maximum)) {
                gradient->storage->data[gradient_index] = NAN;
            } else if (input->storage->data[input_index] == maximum) {
                gradient->storage->data[gradient_index] = upstream / (float)ties;
            }
        }
        advance_coords(output_coords, output->dims, output->ndim);
    }

    free(output_coords);
    free(input_coords);
    input_gradients[0] = gradient;
    return 0;
}

typedef tensor* (*reduction_forward_fn)(tensor*, int);

static ag_tensor* apply_reduction(const ag_tensor* value,
                                  int dim,
                                  int keepdim,
                                  ag_op operation,
                                  reduction_forward_fn remove_dim,
                                  reduction_forward_fn keep_dim,
                                  ag_backward_fn backward) {
    if (value == NULL) return NULL;
    tensor* output = keepdim ? keep_dim(value->value, dim) : remove_dim(value->value, dim);
    if (output == NULL) return NULL;
    reduction_context* context = (reduction_context*)malloc(sizeof(*context));
    if (context == NULL) { t_free(output); return NULL; }
    context->dim = dim;
    context->keepdim = keepdim != 0;
    ag_tensor* inputs[1] = {(ag_tensor*)value};
    return ag_make_result(output, operation, 1, inputs, backward, context, free);
}

ag_tensor* ag_sum(const ag_tensor* value, int dim, int keepdim) {
    return apply_reduction(value, dim, keepdim, AG_OP_SUM,
                           t_sum, t_sum_keepdim, backward_sum);
}

ag_tensor* ag_mean(const ag_tensor* value, int dim, int keepdim) {
    return apply_reduction(value, dim, keepdim, AG_OP_MEAN,
                           t_mean, t_mean_keepdim, backward_mean);
}

ag_tensor* ag_max(const ag_tensor* value, int dim, int keepdim) {
    return apply_reduction(value, dim, keepdim, AG_OP_MAX,
                           t_max, t_max_keepdim, backward_max);
}
