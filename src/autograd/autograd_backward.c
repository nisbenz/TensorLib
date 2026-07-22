#include <stdlib.h>

#include "./../../include/tensorlib/autograd_internal.h"

typedef struct {
    ag_tensor** values;
    int count;
    int capacity;
} tensor_list;

typedef struct {
    ag_node** values;
    int count;
    int capacity;
} node_list;

static int tensor_index(const tensor_list* list, const ag_tensor* value) {
    for (int i = 0; i < list->count; ++i) {
        if (list->values[i] == value) return i;
    }
    return -1;
}

static int append_tensor(tensor_list* list, ag_tensor* value) {
    if (list->count == list->capacity) {
        int capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        ag_tensor** values = (ag_tensor**)realloc(list->values,
                                                  (size_t)capacity * sizeof(*values));
        if (values == NULL) return 1;
        list->values = values;
        list->capacity = capacity;
    }
    list->values[list->count++] = value;
    return 0;
}

static int append_node(node_list* list, ag_node* node) {
    if (list->count == list->capacity) {
        int capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        ag_node** values = (ag_node**)realloc(list->values,
                                              (size_t)capacity * sizeof(*values));
        if (values == NULL) return 1;
        list->values = values;
        list->capacity = capacity;
    }
    list->values[list->count++] = node;
    return 0;
}

static int collect_graph(ag_tensor* value, tensor_list* tensors, node_list* nodes) {
    if (value == NULL || tensor_index(tensors, value) >= 0) return value == NULL;
    if (append_tensor(tensors, value) != 0) return 1;
    if (value->creator == NULL) return 0;
    for (int i = 0; i < value->creator->input_count; ++i) {
        if (collect_graph(value->creator->inputs[i], tensors, nodes) != 0) return 1;
    }
    return append_node(nodes, value->creator);
}

static tensor* reduce_to_shape(tensor* contribution, const tensor* target) {
    if (!tensor_has_valid_metadata(contribution) || !tensor_has_valid_metadata(target) ||
        contribution->ndim < target->ndim) {
        t_free(contribution);
        return NULL;
    }

    tensor* current = contribution;
    while (current->ndim > target->ndim) {
        tensor* reduced = t_sum(current, 0);
        t_free(current);
        if (reduced == NULL) return NULL;
        current = reduced;
    }

    for (int axis = 0; axis < target->ndim; ++axis) {
        if (current->dims[axis] == target->dims[axis]) continue;
        if (target->dims[axis] != 1 || current->dims[axis] == 1) {
            t_free(current);
            return NULL;
        }
        tensor* reduced = t_sum_keepdim(current, axis);
        t_free(current);
        if (reduced == NULL) return NULL;
        current = reduced;
    }
    return current;
}

static int accumulate_pass_gradient(tensor** destination,
                                    tensor* contribution,
                                    const tensor* target) {
    tensor* reduced = reduce_to_shape(contribution, target);
    if (reduced == NULL) return 1;
    if (*destination == NULL) {
        *destination = reduced;
        return 0;
    }
    tensor* sum = t_add(*destination, reduced);
    t_free(reduced);
    if (sum == NULL) return 1;
    t_free(*destination);
    *destination = sum;
    return 0;
}

static void free_tensor_array(tensor** values, int count) {
    if (values == NULL) return;
    for (int i = 0; i < count; ++i) t_free(values[i]);
    free(values);
}

static int merge_persistent_gradients(const tensor_list* tensors, tensor** pass_gradients) {
    tensor** merged = (tensor**)calloc((size_t)tensors->count, sizeof(*merged));
    if (merged == NULL) return 1;

    for (int i = 0; i < tensors->count; ++i) {
        ag_tensor* value = tensors->values[i];
        if (!value->requires_grad || pass_gradients[i] == NULL) continue;
        merged[i] = value->grad == NULL
                  ? t_clone(pass_gradients[i])
                  : t_add(value->grad, pass_gradients[i]);
        if (merged[i] == NULL) {
            free_tensor_array(merged, tensors->count);
            return 1;
        }
    }

    for (int i = 0; i < tensors->count; ++i) {
        if (merged[i] == NULL) continue;
        t_free(tensors->values[i]->grad);
        tensors->values[i]->grad = merged[i];
        merged[i] = NULL;
    }
    free(merged);
    return 0;
}

int ag_backward_with_grad(ag_tensor* output, const tensor* output_gradient) {
    if (output == NULL || !output->requires_grad ||
        !tensor_has_valid_metadata(output->value) ||
        !tensor_has_valid_metadata(output_gradient) ||
        !same_shape(output->value, (tensor*)output_gradient)) return 1;

    tensor_list tensors = {0};
    node_list nodes = {0};
    tensor** pass_gradients = NULL;
    int status = 1;
    if (collect_graph(output, &tensors, &nodes) != 0) goto cleanup;

    pass_gradients = (tensor**)calloc((size_t)tensors.count, sizeof(*pass_gradients));
    if (pass_gradients == NULL) goto cleanup;
    int output_index = tensor_index(&tensors, output);
    pass_gradients[output_index] = t_clone((tensor*)output_gradient);
    if (pass_gradients[output_index] == NULL) goto cleanup;

    for (int node_index = nodes.count - 1; node_index >= 0; --node_index) {
        ag_node* node = nodes.values[node_index];
        int gradient_index = tensor_index(&tensors, node->output);
        if (gradient_index < 0 || pass_gradients[gradient_index] == NULL) goto cleanup;
        tensor** contributions = (tensor**)calloc((size_t)node->input_count,
                                                   sizeof(*contributions));
        if (contributions == NULL) goto cleanup;
        if (node->backward(node, pass_gradients[gradient_index], contributions) != 0) {
            free_tensor_array(contributions, node->input_count);
            goto cleanup;
        }

        for (int input_index = 0; input_index < node->input_count; ++input_index) {
            ag_tensor* input = node->inputs[input_index];
            if (!input->requires_grad) {
                t_free(contributions[input_index]);
                contributions[input_index] = NULL;
                continue;
            }
            int destination = tensor_index(&tensors, input);
            if (destination < 0 || contributions[input_index] == NULL ||
                accumulate_pass_gradient(&pass_gradients[destination],
                                         contributions[input_index], input->value) != 0) {
                contributions[input_index] = NULL;
                free_tensor_array(contributions, node->input_count);
                goto cleanup;
            }
            contributions[input_index] = NULL;
        }
        free(contributions);
    }

    status = merge_persistent_gradients(&tensors, pass_gradients);

cleanup:
    free_tensor_array(pass_gradients, tensors.count);
    free(nodes.values);
    free(tensors.values);
    return status;
}

int ag_backward(ag_tensor* loss) {
    if (loss == NULL || loss->value == NULL || loss->value->ndim != 0) return 1;
    tensor* seed = ag_full_like(loss->value, 1.0f);
    if (seed == NULL) return 1;
    int status = ag_backward_with_grad(loss, seed);
    t_free(seed);
    return status;
}

void ag_zero_grad(ag_tensor* value) {
    if (value == NULL) return;
    t_free(value->grad);
    value->grad = NULL;
}

void ag_zero_grad_all(ag_tensor* root) {
    if (root == NULL) return;
    tensor_list tensors = {0};
    node_list nodes = {0};
    if (collect_graph(root, &tensors, &nodes) == 0) {
        for (int i = 0; i < tensors.count; ++i) ag_zero_grad(tensors.values[i]);
    }
    free(nodes.values);
    free(tensors.values);
}
