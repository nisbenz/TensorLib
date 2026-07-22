#include <stdlib.h>

#include "../../include/tensorlib/autograd.h"

ag_tensor* ag_from_owned_tensor(tensor* value, int requires_grad) {
    if (!tensor_has_valid_metadata(value)) {
        t_free(value);
        return NULL;
    }

    ag_tensor* result = (ag_tensor*)calloc(1, sizeof(*result));
    if (result == NULL) {
        t_free(value);
        return NULL;
    }

    result->value = value;
    result->requires_grad = requires_grad != 0;
    result->ref_count = 1;
    return result;
}

void ag_tensor_retain(ag_tensor* value) {
    if (value == NULL || value->ref_count <= 0) return;
    value->ref_count++;
}

void ag_tensor_release(ag_tensor* value) {
    if (value == NULL || value->ref_count <= 0) return;
    if (--value->ref_count != 0) return;

    ag_node_release(value->creator);
    t_free(value->grad);
    t_free(value->value);
    free(value);
}

void ag_node_retain(ag_node* node) {
    if (node == NULL || node->ref_count <= 0) return;
    node->ref_count++;
}

void ag_node_release(ag_node* node) {
    if (node == NULL || node->ref_count <= 0) return;
    if (--node->ref_count != 0) return;

    if (node->free_context != NULL) node->free_context(node->context);
    for (int i = 0; i < node->input_count; ++i) {
        ag_tensor_release(node->inputs[i]);
    }
    free(node->inputs);
    free(node);
}
