#include <stdlib.h>

#include "../../include/tensorlib/autograd.h"
#include "./../../include/tensorlib/autograd_internal.h"

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
    free(node->input_versions);
    free(node->inputs);
    free(node);
}

ag_tensor* ag_make_result(tensor* output,
                          ag_op operation,
                          int input_count,
                          ag_tensor* const* inputs,
                          ag_backward_fn backward,
                          void* context,
                          void (*free_context)(void*)) {
    int requires_grad = 0;
    if (output == NULL || input_count <= 0 || inputs == NULL || backward == NULL) {
        t_free(output);
        if (free_context != NULL) free_context(context);
        return NULL;
    }
    for (int i = 0; i < input_count; ++i) {
        if (inputs[i] == NULL || !tensor_has_valid_metadata(inputs[i]->value)) {
            t_free(output);
            if (free_context != NULL) free_context(context);
            return NULL;
        }
        requires_grad |= inputs[i]->requires_grad;
    }

    ag_tensor* result = ag_from_owned_tensor(output, requires_grad);
    if (result == NULL) {
        if (free_context != NULL) free_context(context);
        return NULL;
    }
    if (!requires_grad) {
        if (free_context != NULL) free_context(context);
        return result;
    }

    ag_node* node = (ag_node*)calloc(1, sizeof(*node));
    if (node == NULL) {
        if (free_context != NULL) free_context(context);
        ag_tensor_release(result);
        return NULL;
    }
    node->inputs = (ag_tensor**)calloc((size_t)input_count, sizeof(*node->inputs));
    if (node->inputs == NULL) {
        free(node);
        if (free_context != NULL) free_context(context);
        ag_tensor_release(result);
        return NULL;
    }
    node->input_versions = (uint64_t*)calloc((size_t)input_count,
                                             sizeof(*node->input_versions));
    if (node->input_versions == NULL) {
        free(node->inputs);
        free(node);
        if (free_context != NULL) free_context(context);
        ag_tensor_release(result);
        return NULL;
    }

    node->operation = operation;
    node->input_count = input_count;
    node->output = result;
    node->backward = backward;
    node->context = context;
    node->free_context = free_context;
    node->ref_count = 1;
    for (int i = 0; i < input_count; ++i) {
        node->inputs[i] = inputs[i];
        node->input_versions[i] = inputs[i]->value->storage->version;
        ag_tensor_retain(inputs[i]);
    }
    node->output_version = output->storage->version;
    result->creator = node;
    return result;
}

tensor* ag_full_like(const tensor* reference, float value) {
    if (!tensor_has_valid_metadata(reference)) return NULL;
    tensor* result = t_alloc(reference->ndim, reference->dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) {
        result->storage->data[i] = value;
    }
    return result;
}
