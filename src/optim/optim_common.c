#include <math.h>

#include "../../include/tensorlib/nn.h"

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

static int module_valid(const nn_module* module)
{
    if (module == NULL ||
        module->parameter_count > module->parameter_capacity ||
        module->child_count > module->child_capacity ||
        (module->parameter_count > 0 && module->parameters == NULL) ||
        (module->child_count > 0 && module->children == NULL)) {
        return 0;
    }
    for (size_t i = 0; i < module->parameter_count; ++i) {
        if (module->parameters[i] == NULL) return 0;
    }
    for (size_t i = 0; i < module->child_count; ++i) {
        if (!module_valid(module->children[i])) return 0;
    }
    return 1;
}

void nn_module_zero_grad(nn_module* module)
{
    size_t count;

    if (!module_valid(module)) return;
    count = nn_module_parameter_count(module);
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        if (parameter != NULL && parameter->value != NULL) {
            ag_zero_grad(parameter->value);
        }
    }
}

int nn_clip_grad_norm(nn_module* module,
                      float max_norm,
                      float* total_norm)
{
    double squared_norm = 0.0;
    double norm;
    double scale;
    size_t count;

    if (!module_valid(module) || !isfinite(max_norm) || max_norm <= 0.0f) {
        return -1;
    }
    count = nn_module_parameter_count(module);
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        tensor* value;
        tensor* gradient;
        int element_count;

        if (parameter == NULL || parameter->value == NULL ||
            !tensor_has_valid_metadata(parameter->value->value)) {
            return -1;
        }
        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        value = parameter->value->value;
        gradient = parameter->value->grad;
        if (!tensor_has_valid_metadata(gradient) ||
            !same_shape(value, gradient)) {
            return -1;
        }
        element_count = tensor_numel(gradient);
        for (int element = 0; element < element_count; ++element) {
            double grad = gradient->storage->data[
                tensor_flat_index(gradient, element)];
            if (!isfinite(grad)) return -1;
            squared_norm += grad * grad;
        }
    }
    norm = sqrt(squared_norm);
    if (!isfinite(norm)) return -1;
    if (total_norm != NULL) *total_norm = (float)norm;
    if (norm <= (double)max_norm || norm == 0.0) return 0;

    scale = (double)max_norm / norm;
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        tensor* gradient;
        int element_count;

        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        gradient = parameter->value->grad;
        element_count = tensor_numel(gradient);
        for (int element = 0; element < element_count; ++element) {
            int index = tensor_flat_index(gradient, element);
            gradient->storage->data[index] =
                (float)((double)gradient->storage->data[index] * scale);
        }
        tensor_mark_modified(gradient);
    }
    return 0;
}
