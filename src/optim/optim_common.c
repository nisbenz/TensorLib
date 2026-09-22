#include <math.h>

#include "../../include/tensorlib/nn.h"
#include "../nn/nn_internal.h"
#include "../tensor/tensor_internal.h"

void nn_module_zero_grad(nn_module* module)
{
    size_t count;

    if (!nn_module_is_valid(module)) return;
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

    if (!nn_module_is_valid(module) || !isfinite(max_norm) || max_norm <= 0.0f) {
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
