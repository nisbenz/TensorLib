#include <math.h>
#include <stdlib.h>

#include "../../include/tensorlib/nn.h"
#include "../nn/nn_internal.h"
#include "../tensor/tensor_alloc_internal.h"

nn_sgd* nn_sgd_create(nn_module* module, float learning_rate)
{
    nn_sgd* optimizer;

    if (module == NULL || !isfinite(learning_rate) || learning_rate <= 0.0f) {
        return NULL;
    }
    optimizer = (nn_sgd*)malloc(sizeof(*optimizer));
    if (optimizer == NULL) return NULL;
    optimizer->module = module;
    optimizer->learning_rate = learning_rate;
    return optimizer;
}

static int nn_sgd_parameter_valid(const nn_sgd* optimizer,
                                  const nn_parameter* parameter)
{
    tensor* value;
    tensor* gradient;
    int count;

    if (parameter == NULL || parameter->value == NULL ||
        !tensor_has_valid_metadata(parameter->value->value)) {
        return 0;
    }
    if (!parameter->trainable || parameter->value->grad == NULL) return 1;
    value = parameter->value->value;
    gradient = parameter->value->grad;
    if (!tensor_has_valid_metadata(gradient) ||
        !same_shape(value, gradient)) {
        return 0;
    }
    count = tensor_numel(value);
    for (int i = 0; i < count; ++i) {
        float current = value->storage->data[tensor_flat_index(value, i)];
        float grad = gradient->storage->data[tensor_flat_index(gradient, i)];
        float updated;
        if (!isfinite(current) || !isfinite(grad)) return 0;
        updated = current - optimizer->learning_rate * grad;
        if (!isfinite(updated)) return 0;
    }
    return 1;
}

int nn_sgd_step(nn_sgd* optimizer)
{
    size_t count;

    if (optimizer == NULL || !nn_module_is_valid(optimizer->module) ||
        !isfinite(optimizer->learning_rate) ||
        optimizer->learning_rate <= 0.0f) {
        return -1;
    }
    count = nn_module_parameter_count(optimizer->module);
    for (size_t i = 0; i < count; ++i) {
        if (!nn_sgd_parameter_valid(
                optimizer, nn_module_parameter_at(optimizer->module, i))) {
            return -1;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(optimizer->module, i);
        tensor* value;
        tensor* gradient;
        int element_count;

        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        value = parameter->value->value;
        gradient = parameter->value->grad;
        element_count = tensor_numel(value);
        for (int element = 0; element < element_count; ++element) {
            int value_index = tensor_flat_index(value, element);
            int gradient_index = tensor_flat_index(gradient, element);
            value->storage->data[value_index] -=
                optimizer->learning_rate * gradient->storage->data[gradient_index];
        }
        tensor_mark_modified(value);
    }
    return 0;
}

void nn_sgd_zero_grad(nn_sgd* optimizer)
{
    if (optimizer == NULL || !nn_module_is_valid(optimizer->module)) return;
    nn_module_zero_grad(optimizer->module);
}

void nn_sgd_destroy(nn_sgd* optimizer)
{
    free(optimizer);
}
