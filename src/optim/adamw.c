#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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

static int config_valid(const nn_adamw_config* config)
{
    return config != NULL &&
           isfinite(config->learning_rate) && config->learning_rate > 0.0f &&
           isfinite(config->beta1) &&
           config->beta1 >= 0.0f && config->beta1 < 1.0f &&
           isfinite(config->beta2) &&
           config->beta2 >= 0.0f && config->beta2 < 1.0f &&
           isfinite(config->epsilon) && config->epsilon > 0.0f &&
           isfinite(config->weight_decay) && config->weight_decay >= 0.0f &&
           isfinite(config->max_grad_norm) && config->max_grad_norm >= 0.0f;
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

nn_adamw_config nn_adamw_default_config(void)
{
    nn_adamw_config config;

    config.learning_rate = 1e-3f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.epsilon = 1e-8f;
    config.weight_decay = 0.01f;
    config.max_grad_norm = 0.0f;
    return config;
}

static void free_state(nn_adamw* optimizer)
{
    if (optimizer == NULL) return;
    if (optimizer->first_moments != NULL) {
        for (size_t i = 0; i < optimizer->parameter_count; ++i) {
            t_free(optimizer->first_moments[i]);
        }
    }
    if (optimizer->second_moments != NULL) {
        for (size_t i = 0; i < optimizer->parameter_count; ++i) {
            t_free(optimizer->second_moments[i]);
        }
    }
    free(optimizer->steps);
    free(optimizer->second_moments);
    free(optimizer->first_moments);
    free(optimizer->parameters);
}

nn_adamw* nn_adamw_create(nn_module* module,
                           const nn_adamw_config* config)
{
    nn_adamw* optimizer;
    size_t count;

    if (!module_valid(module) || !config_valid(config)) return NULL;
    count = nn_module_parameter_count(module);
    if (count == SIZE_MAX) return NULL;
    optimizer = (nn_adamw*)calloc(1, sizeof(*optimizer));
    if (optimizer == NULL) return NULL;
    optimizer->module = module;
    optimizer->config = *config;
    optimizer->parameter_count = count;
    if (count == 0) return optimizer;

    optimizer->parameters =
        (nn_parameter**)calloc(count, sizeof(*optimizer->parameters));
    optimizer->first_moments =
        (tensor**)calloc(count, sizeof(*optimizer->first_moments));
    optimizer->second_moments =
        (tensor**)calloc(count, sizeof(*optimizer->second_moments));
    optimizer->steps =
        (uint64_t*)calloc(count, sizeof(*optimizer->steps));
    if (optimizer->parameters == NULL ||
        optimizer->first_moments == NULL ||
        optimizer->second_moments == NULL ||
        optimizer->steps == NULL) {
        free_state(optimizer);
        free(optimizer);
        return NULL;
    }
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        tensor* value;

        if (parameter == NULL || parameter->value == NULL ||
            !tensor_has_valid_metadata(parameter->value->value)) {
            free_state(optimizer);
            free(optimizer);
            return NULL;
        }
        optimizer->parameters[i] = parameter;
        value = parameter->value->value;
        optimizer->first_moments[i] = t_alloc(value->ndim, value->dims);
        optimizer->second_moments[i] = t_alloc(value->ndim, value->dims);
        if (optimizer->first_moments[i] == NULL ||
            optimizer->second_moments[i] == NULL) {
            free_state(optimizer);
            free(optimizer);
            return NULL;
        }
        for (int element = 0; element < tensor_numel(value); ++element) {
            optimizer->first_moments[i]->storage->data[element] = 0.0f;
            optimizer->second_moments[i]->storage->data[element] = 0.0f;
        }
    }
    return optimizer;
}

static int topology_valid(const nn_adamw* optimizer)
{
    if (optimizer == NULL || !module_valid(optimizer->module) ||
        nn_module_parameter_count(optimizer->module) !=
            optimizer->parameter_count ||
        !config_valid(&optimizer->config) ||
        (optimizer->parameter_count > 0 &&
         (optimizer->parameters == NULL ||
          optimizer->first_moments == NULL ||
          optimizer->second_moments == NULL ||
          optimizer->steps == NULL))) {
        return 0;
    }
    for (size_t i = 0; i < optimizer->parameter_count; ++i) {
        if (nn_module_parameter_at(optimizer->module, i) !=
            optimizer->parameters[i]) {
            return 0;
        }
    }
    return 1;
}

static int gradient_scale(const nn_adamw* optimizer, double* result)
{
    double squared_norm = 0.0;

    for (size_t i = 0; i < optimizer->parameter_count; ++i) {
        nn_parameter* parameter = optimizer->parameters[i];
        tensor* value;
        tensor* gradient;

        if (parameter == NULL || parameter->value == NULL ||
            !tensor_has_valid_metadata(parameter->value->value) ||
            !tensor_has_valid_metadata(optimizer->first_moments[i]) ||
            !tensor_has_valid_metadata(optimizer->second_moments[i])) {
            return -1;
        }
        value = parameter->value->value;
        if (!same_shape(value, optimizer->first_moments[i]) ||
            !same_shape(value, optimizer->second_moments[i])) {
            return -1;
        }
        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        gradient = parameter->value->grad;
        if (!tensor_has_valid_metadata(gradient) ||
            !same_shape(value, gradient) ||
            optimizer->steps[i] == UINT64_MAX) {
            return -1;
        }
        for (int element = 0; element < tensor_numel(gradient); ++element) {
            double grad = gradient->storage->data[
                tensor_flat_index(gradient, element)];
            if (!isfinite(grad)) return -1;
            squared_norm += grad * grad;
        }
    }
    if (!isfinite(squared_norm)) return -1;
    *result = 1.0;
    if (optimizer->config.max_grad_norm > 0.0f && squared_norm > 0.0) {
        double norm = sqrt(squared_norm);
        if (!isfinite(norm)) return -1;
        if (norm > optimizer->config.max_grad_norm) {
            *result = optimizer->config.max_grad_norm / norm;
        }
    }
    return 0;
}

static int proposed_updates_valid(const nn_adamw* optimizer,
                                  double grad_scale)
{
    const nn_adamw_config* config = &optimizer->config;

    for (size_t i = 0; i < optimizer->parameter_count; ++i) {
        nn_parameter* parameter = optimizer->parameters[i];
        tensor* value;
        tensor* gradient;
        uint64_t step;
        double correction1;
        double correction2;

        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        value = parameter->value->value;
        gradient = parameter->value->grad;
        step = optimizer->steps[i] + 1;
        correction1 = 1.0 - pow((double)config->beta1, (double)step);
        correction2 = 1.0 - pow((double)config->beta2, (double)step);
        if (!(correction1 > 0.0) || !(correction2 > 0.0)) return 0;

        for (int element = 0; element < tensor_numel(value); ++element) {
            int value_index = tensor_flat_index(value, element);
            int grad_index = tensor_flat_index(gradient, element);
            int first_index =
                tensor_flat_index(optimizer->first_moments[i], element);
            int second_index =
                tensor_flat_index(optimizer->second_moments[i], element);
            double current = value->storage->data[value_index];
            double grad = gradient->storage->data[grad_index] * grad_scale;
            double first =
                config->beta1 *
                    optimizer->first_moments[i]->storage->data[first_index] +
                (1.0 - config->beta1) * grad;
            double second =
                config->beta2 *
                    optimizer->second_moments[i]->storage->data[second_index] +
                (1.0 - config->beta2) * grad * grad;
            double normalized =
                (first / correction1) /
                (sqrt(second / correction2) + config->epsilon);
            double updated =
                current - config->learning_rate *
                    (normalized + config->weight_decay * current);
            if (!isfinite(current) || !isfinite(first) || !isfinite(second) ||
                !isfinite(updated)) {
                return 0;
            }
        }
    }
    return 1;
}

int nn_adamw_step(nn_adamw* optimizer)
{
    const nn_adamw_config* config;
    double grad_scale;

    if (!topology_valid(optimizer) ||
        gradient_scale(optimizer, &grad_scale) != 0 ||
        !proposed_updates_valid(optimizer, grad_scale)) {
        return -1;
    }
    config = &optimizer->config;
    for (size_t i = 0; i < optimizer->parameter_count; ++i) {
        nn_parameter* parameter = optimizer->parameters[i];
        tensor* value;
        tensor* gradient;
        uint64_t step;
        double correction1;
        double correction2;

        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        value = parameter->value->value;
        gradient = parameter->value->grad;
        step = ++optimizer->steps[i];
        correction1 = 1.0 - pow((double)config->beta1, (double)step);
        correction2 = 1.0 - pow((double)config->beta2, (double)step);
        for (int element = 0; element < tensor_numel(value); ++element) {
            int value_index = tensor_flat_index(value, element);
            int grad_index = tensor_flat_index(gradient, element);
            int first_index =
                tensor_flat_index(optimizer->first_moments[i], element);
            int second_index =
                tensor_flat_index(optimizer->second_moments[i], element);
            double current = value->storage->data[value_index];
            double grad = gradient->storage->data[grad_index] * grad_scale;
            double first =
                config->beta1 *
                    optimizer->first_moments[i]->storage->data[first_index] +
                (1.0 - config->beta1) * grad;
            double second =
                config->beta2 *
                    optimizer->second_moments[i]->storage->data[second_index] +
                (1.0 - config->beta2) * grad * grad;
            double normalized =
                (first / correction1) /
                (sqrt(second / correction2) + config->epsilon);

            optimizer->first_moments[i]->storage->data[first_index] =
                (float)first;
            optimizer->second_moments[i]->storage->data[second_index] =
                (float)second;
            value->storage->data[value_index] =
                (float)(current - config->learning_rate *
                    (normalized + config->weight_decay * current));
        }
        tensor_mark_modified(optimizer->first_moments[i]);
        tensor_mark_modified(optimizer->second_moments[i]);
        tensor_mark_modified(value);
    }
    return 0;
}

void nn_adamw_zero_grad(nn_adamw* optimizer)
{
    if (!topology_valid(optimizer)) return;
    nn_module_zero_grad(optimizer->module);
}

void nn_adamw_destroy(nn_adamw* optimizer)
{
    if (optimizer == NULL) return;
    free_state(optimizer);
    free(optimizer);
}
