#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "../../include/tensorlib/nn.h"
#include "../nn/nn_internal.h"
#include "../tensor/parallel.h"
#include "../tensor/tensor_internal.h"

#define TENSORLIB_ADAMW_MIN_PARALLEL_ELEMENTS (1 << 16)

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

    if (!nn_module_is_valid(module) || !nn_adamw_config_is_valid(config)) return NULL;
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
        int element_count = tensor_numel(value);
        for (int element = 0; element < element_count; ++element) {
            optimizer->first_moments[i]->storage->data[element] = 0.0f;
            optimizer->second_moments[i]->storage->data[element] = 0.0f;
        }
    }
    return optimizer;
}

static int topology_valid(const nn_adamw* optimizer)
{
    if (optimizer == NULL || !nn_module_is_valid(optimizer->module) ||
        nn_module_parameter_count(optimizer->module) !=
            optimizer->parameter_count ||
        !nn_adamw_config_is_valid(&optimizer->config) ||
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
    long long work = 0;
    int invalid = 0;

    /* Validate all state before entering the parallel reduction so failure
     * remains transactional. */
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
        int count = tensor_numel(gradient);
        work = work > LLONG_MAX - count ? LLONG_MAX : work + count;
    }

    int threads = tensorlib_parallel_threads(
        work, TENSORLIB_ADAMW_MIN_PARALLEL_ELEMENTS,
        optimizer->parameter_count > (size_t)INT_MAX
            ? 1 : (int)optimizer->parameter_count);
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for if(threads > 1) schedule(dynamic, 1) \
    num_threads(threads) reduction(+:squared_norm) reduction(|:invalid)
#endif
    for (long long index = 0;
         index < (long long)optimizer->parameter_count; ++index) {
        nn_parameter* parameter = optimizer->parameters[index];
        tensor* gradient;
        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        gradient = parameter->value->grad;
        int count = tensor_numel(gradient);
        if (is_contiguous(gradient) && gradient->offset == 0) {
            const float* data = gradient->storage->data;
            for (int element = 0; element < count; ++element) {
                double grad = data[element];
                if (!isfinite(grad)) invalid = 1;
                squared_norm += grad * grad;
            }
        } else {
            for (int element = 0; element < count; ++element) {
                double grad = gradient->storage->data[
                    tensor_flat_index(gradient, element)];
                if (!isfinite(grad)) invalid = 1;
                squared_norm += grad * grad;
            }
        }
    }
    if (invalid || !isfinite(squared_norm)) return -1;
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

int nn_adamw_step(nn_adamw* optimizer)
{
    const nn_adamw_config* config;
    double grad_scale;
    uint64_t* next_steps = NULL;
    double* correction1s = NULL;
    double* correction2s = NULL;
    long long work = 0;
    int parameter_count;
    int eligible_count = 0;
    int threads;
    int validation_failed = 0;
    int update_failed = 0;

    if (!topology_valid(optimizer) ||
        gradient_scale(optimizer, &grad_scale) != 0) {
        return -1;
    }
    config = &optimizer->config;
    if (optimizer->parameter_count > (size_t)INT_MAX) return -1;
    parameter_count = (int)optimizer->parameter_count;
    /* First determine the amount of independent work without allocating
     * parallel-only bookkeeping for the normal small serial path. */
    for (int i = 0; i < parameter_count; ++i) {
        nn_parameter* parameter = optimizer->parameters[i];
        tensor* value;

        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        value = parameter->value->value;
        ++eligible_count;
        int count = tensor_numel(value);
        work = work > LLONG_MAX - count ? LLONG_MAX : work + count;
    }
    threads = tensorlib_parallel_threads(work,
                                         TENSORLIB_ADAMW_MIN_PARALLEL_ELEMENTS,
                                         eligible_count);
    if (threads > 1) {
        next_steps = (uint64_t*)calloc((size_t)parameter_count,
                                       sizeof(*next_steps));
        correction1s = (double*)calloc((size_t)parameter_count,
                                       sizeof(*correction1s));
        correction2s = (double*)calloc((size_t)parameter_count,
                                       sizeof(*correction2s));
        if (next_steps == NULL || correction1s == NULL || correction2s == NULL) {
            free(correction2s); free(correction1s); free(next_steps);
            return -1;
        }
        /* Validate every value and precompute all step-dependent factors before
         * any parameter or optimizer state is modified. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) \
    reduction(|:validation_failed)
#endif
        for (int i = 0; i < parameter_count; ++i) {
            nn_parameter* parameter = optimizer->parameters[i];
            tensor* value;
            uint64_t step;
            if (!parameter->trainable || parameter->value->grad == NULL) continue;
            value = parameter->value->value;
            step = optimizer->steps[i] + 1;
            correction1s[i] = 1.0 - pow((double)config->beta1, (double)step);
            correction2s[i] = 1.0 - pow((double)config->beta2, (double)step);
            if (!(correction1s[i] > 0.0) || !(correction2s[i] > 0.0) ||
                optimizer->steps[i] == UINT64_MAX) {
                validation_failed = 1;
                continue;
            }
            next_steps[i] = step;
            int count = tensor_numel(value);
            tensor* first_moment = optimizer->first_moments[i];
            tensor* second_moment = optimizer->second_moments[i];
            if (is_contiguous(value) && is_contiguous(first_moment) &&
                is_contiguous(second_moment)) {
                const float* value_data = value->storage->data + value->offset;
                const float* first_data = first_moment->storage->data +
                                          first_moment->offset;
                const float* second_data = second_moment->storage->data +
                                           second_moment->offset;
                for (int element = 0; element < count; ++element) {
                    if (!isfinite(value_data[element]) ||
                        !isfinite(first_data[element]) ||
                        !isfinite(second_data[element])) {
                        validation_failed = 1;
                        break;
                    }
                }
            } else {
                for (int element = 0; element < count; ++element) {
                    int value_index = tensor_flat_index(value, element);
                    int first_index = tensor_flat_index(first_moment, element);
                    int second_index = tensor_flat_index(second_moment, element);
                    if (isfinite(value->storage->data[value_index]) &&
                        isfinite(first_moment->storage->data[first_index]) &&
                        isfinite(second_moment->storage->data[second_index])) {
                        continue;
                    }
                    validation_failed = 1;
                    break;
                }
            }
        }
        if (validation_failed) {
            free(correction2s); free(correction1s); free(next_steps);
            return -1;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for if(threads > 1) schedule(dynamic, 1) num_threads(threads)
#endif
    for (int i = 0; i < parameter_count; ++i) {
        nn_parameter* parameter = optimizer->parameters[i];
        tensor* value;
        tensor* gradient;
        double correction1;
        double correction2;

        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        value = parameter->value->value;
        gradient = parameter->value->grad;
        uint64_t step = optimizer->steps[i] + 1;
        correction1 = threads > 1 ? correction1s[i]
                                   : 1.0 - pow((double)config->beta1, (double)step);
        correction2 = threads > 1 ? correction2s[i]
                                   : 1.0 - pow((double)config->beta2, (double)step);
        if (!(correction1 > 0.0) || !(correction2 > 0.0) ||
            optimizer->steps[i] == UINT64_MAX) {
            update_failed = 1;
            continue;
        }
        optimizer->steps[i] = step;
        /* Fast path: when value/gradient/first/second-moment share a
         * contiguous layout, the flat element index is the storage index,
         * avoiding the per-element tensor_flat_index ndim decomposition.
         * The buffer is also touched exactly once (single pass, streamed). */
        if (is_contiguous(value) && is_contiguous(gradient) &&
            is_contiguous(optimizer->first_moments[i]) &&
            is_contiguous(optimizer->second_moments[i]) &&
            value->offset == 0 && gradient->offset == 0 &&
            optimizer->first_moments[i]->offset == 0 &&
            optimizer->second_moments[i]->offset == 0) {
            int count = tensor_numel(value);
            float* value_data = value->storage->data;
            float* grad_data = gradient->storage->data;
            float* first_data = optimizer->first_moments[i]->storage->data;
            float* second_data = optimizer->second_moments[i]->storage->data;
            for (int element = 0; element < count; ++element) {
                double current = value_data[element];
                double grad = grad_data[element] * grad_scale;
                double first =
                    config->beta1 * first_data[element] +
                    (1.0 - config->beta1) * grad;
                double second =
                    config->beta2 * second_data[element] +
                    (1.0 - config->beta2) * grad * grad;
                double normalized =
                    (first / correction1) /
                    (sqrt(second / correction2) + config->epsilon);
                double updated =
                    current - config->learning_rate *
                        (normalized + config->weight_decay * current);
                if (!isfinite(current) || !isfinite(first) || !isfinite(second) ||
                    !isfinite(updated)) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                    update_failed = 1;
                    continue;
                }
                first_data[element] = (float)first;
                second_data[element] = (float)second;
                value_data[element] = (float)updated;
            }
        } else {
        int element_count = tensor_numel(value);
        for (int element = 0; element < element_count; ++element) {
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
#ifdef _OPENMP
#pragma omp atomic write
#endif
                update_failed = 1;
                continue;
            }
            optimizer->first_moments[i]->storage->data[first_index] =
                (float)first;
            optimizer->second_moments[i]->storage->data[second_index] =
                (float)second;
            value->storage->data[value_index] = (float)updated;
        }
        }
    }
    if (update_failed) {
        free(correction2s); free(correction1s); free(next_steps);
        return -1;
    }
    for (int i = 0; i < parameter_count; ++i) {
        nn_parameter* parameter = optimizer->parameters[i];
        if (!parameter->trainable || parameter->value->grad == NULL) continue;
        tensor_mark_modified(optimizer->first_moments[i]);
        tensor_mark_modified(optimizer->second_moments[i]);
        tensor_mark_modified(parameter->value->value);
    }
    free(correction2s);
    free(correction1s);
    free(next_steps);
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
