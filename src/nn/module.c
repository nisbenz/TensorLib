#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "nn_internal.h"

static char* nn_module_copy_string(const char* value)
{
    size_t length;
    char* copy;

    if (value == NULL || value[0] == '\0') return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1);
    if (copy != NULL) memcpy(copy, value, length + 1);
    return copy;
}

static int nn_module_contains(const nn_module* root, const nn_module* target)
{
    if (root == NULL) return 0;
    if (root == target) return 1;
    for (size_t i = 0; i < root->child_count; ++i) {
        if (nn_module_contains(root->children[i], target)) return 1;
    }
    return 0;
}

int nn_module_is_valid(const nn_module* module)
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
        if (!nn_module_is_valid(module->children[i])) return 0;
    }
    return 1;
}

int nn_adamw_config_is_valid(const nn_adamw_config* config)
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

char* nn_qualified_name(const char* module_name, const char* suffix)
{
    size_t module_length;
    size_t suffix_length;
    char* result;

    if (module_name == NULL || suffix == NULL) return NULL;
    module_length = strlen(module_name);
    suffix_length = strlen(suffix);
    if (module_length > SIZE_MAX - suffix_length - 2) return NULL;
    result = (char*)malloc(module_length + suffix_length + 2);
    if (result == NULL) return NULL;
    memcpy(result, module_name, module_length);
    result[module_length] = '.';
    memcpy(result + module_length + 1, suffix, suffix_length + 1);
    return result;
}

char* nn_indexed_name(const char* module_name,
                      const char* collection,
                      size_t index)
{
    int required;
    char* result;

    if (module_name == NULL || collection == NULL) return NULL;
    required = snprintf(NULL, 0, "%s.%s.%zu",
                        module_name, collection, index);
    if (required < 0 || (size_t)required == SIZE_MAX) return NULL;
    result = (char*)malloc((size_t)required + 1);
    if (result != NULL) {
        snprintf(result, (size_t)required + 1, "%s.%s.%zu",
                 module_name, collection, index);
    }
    return result;
}

nn_parameter* nn_create_named_parameter(const char* module_name,
                                         const char* suffix,
                                         int ndim,
                                         const int* dims,
                                         nn_init_kind initializer,
                                         nn_rng* rng)
{
    char* name = nn_qualified_name(module_name, suffix);
    nn_parameter* result;

    if (name == NULL) return NULL;
    result = nn_parameter_create(name, ndim, dims, 1, initializer, rng);
    free(name);
    return result;
}

static int nn_reserve_parameters(nn_module* module)
{
    size_t capacity;
    nn_parameter** parameters;

    if (module->parameter_count < module->parameter_capacity) return 0;
    capacity = module->parameter_capacity == 0 ? 4 : module->parameter_capacity * 2;
    if (capacity < module->parameter_capacity ||
        capacity > SIZE_MAX / sizeof(*parameters)) {
        return -1;
    }
    parameters = (nn_parameter**)realloc(
        module->parameters, capacity * sizeof(*parameters));
    if (parameters == NULL) return -1;
    module->parameters = parameters;
    module->parameter_capacity = capacity;
    return 0;
}

static int nn_reserve_children(nn_module* module)
{
    size_t capacity;
    nn_module** children;

    if (module->child_count < module->child_capacity) return 0;
    capacity = module->child_capacity == 0 ? 4 : module->child_capacity * 2;
    if (capacity < module->child_capacity ||
        capacity > SIZE_MAX / sizeof(*children)) {
        return -1;
    }
    children = (nn_module**)realloc(
        module->children, capacity * sizeof(*children));
    if (children == NULL) return -1;
    module->children = children;
    module->child_capacity = capacity;
    return 0;
}

static ag_tensor* nn_relu_forward(const nn_activation* activation,
                                  const ag_tensor* input)
{
    (void)activation;
    return ag_relu(input);
}

static ag_tensor* nn_gelu_forward(const nn_activation* activation,
                                  const ag_tensor* input)
{
    (void)activation;
    return ag_gelu(input);
}

static ag_tensor* nn_sigmoid_forward(const nn_activation* activation,
                                     const ag_tensor* input)
{
    (void)activation;
    return ag_sigmoid(input);
}

static ag_tensor* nn_tanh_forward(const nn_activation* activation,
                                  const ag_tensor* input)
{
    (void)activation;
    return ag_tanh(input);
}

nn_activation nn_activation_relu(void)
{
    nn_activation activation = {"ReLU", nn_relu_forward, NULL};
    return activation;
}

nn_activation nn_activation_gelu(void)
{
    nn_activation activation = {"GELU", nn_gelu_forward, NULL};
    return activation;
}

nn_activation nn_activation_sigmoid(void)
{
    nn_activation activation = {"Sigmoid", nn_sigmoid_forward, NULL};
    return activation;
}

nn_activation nn_activation_tanh(void)
{
    nn_activation activation = {"Tanh", nn_tanh_forward, NULL};
    return activation;
}

nn_activation nn_activation_custom(const char* name,
                                    nn_activation_forward_fn forward,
                                    const void* context)
{
    nn_activation activation = {name, forward, context};
    return activation;
}

int nn_module_init_base(nn_module* module,
                        const char* type_name,
                        const char* name,
                        nn_module_forward_fn forward,
                        nn_module_destroy_fn destroy)
{
    if (module == NULL || type_name == NULL || type_name[0] == '\0' ||
        name == NULL || name[0] == '\0' || forward == NULL || destroy == NULL) {
        return -1;
    }
    memset(module, 0, sizeof(*module));
    module->name = nn_module_copy_string(name);
    if (module->name == NULL) return -1;
    module->type_name = type_name;
    module->forward = forward;
    module->destroy = destroy;
    module->training = 1;
    return 0;
}

void nn_module_destroy_base(nn_module* module)
{
    if (module == NULL) return;
    for (size_t i = 0; i < module->parameter_count; ++i) {
        nn_parameter_destroy(module->parameters[i]);
    }
    for (size_t i = 0; i < module->child_count; ++i) {
        if (module->children[i] != NULL && module->children[i]->destroy != NULL) {
            module->children[i]->destroy(module->children[i]);
        }
    }
    free(module->parameters);
    free(module->children);
    free(module->name);
    memset(module, 0, sizeof(*module));
}

int nn_module_register_parameter(nn_module* module, nn_parameter* parameter)
{
    if (module == NULL || parameter == NULL) return -1;
    for (size_t i = 0; i < module->parameter_count; ++i) {
        if (module->parameters[i] == parameter) return -1;
    }
    if (nn_reserve_parameters(module) != 0) return -1;
    module->parameters[module->parameter_count++] = parameter;
    return 0;
}

int nn_module_register_child(nn_module* module, nn_module* child)
{
    if (module == NULL || child == NULL || nn_module_contains(child, module)) {
        return -1;
    }
    for (size_t i = 0; i < module->child_count; ++i) {
        if (module->children[i] == child) return -1;
    }
    if (nn_reserve_children(module) != 0) return -1;
    module->children[module->child_count++] = child;
    return 0;
}

int nn_register_owned_parameter(nn_module* module, nn_parameter** parameter)
{
    if (parameter == NULL || *parameter == NULL ||
        nn_module_register_parameter(module, *parameter) != 0) {
        if (parameter != NULL && *parameter != NULL) {
            nn_parameter_destroy(*parameter);
            *parameter = NULL;
        }
        return -1;
    }
    return 0;
}

int nn_register_owned_child(nn_module* module, nn_module* child)
{
    if (child == NULL || nn_module_register_child(module, child) != 0) {
        if (child != NULL && child->destroy != NULL) child->destroy(child);
        return -1;
    }
    return 0;
}

size_t nn_module_parameter_count(const nn_module* module)
{
    size_t count;

    if (module == NULL) return 0;
    count = module->parameter_count;
    for (size_t i = 0; i < module->child_count; ++i) {
        size_t child_count = nn_module_parameter_count(module->children[i]);
        if (child_count > SIZE_MAX - count) return SIZE_MAX;
        count += child_count;
    }
    return count;
}

nn_parameter* nn_module_parameter_at(const nn_module* module, size_t index)
{
    if (module == NULL) return NULL;
    if (index < module->parameter_count) return module->parameters[index];
    index -= module->parameter_count;
    for (size_t i = 0; i < module->child_count; ++i) {
        size_t child_count = nn_module_parameter_count(module->children[i]);
        if (index < child_count) {
            return nn_module_parameter_at(module->children[i], index);
        }
        index -= child_count;
    }
    return NULL;
}

ag_tensor* nn_module_forward(const nn_module* module, const ag_tensor* input)
{
    if (module == NULL || input == NULL || module->forward == NULL) return NULL;
    return module->forward(module, input);
}

void nn_module_set_training(nn_module* module, int training)
{
    if (module == NULL) return;
    module->training = training != 0;
    for (size_t i = 0; i < module->child_count; ++i) {
        nn_module_set_training(module->children[i], training);
    }
}

int nn_module_is_training(const nn_module* module)
{
    return module != NULL && module->training != 0;
}
