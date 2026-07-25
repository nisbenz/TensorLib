#include <stdint.h>
#include <stdlib.h>
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
