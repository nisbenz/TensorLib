#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tensorlib/nn.h>

typedef struct {
    nn_module base;
} dummy_module;

static int failures;
static int destroyed_modules;
static int forward_calls;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static char* copy_string(const char* source)
{
    size_t length = strlen(source) + 1;
    char* result = (char*)malloc(length);
    if (result != NULL) memcpy(result, source, length);
    return result;
}

static void dummy_destroy(nn_module* module)
{
    if (module == NULL) return;
    for (size_t i = 0; i < module->parameter_count; ++i) {
        nn_parameter_destroy(module->parameters[i]);
    }
    for (size_t i = 0; i < module->child_count; ++i) {
        module->children[i]->destroy(module->children[i]);
    }
    free(module->parameters);
    free(module->children);
    free(module->name);
    ++destroyed_modules;
    free(module);
}

static ag_tensor* dummy_forward(const nn_module* module, const ag_tensor* input)
{
    (void)module;
    ++forward_calls;
    ag_tensor_retain((ag_tensor*)input);
    return (ag_tensor*)input;
}

static dummy_module* dummy_create(const char* name)
{
    dummy_module* result = (dummy_module*)calloc(1, sizeof(*result));
    if (result == NULL) return NULL;
    result->base.type_name = "dummy";
    result->base.name = copy_string(name);
    result->base.forward = dummy_forward;
    result->base.destroy = dummy_destroy;
    result->base.training = 1;
    return result;
}

static nn_parameter* make_parameter(const char* name)
{
    int dims[] = {1};
    return nn_parameter_create(name, 1, dims, 1, NN_INIT_ZERO, NULL);
}

static ag_tensor* make_vector(const float* values, int count, int requires_grad)
{
    int dims[] = {count};
    tensor* raw = t_alloc(1, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < count; ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static void test_registration_and_traversal(void)
{
    dummy_module* root = dummy_create("root");
    dummy_module* first = dummy_create("first");
    dummy_module* grandchild = dummy_create("grandchild");
    dummy_module* second = dummy_create("second");
    nn_parameter* expected[5];

    for (int i = 0; i < 5; ++i) {
        char name[8];
        snprintf(name, sizeof(name), "p%d", i);
        expected[i] = make_parameter(name);
    }
    CHECK(nn_module_register_parameter(&root->base, expected[0]) == 0);
    CHECK(nn_module_register_parameter(&root->base, expected[1]) == 0);
    CHECK(nn_module_register_parameter(&first->base, expected[2]) == 0);
    CHECK(nn_module_register_parameter(&grandchild->base, expected[3]) == 0);
    CHECK(nn_module_register_parameter(&second->base, expected[4]) == 0);
    CHECK(nn_module_register_child(&first->base, &grandchild->base) == 0);
    CHECK(nn_module_register_child(&root->base, &first->base) == 0);
    CHECK(nn_module_register_child(&root->base, &second->base) == 0);
    CHECK(nn_module_parameter_count(&root->base) == 5);
    for (size_t i = 0; i < 5; ++i) {
        CHECK(nn_module_parameter_at(&root->base, i) == expected[i]);
    }
    CHECK(nn_module_parameter_at(&root->base, 5) == NULL);
    CHECK(nn_module_parameter_count(NULL) == 0);
    CHECK(nn_module_parameter_at(NULL, 0) == NULL);
    CHECK(nn_module_register_parameter(&root->base, expected[0]) != 0);
    CHECK(nn_module_register_parameter(NULL, expected[0]) != 0);
    CHECK(nn_module_register_parameter(&root->base, NULL) != 0);
    CHECK(nn_module_register_child(&root->base, &root->base) != 0);
    CHECK(nn_module_register_child(&root->base, &first->base) != 0);
    CHECK(nn_module_register_child(&grandchild->base, &root->base) != 0);

    root->base.destroy(&root->base);
    CHECK(destroyed_modules == 4);
}

static void test_capacity_and_dispatch(void)
{
    dummy_module* module = dummy_create("growth");
    float value = 3.0f;
    ag_tensor* input = make_vector(&value, 1, 0);
    ag_tensor* output;

    for (int i = 0; i < 20; ++i) {
        char name[16];
        dummy_module* child;
        snprintf(name, sizeof(name), "p%d", i);
        CHECK(nn_module_register_parameter(
            &module->base, make_parameter(name)) == 0);
        snprintf(name, sizeof(name), "c%d", i);
        child = dummy_create(name);
        CHECK(nn_module_register_child(&module->base, &child->base) == 0);
    }
    CHECK(module->base.parameter_capacity >= 20);
    CHECK(module->base.child_capacity >= 20);
    output = nn_module_forward(&module->base, input);
    CHECK(output == input);
    CHECK(forward_calls == 1);
    CHECK(input->ref_count == 2);
    ag_tensor_release(output);
    CHECK(nn_module_forward(NULL, input) == NULL);
    CHECK(nn_module_forward(&module->base, NULL) == NULL);
    module->base.forward = NULL;
    CHECK(nn_module_forward(&module->base, input) == NULL);
    ag_tensor_release(input);
    module->base.destroy(&module->base);
    CHECK(destroyed_modules == 25);
}

static void test_activation(nn_activation activation,
                            const char* name,
                            ag_op operation)
{
    float values[] = {-0.5f, 0.5f};
    ag_tensor* input = make_vector(values, 2, 1);
    ag_tensor* output;

    CHECK(strcmp(activation.name, name) == 0);
    CHECK(activation.context == NULL);
    CHECK(activation.forward != NULL);
    output = activation.forward(&activation, input);
    CHECK(output != NULL);
    CHECK(output->creator != NULL);
    CHECK(output->creator->operation == operation);
    if (operation == AG_OP_RELU) {
        CHECK(output->value->storage->data[0] == 0.0f);
        CHECK(output->value->storage->data[1] == 0.5f);
    }
    ag_tensor_release(output);
    ag_tensor_release(input);
}

static void test_activations(void)
{
    float context = 2.5f;
    char name[] = "custom";
    nn_activation custom = nn_activation_custom(name, NULL, &context);
    nn_activation relu = nn_activation_relu();
    nn_activation relu_again = nn_activation_relu();

    CHECK(relu.name == relu_again.name);
    test_activation(relu, "ReLU", AG_OP_RELU);
    test_activation(nn_activation_gelu(), "GELU", AG_OP_GELU);
    test_activation(nn_activation_sigmoid(), "Sigmoid", AG_OP_SIGMOID);
    test_activation(nn_activation_tanh(), "Tanh", AG_OP_TANH);
    CHECK(custom.name == name);
    CHECK(custom.forward == NULL);
    CHECK(custom.context == &context);
}

int main(void)
{
    test_registration_and_traversal();
    test_capacity_and_dispatch();
    test_activations();
    if (failures != 0) {
        fprintf(stderr, "%d public-contract checks failed\n", failures);
        return 1;
    }
    printf("All module and activation public-contract checks passed.\n");
    return 0;
}
