#include <math.h>
#include <stdio.h>

#include <tensorlib/nn.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static tensor* gradient(int ndim, const int* dims, const float* values)
{
    tensor* result = t_alloc(ndim, dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) {
        result->storage->data[i] = values[i];
    }
    return result;
}

static ag_tensor* scalar(float value, int requires_grad)
{
    int dims[] = {1};
    tensor* raw = t_alloc(1, dims);
    if (raw == NULL) return NULL;
    raw->storage->data[0] = value;
    return ag_from_owned_tensor(raw, requires_grad);
}

static void test_recursive_update_and_zero(void)
{
    nn_linear* root = nn_linear_create(
        "root", 1, 1, 1, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_linear* child = nn_linear_create(
        "child", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_sgd* optimizer;
    float half = 0.5f;
    float one = 1.0f;
    float minus_two = -2.0f;
    int weight_dims[] = {1, 1};
    int bias_dims[] = {1};
    uint64_t weight_version;
    uint64_t bias_version;
    uint64_t child_version;

    CHECK(nn_module_register_child(&root->base, &child->base) == 0);
    root->weight->value->value->storage->data[0] = 1.0f;
    root->bias->value->value->storage->data[0] = 2.0f;
    child->weight->value->value->storage->data[0] = 3.0f;
    root->weight->value->grad = gradient(2, weight_dims, &half);
    root->bias->value->grad = gradient(1, bias_dims, &one);
    child->weight->value->grad = gradient(2, weight_dims, &minus_two);
    root->bias->trainable = 0;
    weight_version = root->weight->value->value->storage->version;
    bias_version = root->bias->value->value->storage->version;
    child_version = child->weight->value->value->storage->version;

    optimizer = nn_sgd_create(&root->base, 0.1f);
    CHECK(optimizer != NULL);
    CHECK(optimizer->module == &root->base);
    CHECK(optimizer->learning_rate == 0.1f);
    CHECK(nn_sgd_step(optimizer) == 0);
    CHECK(fabsf(root->weight->value->value->storage->data[0] - 0.95f) < 1e-6f);
    CHECK(root->bias->value->value->storage->data[0] == 2.0f);
    CHECK(fabsf(child->weight->value->value->storage->data[0] - 3.2f) < 1e-6f);
    CHECK(root->weight->value->value->storage->version == weight_version + 1);
    CHECK(root->bias->value->value->storage->version == bias_version);
    CHECK(child->weight->value->value->storage->version == child_version + 1);
    nn_sgd_zero_grad(optimizer);
    CHECK(root->weight->value->grad == NULL);
    CHECK(root->bias->value->grad == NULL);
    CHECK(child->weight->value->grad == NULL);
    nn_sgd_zero_grad(optimizer);
    nn_sgd_destroy(optimizer);
    nn_linear_destroy(root);
}

static void test_stale_graph(void)
{
    nn_linear* layer = nn_linear_create(
        "stale", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_sgd* optimizer = nn_sgd_create(&layer->base, 0.25f);
    ag_tensor* input = scalar(2.0f, 0);
    ag_tensor* output;
    ag_tensor* loss;

    layer->weight->value->value->storage->data[0] = 3.0f;
    output = nn_linear_forward(layer, input);
    loss = ag_sum(output, 0, 0);
    CHECK(ag_backward(loss) == 0);
    CHECK(nn_sgd_step(optimizer) == 0);
    CHECK(ag_backward(loss) != 0);
    nn_sgd_zero_grad(optimizer);
    ag_tensor_release(loss);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_sgd_destroy(optimizer);
    nn_linear_destroy(layer);
}

static void test_transactional_failure(void)
{
    nn_linear* root = nn_linear_create(
        "transaction", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_linear* child = nn_linear_create(
        "later", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_sgd* optimizer;
    int wrong_dims[] = {2};
    int weight_dims[] = {1, 1};
    float good = 1.0f;
    float bad[] = {1.0f, 2.0f};
    uint64_t version;

    CHECK(nn_module_register_child(&root->base, &child->base) == 0);
    root->weight->value->value->storage->data[0] = 4.0f;
    child->weight->value->value->storage->data[0] = 5.0f;
    root->weight->value->grad = gradient(2, weight_dims, &good);
    child->weight->value->grad = gradient(1, wrong_dims, bad);
    version = root->weight->value->value->storage->version;
    optimizer = nn_sgd_create(&root->base, 0.1f);
    CHECK(nn_sgd_step(optimizer) != 0);
    CHECK(root->weight->value->value->storage->data[0] == 4.0f);
    CHECK(child->weight->value->value->storage->data[0] == 5.0f);
    CHECK(root->weight->value->value->storage->version == version);
    nn_sgd_destroy(optimizer);
    nn_linear_destroy(root);
}

static void test_invalid(void)
{
    nn_module malformed = {0};
    nn_sgd* optimizer;

    CHECK(nn_sgd_create(NULL, 0.1f) == NULL);
    CHECK(nn_sgd_create(&malformed, 0.0f) == NULL);
    CHECK(nn_sgd_create(&malformed, -1.0f) == NULL);
    CHECK(nn_sgd_create(&malformed, NAN) == NULL);
    CHECK(nn_sgd_create(&malformed, INFINITY) == NULL);
    malformed.parameter_count = 1;
    malformed.parameter_capacity = 1;
    optimizer = nn_sgd_create(&malformed, 0.1f);
    CHECK(optimizer != NULL);
    CHECK(nn_sgd_step(optimizer) != 0);
    nn_sgd_zero_grad(optimizer);
    nn_sgd_destroy(optimizer);
    CHECK(nn_sgd_step(NULL) != 0);
    nn_sgd_zero_grad(NULL);
    nn_sgd_destroy(NULL);
}

int main(void)
{
    test_recursive_update_and_zero();
    test_stale_graph();
    test_transactional_failure();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d SGD contract checks failed\n", failures);
        return 1;
    }
    printf("All SGD public-contract checks passed.\n");
    return 0;
}
