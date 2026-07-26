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

static void test_exact_first_step_and_zero(void)
{
    nn_linear* layer = nn_linear_create(
        "adam", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* optimizer;
    int dims[] = {1, 1};
    float grad = 0.5f;

    config.learning_rate = 0.1f;
    layer->weight->value->value->storage->data[0] = 1.0f;
    layer->weight->value->grad = gradient(2, dims, &grad);
    optimizer = nn_adamw_create(&layer->base, &config);
    CHECK(optimizer != NULL);
    CHECK(nn_adamw_step(optimizer) == 0);
    CHECK(fabsf(layer->weight->value->value->storage->data[0] - 0.899f) <
          2e-6f);
    CHECK(fabsf(optimizer->first_moments[0]->storage->data[0] - 0.05f) <
          1e-7f);
    CHECK(fabsf(optimizer->second_moments[0]->storage->data[0] - 0.00025f) <
          1e-8f);
    CHECK(optimizer->steps[0] == 1);
    nn_adamw_zero_grad(optimizer);
    CHECK(layer->weight->value->grad == NULL);

    nn_adamw_destroy(optimizer);
    nn_linear_destroy(layer);
}

static void test_global_clipping(void)
{
    nn_linear* layer = nn_linear_create(
        "clip", 2, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    int dims[] = {1, 2};
    float values[] = {3.0f, 4.0f};
    float norm = 0.0f;

    layer->weight->value->grad = gradient(2, dims, values);
    CHECK(nn_clip_grad_norm(&layer->base, 1.0f, &norm) == 0);
    CHECK(fabsf(norm - 5.0f) < 1e-6f);
    CHECK(fabsf(layer->weight->value->grad->storage->data[0] - 0.6f) <
          1e-6f);
    CHECK(fabsf(layer->weight->value->grad->storage->data[1] - 0.8f) <
          1e-6f);
    nn_module_zero_grad(&layer->base);
    CHECK(layer->weight->value->grad == NULL);

    nn_linear_destroy(layer);
}

static void test_transactional_failure_and_topology(void)
{
    nn_linear* root = nn_linear_create(
        "root", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_linear* child = nn_linear_create(
        "child", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* optimizer = nn_adamw_create(&root->base, &config);
    int wrong_dims[] = {2};
    float bad[] = {1.0f, 2.0f};
    float before;

    root->weight->value->value->storage->data[0] = 2.0f;
    root->weight->value->grad = gradient(1, wrong_dims, bad);
    before = root->weight->value->value->storage->data[0];
    CHECK(nn_adamw_step(optimizer) != 0);
    CHECK(root->weight->value->value->storage->data[0] == before);
    CHECK(optimizer->steps[0] == 0);
    ag_zero_grad(root->weight->value);
    CHECK(nn_module_register_child(&root->base, &child->base) == 0);
    CHECK(nn_adamw_step(optimizer) != 0);

    nn_adamw_destroy(optimizer);
    nn_linear_destroy(root);
}

static ag_tensor* scalar_matrix(float value)
{
    int dims[] = {1, 1};
    tensor* raw = t_alloc(2, dims);

    if (raw == NULL) return NULL;
    raw->storage->data[0] = value;
    return ag_from_owned_tensor(raw, 0);
}

static void test_linear_converges(void)
{
    nn_linear* layer = nn_linear_create(
        "fit", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* optimizer;
    ag_tensor* input = scalar_matrix(1.0f);
    ag_tensor* target = scalar_matrix(3.0f);

    config.learning_rate = 0.05f;
    config.weight_decay = 0.0f;
    optimizer = nn_adamw_create(&layer->base, &config);
    for (int step = 0; step < 200; ++step) {
        ag_tensor* output = nn_linear_forward(layer, input);
        ag_tensor* error = ag_sub(output, target);
        ag_tensor* square = ag_mul(error, error);
        ag_tensor* row_loss = ag_sum(square, 0, 0);
        ag_tensor* loss = ag_sum(row_loss, 0, 0);

        CHECK(loss != NULL && ag_backward(loss) == 0);
        CHECK(nn_adamw_step(optimizer) == 0);
        nn_adamw_zero_grad(optimizer);
        ag_tensor_release(loss);
        ag_tensor_release(row_loss);
        ag_tensor_release(square);
        ag_tensor_release(error);
        ag_tensor_release(output);
    }
    CHECK(fabsf(layer->weight->value->value->storage->data[0] - 3.0f) <
          0.002f);

    ag_tensor_release(target);
    ag_tensor_release(input);
    nn_adamw_destroy(optimizer);
    nn_linear_destroy(layer);
}

static void test_invalid(void)
{
    nn_module empty = {0};
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* optimizer;

    CHECK(nn_adamw_create(NULL, &config) == NULL);
    config.beta1 = 1.0f;
    CHECK(nn_adamw_create(&empty, &config) == NULL);
    config = nn_adamw_default_config();
    config.max_grad_norm = -1.0f;
    CHECK(nn_adamw_create(&empty, &config) == NULL);
    config = nn_adamw_default_config();
    optimizer = nn_adamw_create(&empty, &config);
    CHECK(optimizer != NULL);
    CHECK(nn_adamw_step(optimizer) == 0);
    nn_adamw_destroy(optimizer);
    CHECK(nn_adamw_step(NULL) != 0);
    CHECK(nn_clip_grad_norm(NULL, 1.0f, NULL) != 0);
    CHECK(nn_clip_grad_norm(&empty, 0.0f, NULL) != 0);
    nn_module_zero_grad(NULL);
    nn_adamw_zero_grad(NULL);
    nn_adamw_destroy(NULL);
}

int main(void)
{
    test_exact_first_step_and_zero();
    test_global_clipping();
    test_transactional_failure_and_topology();
    test_linear_converges();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d AdamW checks failed\n", failures);
        return 1;
    }
    printf("All AdamW checks passed.\n");
    return 0;
}
