#include <math.h>
#include <stdio.h>

#include <tensorlib/nn.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static tensor* constant_gradient(const tensor* value, float fill)
{
    tensor* result = t_alloc(value->ndim, value->dims);

    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) {
        result->storage->data[i] = fill;
    }
    return result;
}

static nn_linear* two_parameter_model(const char* name)
{
    nn_linear* root = nn_linear_create(
        name, 192, 192, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_linear* child = nn_linear_create(
        "adam_child", 192, 192, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);

    if (root == NULL || child == NULL ||
        nn_module_register_child(&root->base, &child->base) != 0) {
        nn_linear_destroy(root);
        nn_linear_destroy(child);
        return NULL;
    }
    root->weight->value->grad = constant_gradient(
        root->weight->value->value, 0.5f);
    child->weight->value->grad = constant_gradient(
        child->weight->value->value, -0.25f);
    return root;
}

static tensor* offset_matrix(const tensor* shape, float fill)
{
    int base_dims[2] = {shape->dims[0] + 1, shape->dims[1]};
    tensor* base = t_alloc(2, base_dims);
    tensor* view;

    if (base == NULL) return NULL;
    view = t_slice(base, 0, 1, base_dims[0]);
    t_free(base);
    if (view == NULL) return NULL;
    for (int i = 0; i < tensor_numel(view); ++i) {
        view->storage->data[view->offset + i] = fill;
    }
    return view;
}

static tensor* strided_matrix(const tensor* shape, float fill)
{
    tensor* base = t_alloc(shape->ndim, shape->dims);
    tensor* view;

    if (base == NULL) return NULL;
    for (int i = 0; i < tensor_numel(base); ++i) {
        base->storage->data[i] = fill;
    }
    view = t_transpose(base, 0, 1);
    t_free(base);
    return view;
}

static void replace_tensor(tensor** destination, tensor* replacement)
{
    t_free(*destination);
    *destination = replacement;
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

static void test_parallel_matches_serial_with_clipping(void)
{
    nn_linear* serial = two_parameter_model("serial");
    nn_linear* parallel = two_parameter_model("parallel");
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* serial_optimizer;
    nn_adamw* parallel_optimizer;

    config.max_grad_norm = 1.0f;
    config.weight_decay = 0.02f;
    serial_optimizer = nn_adamw_create(&serial->base, &config);
    parallel_optimizer = nn_adamw_create(&parallel->base, &config);
    CHECK(serial_optimizer != NULL && parallel_optimizer != NULL);
    for (int step = 0; step < 3; ++step) {
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
        CHECK(nn_adamw_step(serial_optimizer) == 0);
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        CHECK(nn_adamw_step(parallel_optimizer) == 0);
    }
    for (size_t parameter = 0; parameter < 2; ++parameter) {
        tensor* serial_value = serial_optimizer->parameters[parameter]->value->value;
        tensor* parallel_value = parallel_optimizer->parameters[parameter]->value->value;
        CHECK(serial_optimizer->steps[parameter] == 3);
        CHECK(parallel_optimizer->steps[parameter] == 3);
        for (int element = 0; element < tensor_numel(serial_value); ++element) {
            CHECK(serial_value->storage->data[element] ==
                  parallel_value->storage->data[element]);
            CHECK(serial_optimizer->first_moments[parameter]->storage->data[element] ==
                  parallel_optimizer->first_moments[parameter]->storage->data[element]);
            CHECK(serial_optimizer->second_moments[parameter]->storage->data[element] ==
                  parallel_optimizer->second_moments[parameter]->storage->data[element]);
        }
    }

    nn_adamw_destroy(parallel_optimizer);
    nn_adamw_destroy(serial_optimizer);
    nn_linear_destroy(parallel);
    nn_linear_destroy(serial);
}

static void test_parallel_validation_layout_paths(void)
{
    nn_linear* model = two_parameter_model("layouts");
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* optimizer = nn_adamw_create(&model->base, &config);
    tensor* first_shape = optimizer->parameters[0]->value->value;
    tensor* second_shape = optimizer->parameters[1]->value->value;
    tensor* offset_value = offset_matrix(first_shape, 2.0f);
    tensor* offset_gradient = offset_matrix(first_shape, 0.5f);
    tensor* offset_first = offset_matrix(first_shape, 0.0f);
    tensor* offset_second = offset_matrix(first_shape, 0.0f);
    tensor* strided_value = strided_matrix(second_shape, -1.0f);
    tensor* strided_gradient = strided_matrix(second_shape, -0.25f);
    tensor* strided_first = strided_matrix(second_shape, 0.0f);
    tensor* strided_second = strided_matrix(second_shape, 0.0f);

    replace_tensor(&optimizer->parameters[0]->value->value,
                   offset_value);
    replace_tensor(&optimizer->parameters[0]->value->grad,
                   offset_gradient);
    replace_tensor(&optimizer->first_moments[0], offset_first);
    replace_tensor(&optimizer->second_moments[0], offset_second);
    replace_tensor(&optimizer->parameters[1]->value->value,
                   strided_value);
    replace_tensor(&optimizer->parameters[1]->value->grad,
                   strided_gradient);
    replace_tensor(&optimizer->first_moments[1], strided_first);
    replace_tensor(&optimizer->second_moments[1], strided_second);

    CHECK(optimizer->parameters[0]->value->value->offset != 0);
    CHECK(is_contiguous(optimizer->parameters[0]->value->value));
    CHECK(!is_contiguous(optimizer->parameters[1]->value->value));
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(4);
#endif
    CHECK(nn_adamw_step(optimizer) == 0);
    for (size_t parameter = 0; parameter < 2; ++parameter) {
        tensor* value = optimizer->parameters[parameter]->value->value;
        int index = value->offset;
        CHECK(optimizer->steps[parameter] == 1);
        CHECK(value->storage->version == 1);
        CHECK(optimizer->first_moments[parameter]->storage->version == 1);
        CHECK(optimizer->second_moments[parameter]->storage->version == 1);
        CHECK(value->storage->data[index] != (parameter == 0 ? 2.0f : -1.0f));
    }
    CHECK(optimizer->parameters[0]->value->value->storage->data[0] == 0.0f);

    nn_adamw_destroy(optimizer);
    nn_linear_destroy(model);
}

#ifdef _OPENMP
static void check_unmodified_optimizer_state(const nn_adamw* optimizer)
{
    for (size_t parameter = 0; parameter < 2; ++parameter) {
        CHECK(optimizer->steps[parameter] == 0);
        CHECK(optimizer->parameters[parameter]->value->value->storage->version == 0);
        CHECK(optimizer->first_moments[parameter]->storage->version == 0);
        CHECK(optimizer->second_moments[parameter]->storage->version == 0);
    }
    CHECK(optimizer->parameters[0]->value->value->storage->data[0] == 0.0f);
    CHECK(optimizer->first_moments[0]->storage->data[0] == 0.0f);
    CHECK(optimizer->second_moments[0]->storage->data[0] == 0.0f);
}

static void test_parallel_validation_is_transactional(void)
{
    nn_linear* model = two_parameter_model("transactional_parallel");
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw* optimizer = nn_adamw_create(&model->base, &config);
    tensor* later_value = optimizer->parameters[1]->value->value;
    tensor* later_second = optimizer->second_moments[1];

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(4);
#endif
    later_value->storage->data[13] = NAN;
    CHECK(nn_adamw_step(optimizer) != 0);
    CHECK(isnan(later_value->storage->data[13]));
    check_unmodified_optimizer_state(optimizer);

    later_value->storage->data[13] = 0.0f;
    later_second->storage->data[23] = INFINITY;
    CHECK(nn_adamw_step(optimizer) != 0);
    CHECK(isinf(later_second->storage->data[23]));
    CHECK(later_value->storage->data[13] == 0.0f);
    check_unmodified_optimizer_state(optimizer);

    nn_adamw_destroy(optimizer);
    nn_linear_destroy(model);
}
#endif

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
    test_parallel_matches_serial_with_clipping();
    test_parallel_validation_layout_paths();
#ifdef _OPENMP
    test_parallel_validation_is_transactional();
#endif
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d AdamW checks failed\n", failures);
        return 1;
    }
    printf("All AdamW checks passed.\n");
    return 0;
}
