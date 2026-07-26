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

static ag_tensor* make_input(int requires_grad)
{
    int dims[] = {2, 3};
    float values[] = {1, 2, 3, 2, 4, 4};
    tensor* raw = t_alloc(2, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < 6; ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static void test_forward_and_backward(void)
{
    nn_layer_norm* layer = nn_layer_norm_create("norm", 3, 1e-5f, 1);
    ag_tensor* input = make_input(1);
    ag_tensor* output = nn_layer_norm_forward(layer, input);
    ag_tensor* first_sum;
    ag_tensor* loss;
    float expected[] = {
        -1.2247356f, 0.0f, 1.2247356f,
        -1.4142056f, 0.7071028f, 0.7071028f
    };

    CHECK(layer != NULL);
    CHECK(output != NULL);
    if (output != NULL) {
        for (int i = 0; i < 6; ++i) {
            CHECK(fabsf(output->value->storage->data[i] - expected[i]) < 2e-5f);
        }
    }
    first_sum = output == NULL ? NULL : ag_sum(output, 0, 0);
    loss = first_sum == NULL ? NULL : ag_sum(first_sum, 0, 0);
    ag_tensor_release(first_sum);
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (layer != NULL && layer->weight->value->grad != NULL) {
        float expected_weight_grad[] = {
            -2.638941f, 0.707103f, 1.931838f
        };
        for (int i = 0; i < 3; ++i) {
            CHECK(fabsf(layer->weight->value->grad->storage->data[i] -
                        expected_weight_grad[i]) < 3e-5f);
            CHECK(fabsf(layer->bias->value->grad->storage->data[i] - 2.0f) <
                  1e-6f);
        }
    } else {
        CHECK(0);
    }
    if (input != NULL && input->grad != NULL) {
        for (int i = 0; i < 6; ++i) {
            CHECK(fabsf(input->grad->storage->data[i]) < 2e-5f);
        }
    } else {
        CHECK(0);
    }

    ag_tensor_release(loss);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_layer_norm_destroy(layer);
}

static void test_non_affine_and_invalid(void)
{
    nn_layer_norm* layer = nn_layer_norm_create("plain", 3, 1e-5f, 0);
    ag_tensor* input = make_input(0);
    ag_tensor* output = nn_module_forward(&layer->base, input);
    int wrong_dims[] = {2, 2};
    ag_tensor* wrong = ag_from_owned_tensor(t_alloc(2, wrong_dims), 0);

    CHECK(layer != NULL);
    CHECK(layer->base.parameter_count == 0);
    CHECK(output != NULL);
    CHECK(nn_layer_norm_create(NULL, 3, 1e-5f, 1) == NULL);
    CHECK(nn_layer_norm_create("bad", 0, 1e-5f, 1) == NULL);
    CHECK(nn_layer_norm_create("bad", 3, 0.0f, 1) == NULL);
    CHECK(nn_layer_norm_create("bad", 3, NAN, 1) == NULL);
    CHECK(nn_layer_norm_forward(layer, wrong) == NULL);

    ag_tensor_release(wrong);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_layer_norm_destroy(layer);
    nn_layer_norm_destroy(NULL);
}

int main(void)
{
    test_forward_and_backward();
    test_non_affine_and_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d LayerNorm checks failed\n", failures);
        return 1;
    }
    printf("All LayerNorm checks passed.\n");
    return 0;
}
