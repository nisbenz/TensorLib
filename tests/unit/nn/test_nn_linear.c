#include <stdio.h>
#include <string.h>

#include <tensorlib/nn.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static ag_tensor* make_ag(int ndim,
                          const int* dims,
                          const float* values,
                          int requires_grad)
{
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) {
        raw->storage->data[i] = values[i];
    }
    return ag_from_owned_tensor(raw, requires_grad);
}

static void set_known_parameters(nn_linear* layer)
{
    float weights[] = {1, 2, 3, -1, 0, 1};
    float biases[] = {0.5f, -0.5f};
    for (int i = 0; i < 6; ++i) {
        layer->weight->value->value->storage->data[i] = weights[i];
    }
    if (layer->bias != NULL) {
        for (int i = 0; i < 2; ++i) {
            layer->bias->value->value->storage->data[i] = biases[i];
        }
    }
}

static void test_constructor(void)
{
    nn_rng first_rng;
    nn_rng second_rng;
    nn_linear* first;
    nn_linear* second;
    char mutable_name[] = "projection";

    nn_rng_seed(&first_rng, 123);
    nn_rng_seed(&second_rng, 123);
    first = nn_linear_create(mutable_name, 3, 2, -7,
                             NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO, &first_rng);
    second = nn_linear_create("projection", 3, 2, 1,
                              NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO, &second_rng);
    CHECK(first != NULL && second != NULL);
    if (first != NULL && second != NULL) {
        mutable_name[0] = 'X';
        CHECK(strcmp(first->base.type_name, "Linear") == 0);
        CHECK(strcmp(first->base.name, "projection") == 0);
        CHECK(first->base.training == 1);
        CHECK(first->base.forward != NULL && first->base.destroy != NULL);
        CHECK(first->in_features == 3 && first->out_features == 2);
        CHECK(first->use_bias == 1);
        CHECK(first->base.parameter_count == 2);
        CHECK(first->base.parameters[0] == first->weight);
        CHECK(first->base.parameters[1] == first->bias);
        CHECK(strcmp(first->weight->name, "projection.weight") == 0);
        CHECK(strcmp(first->bias->name, "projection.bias") == 0);
        CHECK(first->weight->value->value->dims[0] == 2);
        CHECK(first->weight->value->value->dims[1] == 3);
        CHECK(first->bias->value->value->dims[0] == 2);
        for (int i = 0; i < 6; ++i) {
            CHECK(first->weight->value->value->storage->data[i] ==
                  second->weight->value->value->storage->data[i]);
        }
    }
    nn_linear_destroy(first);
    nn_linear_destroy(second);
}

static void test_forward_and_backward(void)
{
    int dims[] = {2, 3};
    float values[] = {1, 2, 3, 4, 5, 6};
    float expected[] = {14.5f, 1.5f, 32.5f, 1.5f};
    float expected_input_grad[] = {0, 2, 4, 0, 2, 4};
    float expected_weight_grad[] = {5, 7, 9, 5, 7, 9};
    nn_linear* layer = nn_linear_create(
        "linear", 3, 2, 1, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    ag_tensor* input = make_ag(2, dims, values, 1);
    ag_tensor* output;
    ag_tensor* row_loss;
    ag_tensor* loss;

    CHECK(layer != NULL && input != NULL);
    set_known_parameters(layer);
    output = nn_linear_forward(layer, input);
    CHECK(output != NULL);
    CHECK(output->value->ndim == 2);
    CHECK(output->value->dims[0] == 2 && output->value->dims[1] == 2);
    for (int i = 0; i < 4; ++i) {
        CHECK(output->value->storage->data[i] == expected[i]);
    }
    CHECK(output->creator != NULL);
    row_loss = ag_sum(output, 1, 0);
    loss = ag_sum(row_loss, 0, 0);
    CHECK(ag_backward(loss) == 0);
    for (int i = 0; i < 6; ++i) {
        CHECK(input->grad->storage->data[i] == expected_input_grad[i]);
        CHECK(layer->weight->value->grad->storage->data[i] ==
              expected_weight_grad[i]);
    }
    CHECK(layer->bias->value->grad->storage->data[0] == 2.0f);
    CHECK(layer->bias->value->grad->storage->data[1] == 2.0f);
    ag_tensor_release(loss);
    ag_tensor_release(row_loss);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_linear_destroy(layer);
}

static void test_vector_batch_and_no_bias(void)
{
    int vector_dims[] = {3};
    int batch_dims[] = {2, 2, 3};
    float vector[] = {1, 2, 3};
    float batch[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    float expected[] = {14.5f, 1.5f, 32.5f, 1.5f,
                        14.5f, 1.5f, 32.5f, 1.5f};
    nn_linear* layer = nn_linear_create(
        "batched", 3, 2, 1, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_linear* no_bias = nn_linear_create(
        "plain", 3, 2, 0, NN_INIT_ZERO, NN_INIT_HE_NORMAL, NULL);
    ag_tensor* vector_input = make_ag(1, vector_dims, vector, 0);
    ag_tensor* batch_input = make_ag(3, batch_dims, batch, 0);
    ag_tensor* vector_output;
    ag_tensor* batch_output;

    set_known_parameters(layer);
    vector_output = nn_module_forward(&layer->base, vector_input);
    batch_output = nn_linear_forward(layer, batch_input);
    CHECK(vector_output->value->dims[0] == 2);
    CHECK(vector_output->value->storage->data[0] == 14.5f);
    CHECK(vector_output->value->storage->data[1] == 1.5f);
    for (int i = 0; i < 8; ++i) {
        CHECK(batch_output->value->storage->data[i] == expected[i]);
    }
    CHECK(no_bias != NULL);
    CHECK(no_bias->bias == NULL && no_bias->base.parameter_count == 1);
    ag_tensor_release(batch_output);
    ag_tensor_release(vector_output);
    ag_tensor_release(batch_input);
    ag_tensor_release(vector_input);
    nn_linear_destroy(no_bias);
    nn_linear_destroy(layer);
}

static void test_weight_update_refreshes_packed_state(void)
{
    int input_dims[] = {2, 3};
    float values[] = {1, 2, 3, 4, 5, 6};
    nn_linear* layer = nn_linear_create(
        "cached", 3, 2, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    ag_tensor* input = make_ag(2, input_dims, values, 0);
    ag_tensor* first;
    ag_tensor* second;

    CHECK(layer != NULL && input != NULL);
    if (layer == NULL || input == NULL) {
        ag_tensor_release(input);
        nn_linear_destroy(layer);
        return;
    }
    for (int i = 0; i < 6; ++i) {
        layer->weight->value->value->storage->data[i] = 1.0f;
    }
    tensor_mark_modified(layer->weight->value->value);
    first = nn_linear_forward(layer, input);
    CHECK(first != NULL);
    if (first != NULL) {
        CHECK(first->value->storage->data[0] == 6.0f);
        CHECK(first->value->storage->data[1] == 6.0f);
    }
    for (int i = 0; i < 6; ++i) {
        layer->weight->value->value->storage->data[i] = 2.0f;
    }
    tensor_mark_modified(layer->weight->value->value);
    second = nn_linear_forward(layer, input);
    CHECK(second != NULL);
    if (second != NULL) {
        CHECK(second->value->storage->data[0] == 12.0f);
        CHECK(second->value->storage->data[1] == 12.0f);
    }
    ag_tensor_release(second);
    ag_tensor_release(first);
    ag_tensor_release(input);
    nn_linear_destroy(layer);
}

static void test_invalid(void)
{
    int bad_dims[] = {2, 4};
    float values[8] = {0};
    ag_tensor* bad = make_ag(2, bad_dims, values, 0);
    nn_linear* layer = nn_linear_create(
        "valid", 3, 2, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    CHECK(nn_linear_create(NULL, 3, 2, 0,
                           NN_INIT_ZERO, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_linear_create("", 3, 2, 0,
                           NN_INIT_ZERO, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_linear_create("bad", 0, 2, 0,
                           NN_INIT_ZERO, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_linear_create("bad", 3, -1, 0,
                           NN_INIT_ZERO, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_linear_create("bad", 3, 2, 0,
                           NN_INIT_HE_NORMAL, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_linear_forward(NULL, bad) == NULL);
    CHECK(nn_linear_forward(layer, NULL) == NULL);
    CHECK(nn_linear_forward(layer, bad) == NULL);
    nn_linear_destroy(NULL);
    ag_tensor_release(bad);
    nn_linear_destroy(layer);
}

int main(void)
{
    test_constructor();
    test_forward_and_backward();
    test_vector_batch_and_no_bias();
    test_weight_update_refreshes_packed_state();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d Linear contract checks failed\n", failures);
        return 1;
    }
    printf("All Linear public-contract checks passed.\n");
    return 0;
}
