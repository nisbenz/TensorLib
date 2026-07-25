#include <math.h>
#include <stdint.h>
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

static ag_tensor* make_ag(int ndim, const int* dims, const float* values)
{
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) {
        raw->storage->data[i] = values[i];
    }
    return ag_from_owned_tensor(raw, 0);
}

static tensor* make_tensor(int ndim, const int* dims, const float* values)
{
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) {
        raw->storage->data[i] = values[i];
    }
    return raw;
}

static int same_float_bits(float a, float b)
{
    uint32_t aa;
    uint32_t bb;
    memcpy(&aa, &a, sizeof(aa));
    memcpy(&bb, &b, sizeof(bb));
    return aa == bb;
}

static void test_structure(void)
{
    int hidden[] = {4};
    nn_activation activations[] = {
        nn_activation_relu(),
        nn_activation_custom("identity", NULL, NULL)
    };
    nn_mlp_config config = {
        3, hidden, 1, 2, activations, 1,
        NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO
    };
    nn_rng first_rng;
    nn_rng second_rng;
    nn_mlp* first;
    nn_mlp* second;
    char mutable_name[] = "network";

    nn_rng_seed(&first_rng, 1234);
    nn_rng_seed(&second_rng, 1234);
    first = nn_mlp_create(mutable_name, &config, &first_rng);
    second = nn_mlp_create("network", &config, &second_rng);
    CHECK(first != NULL && second != NULL);
    if (first != NULL && second != NULL) {
        nn_linear* first_layer = (nn_linear*)first->base.children[0];
        nn_linear* output_layer = (nn_linear*)first->base.children[1];
        mutable_name[0] = 'X';
        CHECK(strcmp(first->base.type_name, "MLP") == 0);
        CHECK(strcmp(first->base.name, "network") == 0);
        CHECK(first->base.training == 1);
        CHECK(first->layer_count == 2);
        CHECK(first->base.child_count == 2);
        CHECK(first->activations != activations);
        CHECK(first->activations[0].forward == activations[0].forward);
        CHECK(first->activations[1].forward == NULL);
        CHECK(strcmp(first_layer->base.name, "network.layers.0") == 0);
        CHECK(strcmp(output_layer->base.name, "network.layers.1") == 0);
        CHECK(first_layer->in_features == 3 && first_layer->out_features == 4);
        CHECK(output_layer->in_features == 4 && output_layer->out_features == 2);
        CHECK(nn_module_parameter_count(&first->base) == 4);
        for (size_t i = 0; i < 4; ++i) {
            nn_parameter* a = nn_module_parameter_at(&first->base, i);
            nn_parameter* b = nn_module_parameter_at(&second->base, i);
            CHECK(a != NULL && b != NULL);
            CHECK(a->value->creator == NULL && a->value->grad == NULL);
            for (int j = 0; j < tensor_numel(a->value->value); ++j) {
                CHECK(same_float_bits(a->value->value->storage->data[j],
                                      b->value->value->storage->data[j]));
            }
        }
        CHECK(nn_module_parameter_at(&first->base, 4) == NULL);
    }
    nn_mlp_destroy(first);
    nn_mlp_destroy(second);
}

static void test_invalid(void)
{
    int valid_hidden[] = {4};
    int invalid_hidden[] = {0};
    nn_activation activations[] = {
        nn_activation_relu(),
        nn_activation_custom("identity", NULL, NULL)
    };
    nn_mlp_config config = {
        2, valid_hidden, 1, 2, activations, 1,
        NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO
    };
    nn_rng rng;

    nn_rng_seed(&rng, 1);
    CHECK(nn_mlp_create(NULL, &config, &rng) == NULL);
    CHECK(nn_mlp_create("", &config, &rng) == NULL);
    CHECK(nn_mlp_create("bad", NULL, &rng) == NULL);
    config.input_features = 0;
    CHECK(nn_mlp_create("bad", &config, &rng) == NULL);
    config.input_features = 2;
    config.output_features = 0;
    CHECK(nn_mlp_create("bad", &config, &rng) == NULL);
    config.output_features = 2;
    config.hidden_sizes = NULL;
    CHECK(nn_mlp_create("bad", &config, &rng) == NULL);
    config.hidden_sizes = invalid_hidden;
    CHECK(nn_mlp_create("bad", &config, &rng) == NULL);
    config.hidden_sizes = valid_hidden;
    config.activations = NULL;
    CHECK(nn_mlp_create("bad", &config, &rng) == NULL);
    config.activations = activations;
    CHECK(nn_mlp_create("bad", &config, NULL) == NULL);
    CHECK(nn_mlp_forward(NULL, NULL) == NULL);
    nn_mlp_destroy(NULL);
}

static void test_xor_training(void)
{
    int hidden[] = {8, 8};
    int input_dims[] = {4, 2};
    int target_dims[] = {4};
    float input_values[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float target_values[] = {0, 1, 1, 0};
    nn_activation activations[] = {
        nn_activation_tanh(),
        nn_activation_tanh(),
        nn_activation_custom("identity", NULL, NULL)
    };
    nn_mlp_config config = {
        2, hidden, 2, 2, activations, 1,
        NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO
    };
    nn_rng rng;
    nn_mlp* model;
    nn_sgd* optimizer;
    ag_tensor* input = make_ag(2, input_dims, input_values);
    tensor* targets = make_tensor(1, target_dims, target_values);
    float initial_loss = 0.0f;
    float final_loss;

    nn_rng_seed(&rng, UINT64_C(0xC0FFEE));
    model = nn_mlp_create("xor", &config, &rng);
    optimizer = model != NULL ? nn_sgd_create(&model->base, 0.1f) : NULL;
    CHECK(model != NULL && optimizer != NULL && input != NULL && targets != NULL);
    for (int step = 0; step < 5000; ++step) {
        ag_tensor* logits = nn_mlp_forward(model, input);
        ag_tensor* loss = nn_cross_entropy(logits, targets);
        CHECK(logits != NULL && loss != NULL);
        if (step == 0) initial_loss = loss->value->storage->data[0];
        CHECK(ag_backward(loss) == 0);
        CHECK(nn_sgd_step(optimizer) == 0);
        nn_sgd_zero_grad(optimizer);
        ag_tensor_release(loss);
        ag_tensor_release(logits);
    }
    {
        ag_tensor* logits = nn_module_forward(&model->base, input);
        ag_tensor* loss = nn_cross_entropy(logits, targets);
        final_loss = loss->value->storage->data[0];
        CHECK(final_loss < initial_loss * 0.25f);
        for (int row = 0; row < 4; ++row) {
            float first = logits->value->storage->data[row * 2];
            float second = logits->value->storage->data[row * 2 + 1];
            int predicted = second > first ? 1 : 0;
            CHECK(predicted == (int)target_values[row]);
        }
        printf("XOR loss %.9g -> %.9g\n",
               (double)initial_loss, (double)final_loss);
        ag_tensor_release(loss);
        ag_tensor_release(logits);
    }
    nn_sgd_destroy(optimizer);
    nn_mlp_destroy(model);
    t_free(targets);
    ag_tensor_release(input);
}

int main(void)
{
    test_structure();
    test_invalid();
    test_xor_training();
    if (failures != 0) {
        fprintf(stderr, "%d MLP contract checks failed\n", failures);
        return 1;
    }
    printf("All MLP and XOR public-contract checks passed.\n");
    return 0;
}
