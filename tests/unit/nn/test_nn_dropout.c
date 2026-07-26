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

static ag_tensor* ones(int count, int requires_grad)
{
    int dims[] = {count};
    tensor* raw = t_alloc(1, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < count; ++i) raw->storage->data[i] = 1.0f;
    return ag_from_owned_tensor(raw, requires_grad);
}

static void test_deterministic_training_and_backward(void)
{
    nn_rng rng;
    nn_rng reference;
    nn_dropout* layer;
    ag_tensor* input = ones(8, 1);
    ag_tensor* output;
    ag_tensor* loss;

    nn_rng_seed(&rng, 1234);
    nn_rng_seed(&reference, 1234);
    layer = nn_dropout_create("drop", 0.25f, &rng);
    output = nn_dropout_forward(layer, input);
    CHECK(output != NULL);
    if (output != NULL) {
        for (int i = 0; i < 8; ++i) {
            float expected =
                nn_rng_uniform(&reference, 0.0f, 1.0f) < 0.25f
                    ? 0.0f : (4.0f / 3.0f);
            CHECK(output->value->storage->data[i] == expected);
        }
    }
    loss = output == NULL ? NULL : ag_sum(output, 0, 0);
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (input != NULL && input->grad != NULL && output != NULL) {
        for (int i = 0; i < 8; ++i) {
            CHECK(input->grad->storage->data[i] ==
                  output->value->storage->data[i]);
        }
    } else {
        CHECK(0);
    }

    ag_tensor_release(loss);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_dropout_destroy(layer);
}

static void test_eval_identity_and_recursive_mode(void)
{
    nn_rng rng;
    nn_dropout* child;
    nn_linear* root;
    ag_tensor* input = ones(3, 0);
    ag_tensor* output;
    uint64_t state;

    nn_rng_seed(&rng, 99);
    child = nn_dropout_create("child", 0.5f, &rng);
    root = nn_linear_create(
        "root", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    CHECK(child != NULL && root != NULL);
    CHECK(nn_module_register_child(&root->base, &child->base) == 0);
    nn_module_set_training(&root->base, 0);
    CHECK(!nn_module_is_training(&root->base));
    CHECK(!nn_module_is_training(&child->base));
    state = rng.state;
    output = nn_dropout_forward(child, input);
    CHECK(output == input);
    CHECK(rng.state == state);
    ag_tensor_release(output);
    nn_module_set_training(&root->base, 1);
    CHECK(nn_module_is_training(&root->base));
    CHECK(nn_module_is_training(&child->base));

    ag_tensor_release(input);
    nn_linear_destroy(root);
}

static void test_zero_probability_and_invalid(void)
{
    nn_dropout* zero = nn_dropout_create("zero", 0.0f, NULL);
    ag_tensor* input = ones(1, 0);
    ag_tensor* output = nn_dropout_forward(zero, input);

    CHECK(output == input);
    CHECK(nn_dropout_create(NULL, 0.0f, NULL) == NULL);
    CHECK(nn_dropout_create("bad", -0.1f, NULL) == NULL);
    CHECK(nn_dropout_create("bad", 1.0f, NULL) == NULL);
    CHECK(nn_dropout_create("bad", NAN, NULL) == NULL);
    CHECK(nn_dropout_create("bad", 0.5f, NULL) == NULL);
    CHECK(nn_dropout_forward(NULL, input) == NULL);
    CHECK(nn_module_is_training(NULL) == 0);
    nn_module_set_training(NULL, 0);

    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_dropout_destroy(zero);
    nn_dropout_destroy(NULL);
}

int main(void)
{
    test_deterministic_training_and_backward();
    test_eval_identity_and_recursive_mode();
    test_zero_probability_and_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d dropout checks failed\n", failures);
        return 1;
    }
    printf("All dropout checks passed.\n");
    return 0;
}
