#include <math.h>
#include <stdio.h>
#include <string.h>

#include <tensorlib/nn.h>

static int checks;
static int failures;

#define CHECK(condition) do { \
    ++checks; \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static void check_zero_parameter(void)
{
    int dims[] = {2, 3, 4};
    char mutable_name[] = "weights";
    nn_parameter* p = nn_parameter_create(
        mutable_name, 3, dims, 1, NN_INIT_ZERO, NULL);
    int nonzero_count = 0;

    CHECK(p != NULL);
    if (p == NULL) return;
    CHECK(p->name != NULL);
    CHECK(strcmp(p->name, "weights") == 0);
    CHECK(p->name != mutable_name);
    mutable_name[0] = 'X';
    CHECK(strcmp(p->name, "weights") == 0);
    CHECK(p->trainable == 1);
    CHECK(p->value != NULL);
    if (p->value != NULL) {
        CHECK(p->value->requires_grad == 1);
        CHECK(p->value->creator == NULL);
        CHECK(p->value->grad == NULL);
        if (p->value->value != NULL) {
            tensor* value = p->value->value;
            CHECK(value->ndim == 3);
            CHECK(value->dims != dims);
            CHECK(value->dims[0] == 2);
            CHECK(value->dims[1] == 3);
            CHECK(value->dims[2] == 4);
            CHECK(value->strides[0] == 12);
            CHECK(value->strides[1] == 4);
            CHECK(value->strides[2] == 1);
            CHECK(value->offset == 0);
            CHECK(is_contiguous(value) == 1);
            CHECK(tensor_numel(value) == 24);
            for (int i = 0; i < 24; ++i) {
                if (value->storage->data[i] != 0.0f) ++nonzero_count;
            }
            CHECK(nonzero_count == 0);
        }
    }
    dims[0] = 99;
    CHECK(p->value->value->dims[0] == 2);
    nn_parameter_destroy(p);
}

static void check_scalar(void)
{
    nn_parameter* p = nn_parameter_create(
        "scalar", 0, NULL, -19, NN_INIT_ZERO, NULL);
    CHECK(p != NULL);
    if (p == NULL) return;
    CHECK(p->trainable == 1);
    CHECK(p->value->requires_grad == 1);
    CHECK(p->value->creator == NULL);
    CHECK(p->value->grad == NULL);
    CHECK(p->value->value->ndim == 0);
    CHECK(tensor_numel(p->value->value) == 1);
    CHECK(is_contiguous(p->value->value) == 1);
    CHECK(p->value->value->storage->data[0] == 0.0f);
    nn_parameter_destroy(p);
}

static void check_trainability(void)
{
    int dims[] = {1};
    nn_parameter* frozen = nn_parameter_create(
        "frozen", 1, dims, 0, NN_INIT_ZERO, NULL);
    nn_parameter* positive = nn_parameter_create(
        "positive", 1, dims, 27, NN_INIT_ZERO, NULL);

    CHECK(frozen != NULL);
    if (frozen != NULL) {
        ag_tensor* derived;
        CHECK(frozen->trainable == 0);
        CHECK(frozen->value->requires_grad == 0);
        derived = ag_neg(frozen->value);
        CHECK(derived != NULL);
        if (derived != NULL) {
            CHECK(derived->requires_grad == 0);
            CHECK(derived->creator == NULL);
            ag_tensor_release(derived);
        }
        nn_parameter_destroy(frozen);
    }
    CHECK(positive != NULL);
    if (positive != NULL) {
        CHECK(positive->trainable == 1);
        CHECK(positive->value->requires_grad == 1);
        nn_parameter_destroy(positive);
    }
}

static void check_invalid_arguments(void)
{
    int good[] = {2, 2};
    int zero[] = {2, 0};
    int negative[] = {2, -1};
    nn_init_kind unsupported[] = {
        NN_INIT_XAVIER_UNIFORM, NN_INIT_XAVIER_NORMAL,
        NN_INIT_HE_UNIFORM, NN_INIT_HE_NORMAL
    };

    CHECK(nn_parameter_create(NULL, 2, good, 1, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_parameter_create("", 2, good, 1, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_parameter_create("bad", -1, good, 1, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_parameter_create("bad", 1, NULL, 1, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_parameter_create("bad", 2, zero, 1, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_parameter_create("bad", 2, negative, 1, NN_INIT_ZERO, NULL) == NULL);
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); ++i) {
        CHECK(nn_parameter_create(
            "unsupported", 2, good, 1, unsupported[i], NULL) == NULL);
    }
    CHECK(nn_parameter_create(
        "unsupported", 2, good, 1, (nn_init_kind)999, NULL) == NULL);
}

static void check_destroy_with_gradient(void)
{
    int dims[] = {2, 2};
    nn_parameter* p = nn_parameter_create(
        "gradient-owner", 2, dims, 1, NN_INIT_ZERO, NULL);
    ag_tensor* row;
    ag_tensor* loss;

    CHECK(p != NULL);
    if (p == NULL) return;
    row = ag_sum(p->value, 0, 0);
    loss = row != NULL ? ag_sum(row, 0, 0) : NULL;
    CHECK(row != NULL);
    CHECK(loss != NULL);
    if (loss != NULL) {
        CHECK(ag_backward(loss) == 0);
        CHECK(p->value->grad != NULL);
        if (p->value->grad != NULL) {
            for (int i = 0; i < 4; ++i) {
                CHECK(p->value->grad->storage->data[i] == 1.0f);
            }
        }
    }
    ag_tensor_release(loss);
    ag_tensor_release(row);
    nn_parameter_destroy(p);
}

int main(void)
{
    check_zero_parameter();
    check_scalar();
    check_trainability();
    check_invalid_arguments();
    nn_parameter_destroy(NULL);
    ++checks;
    check_destroy_with_gradient();
    if (failures != 0) {
        fprintf(stderr, "%d of %d checks failed\n", failures, checks);
        return 1;
    }
    printf("PASS: %d public-contract checks\n", checks);
    return 0;
}
