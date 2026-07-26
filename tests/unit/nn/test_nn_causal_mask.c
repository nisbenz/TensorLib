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

static ag_tensor* scores(int requires_grad)
{
    int dims[] = {2, 3, 3};
    tensor* raw = t_alloc(3, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < 18; ++i) raw->storage->data[i] = (float)(i - 9);
    return ag_from_owned_tensor(raw, requires_grad);
}

static void test_mask_and_softmax(void)
{
    ag_tensor* input = scores(1);
    ag_tensor* masked = nn_apply_causal_mask(input);
    ag_tensor* probabilities = nn_softmax(masked);
    ag_tensor* first_sum;
    ag_tensor* loss;

    CHECK(masked != NULL);
    if (masked != NULL) {
        for (int batch = 0; batch < 2; ++batch) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    int index = batch * 9 + row * 3 + column;
                    if (column > row) {
                        CHECK(isinf(masked->value->storage->data[index]));
                        CHECK(masked->value->storage->data[index] < 0.0f);
                    } else {
                        CHECK(masked->value->storage->data[index] ==
                              input->value->storage->data[index]);
                    }
                }
            }
        }
    }
    CHECK(probabilities != NULL);
    if (probabilities != NULL) {
        for (int batch = 0; batch < 2; ++batch) {
            for (int row = 0; row < 3; ++row) {
                float sum = 0.0f;
                for (int column = 0; column < 3; ++column) {
                    int index = batch * 9 + row * 3 + column;
                    float value = probabilities->value->storage->data[index];
                    if (column > row) CHECK(value == 0.0f);
                    sum += value;
                }
                CHECK(fabsf(sum - 1.0f) < 1e-6f);
            }
        }
    }
    first_sum = masked == NULL ? NULL : ag_sum(masked, 0, 0);
    loss = first_sum == NULL ? NULL : ag_sum(first_sum, 0, 0);
    ag_tensor_release(first_sum);
    if (loss != NULL) {
        ag_tensor* final_sum = ag_sum(loss, 0, 0);
        ag_tensor_release(loss);
        loss = final_sum;
    }
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (input != NULL && input->grad != NULL) {
        for (int i = 0; i < 18; ++i) {
            CHECK(input->grad->storage->data[i] == 1.0f);
        }
    } else {
        CHECK(0);
    }

    ag_tensor_release(loss);
    ag_tensor_release(probabilities);
    ag_tensor_release(masked);
    ag_tensor_release(input);
}

static void test_invalid(void)
{
    int vector_dims[] = {3};
    int rectangle_dims[] = {2, 3};
    ag_tensor* vector =
        ag_from_owned_tensor(t_alloc(1, vector_dims), 0);
    ag_tensor* rectangle =
        ag_from_owned_tensor(t_alloc(2, rectangle_dims), 0);

    CHECK(nn_apply_causal_mask(NULL) == NULL);
    CHECK(nn_apply_causal_mask(vector) == NULL);
    CHECK(nn_apply_causal_mask(rectangle) == NULL);

    ag_tensor_release(rectangle);
    ag_tensor_release(vector);
}

int main(void)
{
    test_mask_and_softmax();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d causal-mask checks failed\n", failures);
        return 1;
    }
    printf("All causal-mask checks passed.\n");
    return 0;
}
