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

#define CHECK_NEAR(actual, expected, tolerance) \
    CHECK(fabsf((actual) - (expected)) <= (tolerance))

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

static tensor* make_tensor(int ndim, const int* dims, const float* values)
{
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) {
        raw->storage->data[i] = values[i];
    }
    return raw;
}

static void reference_softmax(const float* logits, int classes, float* output)
{
    double maximum = logits[0];
    double sum = 0.0;
    for (int i = 1; i < classes; ++i) {
        if (logits[i] > maximum) maximum = logits[i];
    }
    for (int i = 0; i < classes; ++i) {
        output[i] = (float)exp((double)logits[i] - maximum);
        sum += output[i];
    }
    for (int i = 0; i < classes; ++i) output[i] = (float)(output[i] / sum);
}

static void test_softmax(void)
{
    int dims[] = {2, 4};
    float logits[] = {1, 2, 3, 4, -1000, 0, 1000, -500};
    float shifted[] = {101, 102, 103, 104, -990, 10, 1010, -490};
    ag_tensor* input = make_ag(2, dims, logits, 1);
    ag_tensor* shifted_input = make_ag(2, dims, shifted, 0);
    ag_tensor* probabilities = nn_softmax(input);
    ag_tensor* log_probabilities = nn_log_softmax(input);
    ag_tensor* shifted_probabilities = nn_softmax(shifted_input);

    CHECK(probabilities != NULL && log_probabilities != NULL);
    for (int row = 0; row < 2; ++row) {
        float expected[4];
        float sum = 0.0f;
        reference_softmax(logits + row * 4, 4, expected);
        for (int column = 0; column < 4; ++column) {
            int index = row * 4 + column;
            float actual = probabilities->value->storage->data[index];
            CHECK(isfinite(actual));
            CHECK_NEAR(actual, expected[column], 2.0e-6f);
            CHECK_NEAR(expf(log_probabilities->value->storage->data[index]),
                       actual, 2.0e-6f);
            CHECK_NEAR(shifted_probabilities->value->storage->data[index],
                       actual, 2.0e-6f);
            sum += actual;
        }
        CHECK_NEAR(sum, 1.0f, 2.0e-6f);
    }
    CHECK(probabilities->creator != NULL);
    ag_tensor_release(shifted_probabilities);
    ag_tensor_release(log_probabilities);
    ag_tensor_release(probabilities);
    ag_tensor_release(shifted_input);
    ag_tensor_release(input);
}

static void test_cross_entropy(void)
{
    int logit_dims[] = {2, 3};
    int target_dims[] = {2};
    float logits[] = {1, 2, 3, 1, 0, -1};
    float targets_data[] = {2, 0};
    float expected_probability[6];
    ag_tensor* input = make_ag(2, logit_dims, logits, 1);
    tensor* targets = make_tensor(1, target_dims, targets_data);
    ag_tensor* loss = nn_cross_entropy(input, targets);
    double expected_loss;

    reference_softmax(logits, 3, expected_probability);
    reference_softmax(logits + 3, 3, expected_probability + 3);
    expected_loss = -(log(expected_probability[2]) +
                      log(expected_probability[3])) / 2.0;
    CHECK(loss != NULL);
    CHECK(loss->value->ndim == 0);
    CHECK_NEAR(loss->value->storage->data[0], (float)expected_loss, 2.0e-6f);
    CHECK(ag_backward(loss) == 0);
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            int index = row * 3 + column;
            float expected = expected_probability[index] / 2.0f;
            if (column == (int)targets_data[row]) expected -= 0.5f;
            CHECK_NEAR(input->grad->storage->data[index], expected, 3.0e-6f);
        }
    }
    ag_tensor_release(loss);
    t_free(targets);
    ag_tensor_release(input);
}

static void test_vector_and_rank_three(void)
{
    int vector_dims[] = {3};
    int cube_dims[] = {2, 2, 3};
    int target_dims[] = {2, 2};
    float vector[] = {10000.0f, -10000.0f, 0};
    float cube[] = {1, 2, 3, 3, 2, 1, 0, 0, 0, -1, 0, 1};
    float vector_target = 0;
    float cube_targets[] = {2, 0, 1, 2};
    ag_tensor* vector_logits = make_ag(1, vector_dims, vector, 0);
    ag_tensor* cube_logits = make_ag(3, cube_dims, cube, 0);
    tensor* scalar_target = make_tensor(0, NULL, &vector_target);
    tensor* targets = make_tensor(2, target_dims, cube_targets);
    ag_tensor* probability = nn_softmax(vector_logits);
    ag_tensor* vector_loss = nn_cross_entropy(vector_logits, scalar_target);
    ag_tensor* cube_loss = nn_cross_entropy(cube_logits, targets);

    CHECK(probability != NULL && isfinite(probability->value->storage->data[0]));
    CHECK_NEAR(probability->value->storage->data[0], 1.0f, 0.0f);
    CHECK(vector_loss != NULL && isfinite(vector_loss->value->storage->data[0]));
    CHECK(cube_loss != NULL && cube_loss->value->ndim == 0);
    ag_tensor_release(cube_loss);
    ag_tensor_release(vector_loss);
    ag_tensor_release(probability);
    t_free(targets);
    t_free(scalar_target);
    ag_tensor_release(cube_logits);
    ag_tensor_release(vector_logits);
}

static void test_rank_three_softmax_and_repeated_targets(void)
{
    int logit_dims[] = {2, 2, 4};
    int target_dims[] = {2, 2};
    float logits[] = {10000.0f, 9999.0f, -10000.0f, 0.0f,
                      -3.0f, -2.0f, -1.0f, 0.0f,
                      1.0f, 2.0f, 3.0f, 4.0f,
                      4.0f, 3.0f, 2.0f, 1.0f};
    float targets_data[] = {1, 1, 1, 1};
    ag_tensor* input = make_ag(3, logit_dims, logits, 1);
    tensor* targets = make_tensor(2, target_dims, targets_data);
    ag_tensor* probabilities = nn_softmax(input);
    ag_tensor* loss = nn_cross_entropy(input, targets);

    CHECK(probabilities != NULL && loss != NULL);
    if (probabilities != NULL) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int column = 0; column < 4; ++column) {
                float value = probabilities->value->storage->data[row * 4 + column];
                CHECK(isfinite(value));
                sum += value;
            }
            CHECK_NEAR(sum, 1.0f, 2.0e-6f);
        }
    }
    if (loss != NULL) {
        CHECK(ag_backward(loss) == 0);
        CHECK(input->grad != NULL);
        if (input->grad != NULL) {
            for (int row = 0; row < 4; ++row) {
                CHECK(input->grad->storage->data[row * 4 + 1] < 0.0f);
            }
        }
    }
    ag_tensor_release(loss);
    ag_tensor_release(probabilities);
    t_free(targets);
    ag_tensor_release(input);
}

static void test_invalid(void)
{
    int dims[] = {2, 3};
    int target_dims[] = {2};
    int wrong_dims[] = {1};
    float logits_data[6] = {0};
    float bad_values[] = {-1, 3, 0.5f, NAN, INFINITY};
    ag_tensor* logits = make_ag(2, dims, logits_data, 1);

    CHECK(nn_softmax(NULL) == NULL);
    CHECK(nn_log_softmax(NULL) == NULL);
    CHECK(nn_cross_entropy(NULL, NULL) == NULL);
    for (int i = 0; i < 5; ++i) {
        float values[] = {0, bad_values[i]};
        tensor* targets = make_tensor(1, target_dims, values);
        CHECK(nn_cross_entropy(logits, targets) == NULL);
        t_free(targets);
    }
    {
        float value = 0;
        tensor* wrong = make_tensor(1, wrong_dims, &value);
        CHECK(nn_cross_entropy(logits, wrong) == NULL);
        t_free(wrong);
    }
    ag_tensor_release(logits);
}

int main(void)
{
    test_softmax();
    test_cross_entropy();
    test_vector_and_rank_three();
    test_rank_three_softmax_and_repeated_targets();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d classification checks failed\n", failures);
        return 1;
    }
    printf("All classification public-contract checks passed.\n");
    return 0;
}
