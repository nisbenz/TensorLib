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

static int same_bits(float a, float b)
{
    uint32_t aa;
    uint32_t bb;
    memcpy(&aa, &a, sizeof(aa));
    memcpy(&bb, &b, sizeof(bb));
    return aa == bb;
}

static void check_reproducible(nn_init_kind kind)
{
    int dims[] = {64, 32};
    nn_rng first_rng;
    nn_rng second_rng;
    nn_rng different_rng;
    nn_parameter* first;
    nn_parameter* second;
    nn_parameter* different;
    int any_difference = 0;

    nn_rng_seed(&first_rng, 12345);
    nn_rng_seed(&second_rng, 12345);
    nn_rng_seed(&different_rng, 12346);
    first = nn_parameter_create("first", 2, dims, 1, kind, &first_rng);
    second = nn_parameter_create("second", 2, dims, 1, kind, &second_rng);
    different = nn_parameter_create(
        "different", 2, dims, 1, kind, &different_rng);
    CHECK(first != NULL && second != NULL && different != NULL);
    if (first != NULL && second != NULL && different != NULL) {
        for (int i = 0; i < 2048; ++i) {
            float a = first->value->value->storage->data[i];
            float b = second->value->value->storage->data[i];
            float c = different->value->value->storage->data[i];
            CHECK(same_bits(a, b));
            if (!same_bits(a, c)) any_difference = 1;
        }
        CHECK(first_rng.state == second_rng.state);
        CHECK(any_difference);
    }
    nn_parameter_destroy(first);
    nn_parameter_destroy(second);
    nn_parameter_destroy(different);
}

static void check_distribution(nn_init_kind kind,
                               int ndim,
                               int* dims,
                               float expected_stddev,
                               float uniform_bound)
{
    nn_rng rng;
    nn_parameter* parameter;
    tensor* value;
    double mean = 0.0;
    double m2 = 0.0;
    int count;

    nn_rng_seed(&rng, 987654321);
    parameter = nn_parameter_create("sample", ndim, dims, 1, kind, &rng);
    CHECK(parameter != NULL);
    if (parameter == NULL) return;
    value = parameter->value->value;
    count = tensor_numel(value);
    for (int i = 1; i <= count; ++i) {
        double sample = value->storage->data[i - 1];
        double delta = sample - mean;
        mean += delta / i;
        m2 += delta * (sample - mean);
        if (uniform_bound > 0.0f) {
            CHECK(sample >= -uniform_bound && sample < uniform_bound);
        }
    }
    CHECK(fabs(mean) < expected_stddev * 0.05 + 1.0e-4);
    CHECK(fabs(sqrt(m2 / count) - expected_stddev) <
          expected_stddev * 0.05 + 1.0e-4);
    CHECK(parameter->value->creator == NULL);
    CHECK(parameter->value->grad == NULL);
    nn_parameter_destroy(parameter);
}

static void test_policies(void)
{
    int matrix[] = {1000, 1000};
    int rank_one[] = {500000};
    int receptive[] = {100, 50, 2, 2};
    float xavier_matrix_bound = sqrtf(6.0f / 2000.0f);
    float he_matrix_bound = sqrtf(6.0f / 1000.0f);

    check_reproducible(NN_INIT_XAVIER_UNIFORM);
    check_reproducible(NN_INIT_XAVIER_NORMAL);
    check_reproducible(NN_INIT_HE_UNIFORM);
    check_reproducible(NN_INIT_HE_NORMAL);
    check_distribution(NN_INIT_XAVIER_UNIFORM, 2, matrix,
                       sqrtf(2.0f / 2000.0f), xavier_matrix_bound);
    check_distribution(NN_INIT_XAVIER_NORMAL, 2, matrix,
                       sqrtf(2.0f / 2000.0f), 0.0f);
    check_distribution(NN_INIT_HE_UNIFORM, 2, matrix,
                       sqrtf(2.0f / 1000.0f), he_matrix_bound);
    check_distribution(NN_INIT_HE_NORMAL, 2, matrix,
                       sqrtf(2.0f / 1000.0f), 0.0f);
    check_distribution(NN_INIT_XAVIER_NORMAL, 1, rank_one,
                       sqrtf(1.0f / 500000.0f), 0.0f);
    check_distribution(NN_INIT_HE_NORMAL, 4, receptive,
                       sqrtf(2.0f / 200.0f), 0.0f);
}

static void test_zero_and_invalid(void)
{
    int dims[] = {2, 3};
    nn_rng rng;
    nn_parameter* zero;
    uint64_t before;

    nn_rng_seed(&rng, 77);
    before = rng.state;
    zero = nn_parameter_create("zero", 2, dims, 1, NN_INIT_ZERO, &rng);
    CHECK(zero != NULL);
    CHECK(rng.state == before);
    if (zero != NULL) {
        for (int i = 0; i < 6; ++i) {
            CHECK(same_bits(zero->value->value->storage->data[i], 0.0f));
        }
    }
    nn_parameter_destroy(zero);
    CHECK(nn_parameter_create(
        "missing", 2, dims, 1, NN_INIT_HE_NORMAL, NULL) == NULL);
    CHECK(nn_parameter_create(
        "invalid", 2, dims, 1, (nn_init_kind)999, &rng) == NULL);
    CHECK(nn_parameter_create(
        "scalar", 0, NULL, 1, NN_INIT_XAVIER_NORMAL, &rng) == NULL);
}

int main(void)
{
    test_zero_and_invalid();
    test_policies();
    if (failures != 0) {
        fprintf(stderr, "%d initialization checks failed\n", failures);
        return 1;
    }
    printf("All initialization public-contract checks passed.\n");
    return 0;
}
