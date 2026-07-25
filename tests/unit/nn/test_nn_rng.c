#include <tensorlib/nn.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int checks_run;
static int checks_failed;

#define CHECK(condition, ...) do { \
    ++checks_run; \
    if (!(condition)) { \
        ++checks_failed; \
        fprintf(stderr, "FAIL: "); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static int same_float_bits(float a, float b)
{
    uint32_t a_bits;
    uint32_t b_bits;
    memcpy(&a_bits, &a, sizeof(a_bits));
    memcpy(&b_bits, &b, sizeof(b_bits));
    return a_bits == b_bits;
}

static void test_reproducibility(void)
{
    nn_rng first;
    nn_rng second;
    nn_rng different;
    int any_difference = 0;

    nn_rng_seed(&first, UINT64_C(0x0123456789abcdef));
    nn_rng_seed(&second, UINT64_C(0x0123456789abcdef));
    for (int i = 0; i < 128; ++i) {
        float a = nn_rng_uniform(&first, -9.25f, 17.5f);
        float b = nn_rng_uniform(&second, -9.25f, 17.5f);
        CHECK(same_float_bits(a, b),
              "same seed uniform mismatch at sample %d", i);
    }

    nn_rng_seed(&first, UINT64_C(0xfedcba9876543210));
    nn_rng_seed(&second, UINT64_C(0xfedcba9876543210));
    for (int i = 0; i < 128; ++i) {
        float a = nn_rng_normal(&first, 2.75f, 1.125f);
        float b = nn_rng_normal(&second, 2.75f, 1.125f);
        CHECK(same_float_bits(a, b),
              "same seed normal mismatch at sample %d", i);
    }

    nn_rng_seed(&first, UINT64_C(100));
    nn_rng_seed(&different, UINT64_C(101));
    for (int i = 0; i < 32; ++i) {
        float a = nn_rng_uniform(&first, 0.0f, 1.0f);
        float b = nn_rng_uniform(&different, 0.0f, 1.0f);
        if (!same_float_bits(a, b)) any_difference = 1;
    }
    CHECK(any_difference, "different seeds produced identical uniform sequence");

    any_difference = 0;
    nn_rng_seed(&first, UINT64_C(100));
    nn_rng_seed(&different, UINT64_C(101));
    for (int i = 0; i < 32; ++i) {
        float a = nn_rng_normal(&first, 0.0f, 1.0f);
        float b = nn_rng_normal(&different, 0.0f, 1.0f);
        if (!same_float_bits(a, b)) any_difference = 1;
    }
    CHECK(any_difference, "different seeds produced identical normal sequence");
}

static void test_uniform_contract(void)
{
    nn_rng rng;
    int in_range = 1;

    nn_rng_seed(&rng, UINT64_C(424242));
    for (int i = 0; i < 200000; ++i) {
        float value = nn_rng_uniform(&rng, -123.5f, 87.25f);
        if (!(value >= -123.5f && value < 87.25f)) {
            in_range = 0;
            break;
        }
    }
    CHECK(in_range, "uniform range contract violated");
    CHECK(same_float_bits(nn_rng_uniform(&rng, 4.25f, 4.25f), 4.25f),
          "equal uniform bounds did not return the bound");
}

static void test_normal_contract(void)
{
    enum { sample_count = 250000 };
    nn_rng rng;
    double mean = 0.0;
    double m2 = 0.0;

    nn_rng_seed(&rng, UINT64_C(987654321));
    CHECK(same_float_bits(nn_rng_normal(&rng, 6.5f, 0.0f), 6.5f),
          "zero deviation did not return the mean");

    nn_rng_seed(&rng, UINT64_C(987654321));
    for (int i = 1; i <= sample_count; ++i) {
        double value = nn_rng_normal(&rng, -1.75f, 2.5f);
        double delta = value - mean;
        mean += delta / (double)i;
        m2 += delta * (value - mean);
    }

    CHECK(fabs(mean + 1.75) < 0.025,
          "normal sample mean %.9f is outside tolerance", mean);
    CHECK(fabs(sqrt(m2 / sample_count) - 2.5) < 0.025,
          "normal sample deviation is outside tolerance");
}

static void test_invalid_inputs(void)
{
    nn_rng rng;

    nn_rng_seed(&rng, UINT64_C(1));
    CHECK(isnan(nn_rng_uniform(NULL, 0.0f, 1.0f)), "uniform accepted null RNG");
    CHECK(isnan(nn_rng_normal(NULL, 0.0f, 1.0f)), "normal accepted null RNG");
    CHECK(isnan(nn_rng_uniform(&rng, NAN, 1.0f)), "uniform accepted NaN min");
    CHECK(isnan(nn_rng_uniform(&rng, 0.0f, NAN)), "uniform accepted NaN max");
    CHECK(isnan(nn_rng_uniform(&rng, -INFINITY, 1.0f)),
          "uniform accepted infinite min");
    CHECK(isnan(nn_rng_uniform(&rng, 0.0f, INFINITY)),
          "uniform accepted infinite max");
    CHECK(isnan(nn_rng_uniform(&rng, 2.0f, -3.0f)),
          "uniform accepted reversed bounds");
    CHECK(isnan(nn_rng_normal(&rng, NAN, 1.0f)), "normal accepted NaN mean");
    CHECK(isnan(nn_rng_normal(&rng, INFINITY, 1.0f)),
          "normal accepted infinite mean");
    CHECK(isnan(nn_rng_normal(&rng, 0.0f, NAN)),
          "normal accepted NaN deviation");
    CHECK(isnan(nn_rng_normal(&rng, 0.0f, INFINITY)),
          "normal accepted infinite deviation");
    CHECK(isnan(nn_rng_normal(&rng, 0.0f, -0.25f)),
          "normal accepted negative deviation");
    nn_rng_seed(NULL, UINT64_C(123456));
}

int main(void)
{
    test_reproducibility();
    test_uniform_contract();
    test_normal_contract();
    test_invalid_inputs();
    if (checks_failed != 0) {
        fprintf(stderr, "RNG contract: %d/%d checks failed\n",
                checks_failed, checks_run);
        return 1;
    }
    printf("RNG contract: all %d checks passed\n", checks_run);
    return 0;
}
