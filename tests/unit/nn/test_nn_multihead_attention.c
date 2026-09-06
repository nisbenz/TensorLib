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

static ag_tensor* input_from(const float* values, int time, int requires_grad)
{
    int dims[3] = {1, time, 2};
    tensor* raw = t_alloc(3, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < time * 2; ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static void identity_qkv_projection(nn_multihead_attention* attention)
{
    for (int i = 0; i < 12; ++i) {
        attention->qkv_weight->value->value->storage->data[i] = 0.0f;
    }
    for (int projection = 0; projection < 3; ++projection) {
        int offset = projection * 4;
        attention->qkv_weight->value->value->storage->data[offset] = 1.0f;
        attention->qkv_weight->value->value->storage->data[offset + 3] = 1.0f;
    }
    for (int i = 0; i < 6; ++i) {
        attention->qkv_bias->value->value->storage->data[i] = 0.0f;
    }
}

static void identity_output_projection(nn_linear* layer)
{
    for (int i = 0; i < 4; ++i) {
        layer->weight->value->value->storage->data[i] = 0.0f;
    }
    layer->weight->value->value->storage->data[0] = 1.0f;
    layer->weight->value->value->storage->data[3] = 1.0f;
    for (int i = 0; i < 2; ++i) {
        layer->bias->value->value->storage->data[i] = 0.0f;
    }
}

static nn_multihead_attention* identity_attention(nn_rng* rng)
{
    nn_multihead_attention* attention = nn_multihead_attention_create(
        "attention", 2, 1, 0.0f, rng);

    if (attention == NULL) return NULL;
    identity_qkv_projection(attention);
    identity_output_projection(attention->output);
    return attention;
}

static void reference_attention(const float* input, int time, float* output)
{
    const float scale = sqrtf(2.0f);

    for (int row = 0; row < time; ++row) {
        float maximum = -INFINITY;
        float weights[4] = {0};
        float sum = 0.0f;
        for (int column = 0; column <= row; ++column) {
            float score = (input[row * 2] * input[column * 2] +
                           input[row * 2 + 1] * input[column * 2 + 1]) /
                          scale;
            if (score > maximum) maximum = score;
            weights[column] = score;
        }
        output[row * 2] = 0.0f;
        output[row * 2 + 1] = 0.0f;
        for (int column = 0; column <= row; ++column) {
            weights[column] = expf(weights[column] - maximum);
            sum += weights[column];
        }
        for (int column = 0; column <= row; ++column) {
            float probability = weights[column] / sum;
            output[row * 2] += probability * input[column * 2];
            output[row * 2 + 1] += probability * input[column * 2 + 1];
        }
    }
}

static float output_sum(nn_multihead_attention* attention,
                        const float* values,
                        int time)
{
    ag_tensor* input = input_from(values, time, 0);
    ag_tensor* output = nn_multihead_attention_forward(attention, input);
    float result = 0.0f;

    if (output != NULL) {
        for (int i = 0; i < time * 2; ++i) {
            result += output->value->storage->data[i];
        }
    }
    ag_tensor_release(output);
    ag_tensor_release(input);
    return result;
}

static void test_exact_forward_causality_and_backward(void)
{
    nn_rng rng;
    float values[6] = {1.0f, 0.5f, -0.5f, 1.0f, 2.0f, -1.0f};
    float changed[6] = {1.0f, 0.5f, -0.5f, 1.0f, -7.0f, 9.0f};
    float expected[6];
    ag_tensor* input;
    ag_tensor* output;
    ag_tensor* changed_input;
    ag_tensor* changed_output;
    ag_tensor* channel_sum;
    ag_tensor* loss;
    nn_multihead_attention* attention;
    const float epsilon = 1e-3f;

    nn_rng_seed(&rng, 12);
    attention = identity_attention(&rng);
    input = input_from(values, 3, 1);
    output = nn_multihead_attention_forward(attention, input);
    reference_attention(values, 3, expected);
    CHECK(output != NULL);
    if (output != NULL) {
        CHECK(output->value->dims[0] == 1);
        CHECK(output->value->dims[1] == 3);
        CHECK(output->value->dims[2] == 2);
        for (int i = 0; i < 6; ++i) {
            CHECK(fabsf(output->value->storage->data[i] - expected[i]) < 2e-6f);
        }
    }
    changed_input = input_from(changed, 3, 0);
    changed_output = nn_multihead_attention_forward(attention, changed_input);
    for (int i = 0; i < 4; ++i) {
        CHECK(output->value->storage->data[i] ==
              changed_output->value->storage->data[i]);
    }

    channel_sum = ag_sum(output, 2, 0);
    loss = ag_sum(channel_sum, 1, 0);
    {
        ag_tensor* scalar_loss = ag_sum(loss, 0, 0);
        ag_tensor_release(loss);
        loss = scalar_loss;
    }
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (input != NULL && input->grad != NULL) {
        float original = values[0];
        float plus;
        float minus;
        values[0] = original + epsilon;
        plus = output_sum(attention, values, 3);
        values[0] = original - epsilon;
        minus = output_sum(attention, values, 3);
        values[0] = original;
        CHECK(fabsf(input->grad->storage->data[0] -
                    (plus - minus) / (2.0f * epsilon)) < 2e-3f);
    } else {
        CHECK(0);
    }

    ag_tensor_release(loss);
    ag_tensor_release(channel_sum);
    ag_tensor_release(changed_output);
    ag_tensor_release(changed_input);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_multihead_attention_destroy(attention);
}

static void test_topology_validation_and_eval_rng(void)
{
    nn_rng rng;
    float values[4] = {1, 2, 3, 4};
    ag_tensor* valid = input_from(values, 2, 0);
    int wrong_rank_dims[2] = {2, 2};
    int wrong_width_dims[3] = {1, 2, 3};
    ag_tensor* wrong_rank =
        ag_from_owned_tensor(t_alloc(2, wrong_rank_dims), 0);
    ag_tensor* wrong_width =
        ag_from_owned_tensor(t_alloc(3, wrong_width_dims), 0);
    nn_multihead_attention* attention;
    uint64_t state;

    nn_rng_seed(&rng, 99);
    CHECK(nn_multihead_attention_create("bad", 3, 2, 0, &rng) == NULL);
    CHECK(nn_multihead_attention_create("bad", 2, 0, 0, &rng) == NULL);
    CHECK(nn_multihead_attention_create("bad", 2, 1, 0, NULL) == NULL);
    attention = nn_multihead_attention_create("attention", 2, 2, 0.5f, &rng);
    CHECK(attention != NULL);
    CHECK(attention != NULL && attention->base.child_count == 2);
    CHECK(attention != NULL && nn_module_parameter_count(&attention->base) == 4);
    CHECK(nn_multihead_attention_forward(NULL, valid) == NULL);
    CHECK(nn_multihead_attention_forward(attention, NULL) == NULL);
    CHECK(nn_multihead_attention_forward(attention, wrong_rank) == NULL);
    CHECK(nn_multihead_attention_forward(attention, wrong_width) == NULL);
    nn_module_set_training(&attention->base, 0);
    state = rng.state;
    for (int iteration = 0; iteration < 50; ++iteration) {
        ag_tensor* output = nn_multihead_attention_forward(attention, valid);
        CHECK(output != NULL);
        ag_tensor_release(output);
    }
    CHECK(rng.state == state);
    CHECK(!nn_module_is_training(&attention->output_dropout->base));

    nn_multihead_attention_destroy(attention);
    ag_tensor_release(wrong_width);
    ag_tensor_release(wrong_rank);
    ag_tensor_release(valid);
    nn_multihead_attention_destroy(NULL);
}

static void test_single_token_causal_attention(void)
{
    nn_rng rng;
    float values[2] = {2.0f, -3.0f};
    ag_tensor* input;
    ag_tensor* output;
    nn_multihead_attention* attention;

    nn_rng_seed(&rng, 44);
    attention = identity_attention(&rng);
    input = input_from(values, 1, 0);
    output = nn_multihead_attention_forward(attention, input);
    CHECK(output != NULL);
    if (output != NULL) {
        CHECK(fabsf(output->value->storage->data[0] - values[0]) < 1e-6f);
        CHECK(fabsf(output->value->storage->data[1] - values[1]) < 1e-6f);
    }
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_multihead_attention_destroy(attention);
}

int main(void)
{
    test_exact_forward_causality_and_backward();
    test_topology_validation_and_eval_rng();
    test_single_token_causal_attention();
    if (failures != 0) {
        fprintf(stderr, "%d multi-head-attention checks failed\n", failures);
        return 1;
    }
    printf("All multi-head-attention checks passed.\n");
    return 0;
}
