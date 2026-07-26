#include <math.h>
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

static nn_decoder_config test_config(float dropout)
{
    nn_decoder_config config;
    config.vocabulary_size = 5;
    config.context_length = 4;
    config.channels = 8;
    config.head_count = 2;
    config.layer_count = 2;
    config.dropout_probability = dropout;
    config.layer_norm_epsilon = 1e-5f;
    return config;
}

static ag_tensor* token_ids(const float* values,
                            int batch,
                            int time,
                            int requires_grad)
{
    int dims[2] = {batch, time};
    tensor* raw = t_alloc(2, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < batch * time; ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static tensor* target_ids(const float* values, int batch, int time)
{
    int dims[2] = {batch, time};
    tensor* result = t_alloc(2, dims);

    if (result == NULL) return NULL;
    for (int i = 0; i < batch * time; ++i) {
        result->storage->data[i] = values[i];
    }
    return result;
}

static void test_topology_forward_loss_and_backward(void)
{
    nn_rng rng;
    nn_decoder_config config = test_config(0.0f);
    float inputs[8] = {0, 1, 2, 3, 1, 2, 3, 4};
    float targets_values[8] = {1, 2, 3, 4, 2, 3, 4, 0};
    nn_decoder* decoder;
    ag_tensor* tokens;
    tensor* targets;
    ag_tensor* logits;
    ag_tensor* loss;
    size_t parameter_count;

    nn_rng_seed(&rng, 123);
    decoder = nn_decoder_create("decoder", &config, &rng);
    tokens = token_ids(inputs, 2, 4, 0);
    targets = target_ids(targets_values, 2, 4);
    CHECK(decoder != NULL);
    CHECK(decoder != NULL && decoder->block_count == 2);
    CHECK(decoder != NULL && decoder->base.child_count == 6);
    CHECK(decoder != NULL && decoder->blocks[0] != decoder->blocks[1]);
    parameter_count = nn_module_parameter_count(&decoder->base);
    CHECK(parameter_count == 30);
    for (size_t i = 0; i < parameter_count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(&decoder->base, i);
        CHECK(parameter != NULL && parameter->name != NULL);
        for (size_t earlier = 0; earlier < i; ++earlier) {
            CHECK(strcmp(parameter->name,
                         nn_module_parameter_at(
                             &decoder->base, earlier)->name) != 0);
        }
    }
    logits = nn_module_forward(&decoder->base, tokens);
    CHECK(logits != NULL);
    if (logits != NULL) {
        CHECK(logits->value->ndim == 3);
        CHECK(logits->value->dims[0] == 2);
        CHECK(logits->value->dims[1] == 4);
        CHECK(logits->value->dims[2] == 5);
    }
    loss = nn_decoder_loss(decoder, tokens, targets);
    CHECK(loss != NULL && loss->value->ndim == 0);
    CHECK(loss != NULL && isfinite(loss->value->storage->data[0]));
    CHECK(loss != NULL && ag_backward(loss) == 0);
    for (size_t i = 0; i < parameter_count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(&decoder->base, i);
        CHECK(parameter->value->grad != NULL);
    }

    ag_tensor_release(loss);
    ag_tensor_release(logits);
    t_free(targets);
    ag_tensor_release(tokens);
    nn_decoder_destroy(decoder);
}

static void test_validation_and_mode_propagation(void)
{
    nn_rng rng;
    nn_decoder_config config = test_config(0.2f);
    nn_decoder_config invalid = config;
    float valid_values[4] = {0, 1, 2, 3};
    float bad_values[4] = {0, 1, NAN, 3};
    float target_values[4] = {1, 2, 3, 4};
    nn_decoder* decoder;
    ag_tensor* valid;
    ag_tensor* bad;
    ag_tensor* tracked;
    tensor* targets;
    int long_dims[2] = {1, 5};
    ag_tensor* too_long =
        ag_from_owned_tensor(t_alloc(2, long_dims), 0);
    int wrong_targets_dims[2] = {2, 2};
    tensor* wrong_targets = t_alloc(2, wrong_targets_dims);

    nn_rng_seed(&rng, 44);
    invalid.head_count = 3;
    CHECK(nn_decoder_create("bad", &invalid, &rng) == NULL);
    invalid = config;
    invalid.layer_count = 0;
    CHECK(nn_decoder_create("bad", &invalid, &rng) == NULL);
    invalid = config;
    invalid.dropout_probability = 1.0f;
    CHECK(nn_decoder_create("bad", &invalid, &rng) == NULL);
    CHECK(nn_decoder_create(NULL, &config, &rng) == NULL);
    CHECK(nn_decoder_create("bad", &config, NULL) == NULL);
    decoder = nn_decoder_create("decoder", &config, &rng);
    valid = token_ids(valid_values, 1, 4, 0);
    bad = token_ids(bad_values, 1, 4, 0);
    tracked = token_ids(valid_values, 1, 4, 1);
    targets = target_ids(target_values, 1, 4);
    CHECK(nn_decoder_forward(NULL, valid) == NULL);
    CHECK(nn_decoder_forward(decoder, NULL) == NULL);
    CHECK(nn_decoder_forward(decoder, bad) == NULL);
    CHECK(nn_decoder_forward(decoder, tracked) == NULL);
    CHECK(nn_decoder_forward(decoder, too_long) == NULL);
    CHECK(nn_decoder_loss(decoder, valid, wrong_targets) == NULL);
    nn_module_set_training(&decoder->base, 0);
    CHECK(!nn_module_is_training(&decoder->blocks[0]->base));
    CHECK(!nn_module_is_training(
        &decoder->blocks[1]->attention->output_dropout->base));
    nn_module_set_training(&decoder->base, 1);
    CHECK(nn_module_is_training(&decoder->blocks[1]->mlp_dropout->base));

    t_free(wrong_targets);
    ag_tensor_release(too_long);
    t_free(targets);
    ag_tensor_release(tracked);
    ag_tensor_release(bad);
    ag_tensor_release(valid);
    nn_decoder_destroy(decoder);
    nn_decoder_destroy(NULL);
}

static void test_configurable_depth(void)
{
    nn_rng rng;
    nn_decoder_config config = test_config(0.0f);
    float input_values[4] = {0, 1, 2, 3};
    nn_decoder* decoder;
    ag_tensor* tokens;
    ag_tensor* logits;

    config.layer_count = 3;
    nn_rng_seed(&rng, 55);
    decoder = nn_decoder_create("deep", &config, &rng);
    tokens = token_ids(input_values, 1, 4, 0);
    CHECK(decoder != NULL);
    CHECK(decoder != NULL && decoder->block_count == 3);
    CHECK(decoder != NULL && decoder->base.child_count == 7);
    CHECK(decoder != NULL &&
          nn_module_parameter_count(&decoder->base) == 42);
    CHECK(decoder != NULL &&
          strcmp(decoder->blocks[2]->base.name, "deep.blocks.2") == 0);
    logits = nn_decoder_forward(decoder, tokens);
    CHECK(logits != NULL);
    if (logits != NULL) {
        CHECK(logits->value->ndim == 3);
        CHECK(logits->value->dims[0] == 1);
        CHECK(logits->value->dims[1] == 4);
        CHECK(logits->value->dims[2] == 5);
    }

    ag_tensor_release(logits);
    ag_tensor_release(tokens);
    nn_decoder_destroy(decoder);
}

static void test_tiny_batch_overfit(void)
{
    nn_rng rng;
    nn_decoder_config config = test_config(0.0f);
    nn_adamw_config optimizer_config = nn_adamw_default_config();
    float input_values[4] = {0, 1, 2, 3};
    float target_values[4] = {1, 2, 3, 4};
    nn_decoder* decoder;
    nn_adamw* optimizer;
    ag_tensor* tokens;
    tensor* targets;
    float final_loss = INFINITY;

    nn_rng_seed(&rng, 2026);
    decoder = nn_decoder_create("decoder", &config, &rng);
    optimizer_config.learning_rate = 0.03f;
    optimizer_config.weight_decay = 0.0f;
    optimizer_config.max_grad_norm = 1.0f;
    optimizer = nn_adamw_create(&decoder->base, &optimizer_config);
    tokens = token_ids(input_values, 1, 4, 0);
    targets = target_ids(target_values, 1, 4);
    for (int step = 0; step < 300; ++step) {
        ag_tensor* loss;
        nn_adamw_zero_grad(optimizer);
        loss = nn_decoder_loss(decoder, tokens, targets);
        CHECK(loss != NULL);
        if (loss == NULL) break;
        final_loss = loss->value->storage->data[0];
        CHECK(ag_backward(loss) == 0);
        CHECK(nn_adamw_step(optimizer) == 0);
        ag_tensor_release(loss);
        if (final_loss < 0.03f) break;
    }
    {
        ag_tensor* loss = nn_decoder_loss(decoder, tokens, targets);
        ag_tensor* logits;
        CHECK(loss != NULL);
        if (loss != NULL) final_loss = loss->value->storage->data[0];
        CHECK(final_loss < 0.05f);
        ag_tensor_release(loss);
        nn_module_set_training(&decoder->base, 0);
        logits = nn_decoder_forward(decoder, tokens);
        CHECK(logits != NULL);
        if (logits != NULL) {
            for (int row = 0; row < 4; ++row) {
                int best = 0;
                for (int class_index = 1; class_index < 5; ++class_index) {
                    if (logits->value->storage->data[row * 5 + class_index] >
                        logits->value->storage->data[row * 5 + best]) {
                        best = class_index;
                    }
                }
                CHECK(best == (int)target_values[row]);
            }
        }
        ag_tensor_release(logits);
    }

    t_free(targets);
    ag_tensor_release(tokens);
    nn_adamw_destroy(optimizer);
    nn_decoder_destroy(decoder);
}

int main(void)
{
    test_topology_forward_loss_and_backward();
    test_validation_and_mode_propagation();
    test_configurable_depth();
    test_tiny_batch_overfit();
    if (failures != 0) {
        fprintf(stderr, "%d decoder checks failed\n", failures);
        return 1;
    }
    printf("All decoder checks passed.\n");
    return 0;
}
