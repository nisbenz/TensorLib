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

static tensor* gradient(float value)
{
    int dims[] = {1, 1};
    tensor* result = t_alloc(2, dims);
    if (result != NULL) result->storage->data[0] = value;
    return result;
}

static void test_round_trip_and_resume(void)
{
    const char* path = "tensorlib_checkpoint_test.bin";
    nn_linear* first = nn_linear_create(
        "model", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_linear* second = nn_linear_create(
        "model", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    nn_adamw_config config = nn_adamw_default_config();
    nn_adamw_config other = nn_adamw_default_config();
    nn_adamw* first_optimizer;
    nn_adamw* second_optimizer;
    nn_rng first_rng;
    nn_rng second_rng;

    config.learning_rate = 0.025f;
    config.weight_decay = 0.02f;
    other.learning_rate = 0.5f;
    first_optimizer = nn_adamw_create(&first->base, &config);
    second_optimizer = nn_adamw_create(&second->base, &other);
    first->weight->value->value->storage->data[0] = 2.0f;
    first->weight->value->grad = gradient(0.75f);
    CHECK(nn_adamw_step(first_optimizer) == 0);
    nn_adamw_zero_grad(first_optimizer);
    nn_rng_seed(&first_rng, 123);
    (void)nn_rng_uniform(&first_rng, 0.0f, 1.0f);
    nn_rng_seed(&second_rng, 999);

    CHECK(nn_checkpoint_save(
        path, &first->base, first_optimizer, &first_rng) == 0);
    CHECK(nn_checkpoint_load(path, &second->base, NULL, NULL) != 0);
    CHECK(second->weight->value->value->storage->data[0] == 0.0f);
    CHECK(nn_checkpoint_load(
        path, &second->base, second_optimizer, &second_rng) == 0);
    CHECK(second->weight->value->value->storage->data[0] ==
          first->weight->value->value->storage->data[0]);
    CHECK(second_optimizer->config.learning_rate ==
          first_optimizer->config.learning_rate);
    CHECK(second_optimizer->config.weight_decay ==
          first_optimizer->config.weight_decay);
    CHECK(second_optimizer->steps[0] == first_optimizer->steps[0]);
    CHECK(second_optimizer->first_moments[0]->storage->data[0] ==
          first_optimizer->first_moments[0]->storage->data[0]);
    CHECK(second_optimizer->second_moments[0]->storage->data[0] ==
          first_optimizer->second_moments[0]->storage->data[0]);
    CHECK(second_rng.state == first_rng.state);

    first->weight->value->grad = gradient(-0.25f);
    second->weight->value->grad = gradient(-0.25f);
    CHECK(nn_adamw_step(first_optimizer) == 0);
    CHECK(nn_adamw_step(second_optimizer) == 0);
    CHECK(second->weight->value->value->storage->data[0] ==
          first->weight->value->value->storage->data[0]);
    CHECK(second_optimizer->steps[0] == first_optimizer->steps[0]);

    remove(path);
    nn_adamw_destroy(second_optimizer);
    nn_adamw_destroy(first_optimizer);
    nn_linear_destroy(second);
    nn_linear_destroy(first);
}

static void test_failed_load_is_transactional(void)
{
    const char* path = "tensorlib_checkpoint_bad.bin";
    nn_linear* layer = nn_linear_create(
        "model", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    FILE* file = fopen(path, "wb");
    float before;
    static const char bad[] = "TLCKPT";

    fwrite(bad, 1, sizeof(bad), file);
    fclose(file);
    layer->weight->value->value->storage->data[0] = 7.0f;
    before = layer->weight->value->value->storage->data[0];
    CHECK(nn_checkpoint_load(path, &layer->base, NULL, NULL) != 0);
    CHECK(layer->weight->value->value->storage->data[0] == before);

    remove(path);
    nn_linear_destroy(layer);
}

static void write_u32_native(FILE* file, uint32_t value)
{
    fwrite(&value, sizeof(value), 1, file);
}

static void test_legacy_weights_load(void)
{
    const char* path = "tensorlib_legacy_weights_test.bin";
    static const unsigned char magic[8] = {
        'T', 'L', 'W', 'E', 'I', 'G', 'H', 'T'
    };
    nn_linear* layer = nn_linear_create(
        "legacy", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);
    const char* name = "legacy.weight";
    uint64_t count = 1;
    int32_t dimension = 1;
    float value = 4.25f;
    FILE* file = fopen(path, "wb");

    fwrite(magic, 1, sizeof(magic), file);
    write_u32_native(file, 1);
    write_u32_native(file, 1);
    write_u32_native(file, (uint32_t)strlen(name));
    fwrite(name, 1, strlen(name), file);
    write_u32_native(file, 2);
    fwrite(&dimension, sizeof(dimension), 1, file);
    fwrite(&dimension, sizeof(dimension), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    fclose(file);

    CHECK(nn_checkpoint_load(path, &layer->base, NULL, NULL) == 0);
    CHECK(layer->weight->value->value->storage->data[0] == value);

    remove(path);
    nn_linear_destroy(layer);
}

static void test_invalid(void)
{
    nn_linear* layer = nn_linear_create(
        "invalid", 1, 1, 0, NN_INIT_ZERO, NN_INIT_ZERO, NULL);

    CHECK(nn_checkpoint_save(NULL, &layer->base, NULL, NULL) != 0);
    CHECK(nn_checkpoint_save("", &layer->base, NULL, NULL) != 0);
    CHECK(nn_checkpoint_save("bad.bin", NULL, NULL, NULL) != 0);
    CHECK(nn_checkpoint_load(NULL, &layer->base, NULL, NULL) != 0);
    CHECK(nn_checkpoint_load("missing.bin", &layer->base, NULL, NULL) != 0);

    nn_linear_destroy(layer);
}

static nn_decoder_config decoder_config(float dropout)
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

static ag_tensor* decoder_tokens(void)
{
    int dims[2] = {1, 4};
    float values[4] = {0, 1, 2, 3};
    tensor* raw = t_alloc(2, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < 4; ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, 0);
}

static tensor* decoder_targets(void)
{
    int dims[2] = {1, 4};
    float values[4] = {1, 2, 3, 4};
    tensor* result = t_alloc(2, dims);

    if (result == NULL) return NULL;
    for (int i = 0; i < 4; ++i) result->storage->data[i] = values[i];
    return result;
}

static float decoder_train_step(nn_decoder* decoder,
                                nn_adamw* optimizer,
                                const ag_tensor* tokens,
                                const tensor* targets)
{
    ag_tensor* loss;
    float value = 0.0f;

    nn_adamw_zero_grad(optimizer);
    loss = nn_decoder_loss(decoder, tokens, targets);
    CHECK(loss != NULL);
    if (loss == NULL) return value;
    value = loss->value->storage->data[0];
    CHECK(ag_backward(loss) == 0);
    CHECK(nn_adamw_step(optimizer) == 0);
    ag_tensor_release(loss);
    return value;
}

static void check_decoder_state_equal(const nn_decoder* first,
                                      const nn_decoder* second,
                                      const nn_adamw* first_optimizer,
                                      const nn_adamw* second_optimizer)
{
    size_t count = nn_module_parameter_count(&first->base);

    CHECK(count == nn_module_parameter_count(&second->base));
    CHECK(count == first_optimizer->parameter_count);
    CHECK(count == second_optimizer->parameter_count);
    for (size_t parameter_index = 0; parameter_index < count;
         ++parameter_index) {
        nn_parameter* first_parameter =
            nn_module_parameter_at(&first->base, parameter_index);
        nn_parameter* second_parameter =
            nn_module_parameter_at(&second->base, parameter_index);
        CHECK(strcmp(first_parameter->name, second_parameter->name) == 0);
        CHECK(tensor_numel(first_parameter->value->value) ==
              tensor_numel(second_parameter->value->value));
        CHECK((first_parameter->value->grad == NULL) ==
              (second_parameter->value->grad == NULL));
        for (int value_index = 0;
             value_index < tensor_numel(first_parameter->value->value);
             ++value_index) {
            CHECK(first_parameter->value->value->storage->data[value_index] ==
                  second_parameter->value->value->storage->data[value_index]);
            CHECK(first_optimizer->first_moments[parameter_index]->
                      storage->data[value_index] ==
                  second_optimizer->first_moments[parameter_index]->
                      storage->data[value_index]);
            CHECK(first_optimizer->second_moments[parameter_index]->
                      storage->data[value_index] ==
                  second_optimizer->second_moments[parameter_index]->
                      storage->data[value_index]);
            if (first_parameter->value->grad != NULL &&
                second_parameter->value->grad != NULL) {
                CHECK(first_parameter->value->grad->
                          storage->data[value_index] ==
                      second_parameter->value->grad->
                          storage->data[value_index]);
            }
        }
        CHECK(first_optimizer->steps[parameter_index] ==
              second_optimizer->steps[parameter_index]);
    }
}

static void test_decoder_checkpoint_resume(void)
{
    const char* path = "tensorlib_decoder_checkpoint_test.bin";
    nn_decoder_config config = decoder_config(0.25f);
    nn_decoder_config mismatch_config = config;
    nn_adamw_config optimizer_config = nn_adamw_default_config();
    nn_adamw_config other_config = nn_adamw_default_config();
    nn_rng first_rng;
    nn_rng second_rng;
    nn_rng mismatch_rng;
    nn_decoder* first;
    nn_decoder* second;
    nn_decoder* mismatch;
    nn_adamw* first_optimizer;
    nn_adamw* second_optimizer;
    nn_adamw* mismatch_optimizer;
    ag_tensor* tokens;
    tensor* targets;
    float mismatch_before;
    uint64_t mismatch_rng_before;

    optimizer_config.learning_rate = 0.02f;
    optimizer_config.weight_decay = 0.01f;
    optimizer_config.max_grad_norm = 1.0f;
    other_config.learning_rate = 0.5f;
    nn_rng_seed(&first_rng, 700);
    nn_rng_seed(&second_rng, 900);
    nn_rng_seed(&mismatch_rng, 1100);
    first = nn_decoder_create("decoder", &config, &first_rng);
    second = nn_decoder_create("decoder", &config, &second_rng);
    mismatch_config.channels = 4;
    mismatch = nn_decoder_create("decoder", &mismatch_config, &mismatch_rng);
    first_optimizer = nn_adamw_create(&first->base, &optimizer_config);
    second_optimizer = nn_adamw_create(&second->base, &other_config);
    mismatch_optimizer = nn_adamw_create(
        &mismatch->base, &optimizer_config);
    tokens = decoder_tokens();
    targets = decoder_targets();
    for (int step = 0; step < 3; ++step) {
        (void)decoder_train_step(
            first, first_optimizer, tokens, targets);
    }
    nn_adamw_zero_grad(first_optimizer);
    CHECK(nn_checkpoint_save(
        path, &first->base, first_optimizer, &first_rng) == 0);

    mismatch_before = mismatch->token_embedding->weight->value->value->
        storage->data[0];
    mismatch_rng_before = mismatch_rng.state;
    CHECK(nn_checkpoint_load(
        path, &mismatch->base, mismatch_optimizer, &mismatch_rng) != 0);
    CHECK(mismatch->token_embedding->weight->value->value->
              storage->data[0] == mismatch_before);
    CHECK(mismatch_rng.state == mismatch_rng_before);

    CHECK(nn_checkpoint_load(
        path, &second->base, second_optimizer, &second_rng) == 0);
    CHECK(second_rng.state == first_rng.state);
    CHECK(second_optimizer->config.learning_rate ==
          first_optimizer->config.learning_rate);
    CHECK(second_optimizer->config.weight_decay ==
          first_optimizer->config.weight_decay);
    check_decoder_state_equal(
        first, second, first_optimizer, second_optimizer);

    nn_module_set_training(&first->base, 0);
    nn_module_set_training(&second->base, 0);
    {
        ag_tensor* first_loss =
            nn_decoder_loss(first, tokens, targets);
        ag_tensor* second_loss =
            nn_decoder_loss(second, tokens, targets);
        CHECK(first_loss != NULL && second_loss != NULL);
        CHECK(first_loss->value->storage->data[0] ==
              second_loss->value->storage->data[0]);
        ag_tensor_release(second_loss);
        ag_tensor_release(first_loss);
    }

    nn_module_set_training(&first->base, 1);
    nn_module_set_training(&second->base, 1);
    {
        float first_loss = decoder_train_step(
            first, first_optimizer, tokens, targets);
        float second_loss = decoder_train_step(
            second, second_optimizer, tokens, targets);
        CHECK(first_loss == second_loss);
        CHECK(first_rng.state == second_rng.state);
    }
    check_decoder_state_equal(
        first, second, first_optimizer, second_optimizer);

    remove(path);
    t_free(targets);
    ag_tensor_release(tokens);
    nn_adamw_destroy(second_optimizer);
    nn_adamw_destroy(first_optimizer);
    nn_adamw_destroy(mismatch_optimizer);
    nn_decoder_destroy(mismatch);
    nn_decoder_destroy(second);
    nn_decoder_destroy(first);
}

int main(void)
{
    test_round_trip_and_resume();
    test_failed_load_is_transactional();
    test_legacy_weights_load();
    test_invalid();
    test_decoder_checkpoint_resume();
    if (failures != 0) {
        fprintf(stderr, "%d checkpoint checks failed\n", failures);
        return 1;
    }
    printf("All checkpoint checks passed.\n");
    return 0;
}
