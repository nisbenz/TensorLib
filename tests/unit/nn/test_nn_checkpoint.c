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

int main(void)
{
    test_round_trip_and_resume();
    test_failed_load_is_transactional();
    test_legacy_weights_load();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d checkpoint checks failed\n", failures);
        return 1;
    }
    printf("All checkpoint checks passed.\n");
    return 0;
}
