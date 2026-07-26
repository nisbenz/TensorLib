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

static ag_tensor* embeddings(int batch, int time, int channels, int requires_grad)
{
    int dims[3] = {batch, time, channels};
    tensor* raw = t_alloc(3, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) {
        raw->storage->data[i] = (float)(i + 1);
    }
    return ag_from_owned_tensor(raw, requires_grad);
}

static void test_exact_composition_and_backward(void)
{
    nn_positional_embedding* layer = nn_positional_embedding_create(
        "positions", 4, 2, NN_INIT_ZERO, NULL);
    ag_tensor* input = embeddings(2, 3, 2, 1);
    ag_tensor* output;
    ag_tensor* time_sum;
    ag_tensor* batch_sum;
    ag_tensor* loss;
    float expected_positions[6] = {10, 20, 30, 40, 50, 60};

    CHECK(layer != NULL);
    CHECK(layer != NULL && layer->base.child_count == 1);
    CHECK(layer != NULL && nn_module_parameter_count(&layer->base) == 1);
    CHECK(layer != NULL && strcmp(
        nn_module_parameter_at(&layer->base, 0)->name,
        "positions.table.weight") == 0);
    for (int i = 0; i < 8; ++i) {
        layer->table->weight->value->value->storage->data[i] =
            (float)((i + 1) * 10);
    }
    output = nn_module_forward(&layer->base, input);
    CHECK(output != NULL);
    if (output != NULL) {
        CHECK(output->value->ndim == 3);
        CHECK(output->value->dims[0] == 2);
        CHECK(output->value->dims[1] == 3);
        CHECK(output->value->dims[2] == 2);
        for (int batch = 0; batch < 2; ++batch) {
            for (int i = 0; i < 6; ++i) {
                int index = batch * 6 + i;
                CHECK(output->value->storage->data[index] ==
                      input->value->storage->data[index] +
                      expected_positions[i]);
            }
        }
    }
    time_sum = output == NULL ? NULL : ag_sum(output, 2, 0);
    batch_sum = time_sum == NULL ? NULL : ag_sum(time_sum, 1, 0);
    loss = batch_sum == NULL ? NULL : ag_sum(batch_sum, 0, 0);
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (input != NULL && input->grad != NULL) {
        for (int i = 0; i < 12; ++i) {
            CHECK(input->grad->storage->data[i] == 1.0f);
        }
    } else {
        CHECK(0);
    }
    if (layer != NULL && layer->table->weight->value->grad != NULL) {
        for (int position = 0; position < 3; ++position) {
            for (int channel = 0; channel < 2; ++channel) {
                CHECK(layer->table->weight->value->grad->storage->data[
                    position * 2 + channel] == 2.0f);
            }
        }
        CHECK(layer->table->weight->value->grad->storage->data[6] == 0.0f);
        CHECK(layer->table->weight->value->grad->storage->data[7] == 0.0f);
    } else {
        CHECK(0);
    }

    ag_tensor_release(loss);
    ag_tensor_release(batch_sum);
    ag_tensor_release(time_sum);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_positional_embedding_destroy(layer);
}

static void test_validation_and_training_lifecycle(void)
{
    nn_positional_embedding* layer = nn_positional_embedding_create(
        "positions", 3, 2, NN_INIT_ZERO, NULL);
    ag_tensor* valid = embeddings(1, 3, 2, 0);
    ag_tensor* too_long = embeddings(1, 4, 2, 0);
    ag_tensor* wrong_width = embeddings(1, 3, 3, 0);
    int rank_two_dims[2] = {3, 2};
    ag_tensor* rank_two = ag_from_owned_tensor(t_alloc(2, rank_two_dims), 0);

    CHECK(nn_positional_embedding_create(NULL, 3, 2, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_positional_embedding_create("bad", 0, 2, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_positional_embedding_create("bad", 3, 0, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_positional_embedding_forward(NULL, valid) == NULL);
    CHECK(nn_positional_embedding_forward(layer, NULL) == NULL);
    CHECK(nn_positional_embedding_forward(layer, too_long) == NULL);
    CHECK(nn_positional_embedding_forward(layer, wrong_width) == NULL);
    CHECK(nn_positional_embedding_forward(layer, rank_two) == NULL);
    nn_module_set_training(&layer->base, 0);
    CHECK(!nn_module_is_training(&layer->base));
    CHECK(!nn_module_is_training(&layer->table->base));
    nn_module_set_training(&layer->base, 1);
    CHECK(nn_module_is_training(&layer->table->base));
    for (int iteration = 0; iteration < 100; ++iteration) {
        ag_tensor* output = nn_positional_embedding_forward(layer, valid);
        CHECK(output != NULL);
        ag_tensor_release(output);
    }

    ag_tensor_release(rank_two);
    ag_tensor_release(wrong_width);
    ag_tensor_release(too_long);
    ag_tensor_release(valid);
    nn_positional_embedding_destroy(layer);
    nn_positional_embedding_destroy(NULL);
}

int main(void)
{
    test_exact_composition_and_backward();
    test_validation_and_training_lifecycle();
    if (failures != 0) {
        fprintf(stderr, "%d positional-embedding checks failed\n", failures);
        return 1;
    }
    printf("All positional-embedding checks passed.\n");
    return 0;
}
