#include <stdio.h>

#include <tensorlib/nn.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static void test_embedding_module(void)
{
    nn_embedding* layer =
        nn_embedding_create("tokens", 4, 3, NN_INIT_ZERO, NULL);
    int dims[] = {2};
    tensor* raw_indices = t_alloc(1, dims);
    ag_tensor* indices;
    ag_tensor* output;

    CHECK(layer != NULL);
    if (layer == NULL || raw_indices == NULL) return;
    CHECK(layer->base.parameter_count == 1);
    CHECK(layer->weight != NULL);
    CHECK(layer->weight->value->value->dims[0] == 4);
    CHECK(layer->weight->value->value->dims[1] == 3);
    CHECK(layer->weight->name != NULL);
    for (int i = 0; i < 12; ++i) {
        layer->weight->value->value->storage->data[i] = (float)i;
    }
    raw_indices->storage->data[0] = 3.0f;
    raw_indices->storage->data[1] = 1.0f;
    indices = ag_from_owned_tensor(raw_indices, 0);
    output = nn_module_forward(&layer->base, indices);
    CHECK(output != NULL);
    if (output != NULL) {
        float expected[] = {9, 10, 11, 3, 4, 5};
        for (int i = 0; i < 6; ++i) {
            CHECK(output->value->storage->data[i] == expected[i]);
        }
    }
    indices->requires_grad = 1;
    CHECK(nn_embedding_forward(layer, indices) == NULL);

    ag_tensor_release(output);
    ag_tensor_release(indices);
    nn_embedding_destroy(layer);
}

static void test_invalid(void)
{
    CHECK(nn_embedding_create(NULL, 4, 3, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_embedding_create("bad", 0, 3, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_embedding_create("bad", 4, 0, NN_INIT_ZERO, NULL) == NULL);
    CHECK(nn_embedding_forward(NULL, NULL) == NULL);
    nn_embedding_destroy(NULL);
}

int main(void)
{
    test_embedding_module();
    test_invalid();
    if (failures != 0) {
        fprintf(stderr, "%d embedding checks failed\n", failures);
        return 1;
    }
    printf("All embedding checks passed.\n");
    return 0;
}
