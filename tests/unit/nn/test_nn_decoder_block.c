#include <stdio.h>

#include <tensorlib/nn.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static ag_tensor* block_input(int channels, int requires_grad)
{
    int dims[3] = {2, 3, channels};
    tensor* raw = t_alloc(3, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) {
        raw->storage->data[i] = (float)(i - 5) * 0.1f;
    }
    return ag_from_owned_tensor(raw, requires_grad);
}

static void zero_parameter(nn_parameter* parameter)
{
    tensor* value = parameter->value->value;
    for (int i = 0; i < tensor_numel(value); ++i) {
        value->storage->data[i] = 0.0f;
    }
}

static void zero_residual_branches(nn_decoder_block* block)
{
    zero_parameter(block->attention->qkv_weight);
    zero_parameter(block->attention->qkv_bias);
    zero_parameter(block->attention->output->weight);
    zero_parameter(block->attention->output->bias);
    zero_parameter(block->mlp_input->weight);
    zero_parameter(block->mlp_input->bias);
    zero_parameter(block->mlp_output->weight);
    zero_parameter(block->mlp_output->bias);
}

static void test_residual_forward_backward_and_topology(void)
{
    nn_rng rng;
    nn_decoder_block* block;
    ag_tensor* input;
    ag_tensor* output;
    ag_tensor* channel_sum;
    ag_tensor* time_sum;
    ag_tensor* loss;

    nn_rng_seed(&rng, 7);
    block = nn_decoder_block_create("block", 4, 2, 0.0f, 1e-5f, &rng);
    input = block_input(4, 1);
    CHECK(block != NULL);
    CHECK(block != NULL && block->base.child_count == 6);
    CHECK(block != NULL && nn_module_parameter_count(&block->base) == 12);
    zero_residual_branches(block);
    output = nn_module_forward(&block->base, input);
    CHECK(output != NULL);
    if (output != NULL) {
        CHECK(output->value->ndim == 3);
        CHECK(output->value->dims[0] == 2);
        CHECK(output->value->dims[1] == 3);
        CHECK(output->value->dims[2] == 4);
        for (int i = 0; i < 24; ++i) {
            CHECK(output->value->storage->data[i] ==
                  input->value->storage->data[i]);
        }
    }
    channel_sum = output == NULL ? NULL : ag_sum(output, 2, 0);
    time_sum = channel_sum == NULL ? NULL : ag_sum(channel_sum, 1, 0);
    loss = time_sum == NULL ? NULL : ag_sum(time_sum, 0, 0);
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (input != NULL && input->grad != NULL) {
        for (int i = 0; i < 24; ++i) {
            CHECK(input->grad->storage->data[i] == 1.0f);
        }
    } else {
        CHECK(0);
    }
    for (size_t i = 0; i < nn_module_parameter_count(&block->base); ++i) {
        nn_parameter* parameter = nn_module_parameter_at(&block->base, i);
        CHECK(parameter != NULL);
        CHECK(parameter->value->grad != NULL);
    }

    ag_tensor_release(loss);
    ag_tensor_release(time_sum);
    ag_tensor_release(channel_sum);
    ag_tensor_release(output);
    ag_tensor_release(input);
    nn_decoder_block_destroy(block);
}

static void test_validation_training_and_lifecycle(void)
{
    nn_rng rng;
    nn_decoder_block* block;
    ag_tensor* valid;
    ag_tensor* wrong_width;
    int wrong_rank_dims[2] = {2, 4};
    ag_tensor* wrong_rank =
        ag_from_owned_tensor(t_alloc(2, wrong_rank_dims), 0);
    uint64_t state;

    nn_rng_seed(&rng, 17);
    CHECK(nn_decoder_block_create(NULL, 4, 2, 0, 1e-5f, &rng) == NULL);
    CHECK(nn_decoder_block_create("bad", 3, 2, 0, 1e-5f, &rng) == NULL);
    CHECK(nn_decoder_block_create("bad", 4, 2, 0, 0.0f, &rng) == NULL);
    CHECK(nn_decoder_block_create("bad", 4, 2, 0, 1e-5f, NULL) == NULL);
    block = nn_decoder_block_create("block", 4, 2, 0.25f, 1e-5f, &rng);
    valid = block_input(4, 0);
    wrong_width = block_input(3, 0);
    CHECK(nn_decoder_block_forward(NULL, valid) == NULL);
    CHECK(nn_decoder_block_forward(block, NULL) == NULL);
    CHECK(nn_decoder_block_forward(block, wrong_rank) == NULL);
    CHECK(nn_decoder_block_forward(block, wrong_width) == NULL);
    nn_module_set_training(&block->base, 0);
    state = rng.state;
    for (int iteration = 0; iteration < 30; ++iteration) {
        ag_tensor* output = nn_decoder_block_forward(block, valid);
        CHECK(output != NULL);
        ag_tensor_release(output);
    }
    CHECK(rng.state == state);
    CHECK(!nn_module_is_training(&block->attention->output_dropout->base));
    CHECK(!nn_module_is_training(&block->mlp_dropout->base));

    ag_tensor_release(wrong_width);
    ag_tensor_release(valid);
    ag_tensor_release(wrong_rank);
    nn_decoder_block_destroy(block);
    nn_decoder_block_destroy(NULL);
}

int main(void)
{
    test_residual_forward_backward_and_topology();
    test_validation_training_and_lifecycle();
    if (failures != 0) {
        fprintf(stderr, "%d decoder-block checks failed\n", failures);
        return 1;
    }
    printf("All decoder-block checks passed.\n");
    return 0;
}
