#include "./../../../include/tensorlib/autograd.h"
#include "./../../fixtures/test_common.h"

static ag_tensor* make_ag(int ndim, const int* dims, const float* values) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, 1);
}

static tensor* sequential_gradient(int ndim, const int* dims) {
    tensor* result = t_alloc(ndim, dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) result->storage->data[i] = (float)(i + 1);
    return result;
}

TEST(test_reshape_backward_restores_input_shape) {
    int input_dims[2] = {2, 3}, output_dims[2] = {3, 2};
    float values[6] = {1,2,3,4,5,6};
    ag_tensor* input = make_ag(2, input_dims, values);
    ag_tensor* output = ag_reshape(input, 2, output_dims);
    tensor* upstream = sequential_gradient(2, output_dims);
    tensor* gradients[1] = {NULL};
    ASSERT_NOT_NULL(output);
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    ASSERT_TRUE(same_shape(gradients[0], input->value));
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(gradients[0]->storage->data[i], (float)(i + 1));
    t_free(gradients[0]); t_free(upstream); ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_transpose_backward_applies_inverse_transpose) {
    int dims[2] = {2, 3};
    float values[6] = {1,2,3,4,5,6};
    ag_tensor* input = make_ag(2, dims, values);
    ag_tensor* output = ag_transpose(input, 0, 1);
    tensor* upstream = sequential_gradient(output->value->ndim, output->value->dims);
    tensor* gradients[1] = {NULL};
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    ASSERT_EQ_INT(gradients[0]->dims[0], 2);
    ASSERT_EQ_INT(gradients[0]->dims[1], 3);
    int coords[2] = {1, 2};
    ASSERT_EQ_FLOAT(gradients[0]->storage->data[get_flat_index_nd(gradients[0], coords)], 6.0f);
    t_free(gradients[0]); t_free(upstream); ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_slice_backward_scatters_and_zero_fills) {
    int dims[2] = {3, 4};
    float values[12] = {0};
    ag_tensor* input = make_ag(2, dims, values);
    ag_tensor* output = ag_slice(input, 0, 1, 3);
    tensor* upstream = sequential_gradient(output->value->ndim, output->value->dims);
    tensor* gradients[1] = {NULL};
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    for (int i = 0; i < 4; ++i) ASSERT_EQ_FLOAT(gradients[0]->storage->data[i], 0.0f);
    for (int i = 0; i < 8; ++i) ASSERT_EQ_FLOAT(gradients[0]->storage->data[i + 4], (float)(i + 1));
    t_free(gradients[0]); t_free(upstream); ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_slice_backward_copies_contiguous_middle_axis_blocks) {
    int dims[3] = {2, 4, 3};
    float values[24] = {0};
    ag_tensor* input = make_ag(3, dims, values);
    ag_tensor* output = ag_slice(input, 1, 1, 3);
    tensor* upstream = sequential_gradient(output->value->ndim,
                                           output->value->dims);
    tensor* gradients[1] = {NULL};
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream,
                                            gradients), 0);
    for (int batch = 0; batch < 2; ++batch) {
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 3; ++column) {
                int index = (batch * 4 + row) * 3 + column;
                float expected = row < 1 || row >= 3 ? 0.0f :
                    (float)((batch * 2 + row - 1) * 3 + column + 1);
                ASSERT_EQ_FLOAT(gradients[0]->storage->data[index], expected);
            }
        }
    }
    t_free(gradients[0]); t_free(upstream);
    ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_expand_backward_emits_output_shaped_contribution) {
    int input_dims[2] = {1, 3}, output_dims[3] = {2, 4, 3};
    float values[3] = {1,2,3};
    ag_tensor* input = make_ag(2, input_dims, values);
    ag_tensor* output = ag_expand(input, 3, output_dims);
    tensor* upstream = sequential_gradient(3, output_dims);
    tensor* gradients[1] = {NULL};
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    ASSERT_TRUE(same_shape(gradients[0], output->value));
    ASSERT_EQ_FLOAT(gradients[0]->storage->data[23], 24.0f);
    t_free(gradients[0]); t_free(upstream); ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_view_wrappers_reject_invalid_arguments) {
    int dims[2] = {2, 3}; float values[6] = {0}; int bad[2] = {4, 2};
    ag_tensor* input = make_ag(2, dims, values);
    ASSERT_NULL(ag_reshape(NULL, 2, dims));
    ASSERT_NULL(ag_reshape(input, 2, bad));
    ASSERT_NULL(ag_transpose(input, 0, 2));
    ASSERT_NULL(ag_slice(input, 0, 2, 2));
    ASSERT_NULL(ag_expand(input, 1, dims));
    ag_tensor_release(input);
}

int main(void) {
    printf("== autograd_view.c ==\n");
    RUN_TEST(test_reshape_backward_restores_input_shape);
    RUN_TEST(test_transpose_backward_applies_inverse_transpose);
    RUN_TEST(test_slice_backward_scatters_and_zero_fills);
    RUN_TEST(test_slice_backward_copies_contiguous_middle_axis_blocks);
    RUN_TEST(test_expand_backward_emits_output_shaped_contribution);
    RUN_TEST(test_view_wrappers_reject_invalid_arguments);
    TEST_SUITE_SUMMARY();
}
