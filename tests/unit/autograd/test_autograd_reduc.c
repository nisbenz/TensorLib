#include <math.h>

#include "autograd.h"
#include "test_common.h"

static ag_tensor* make_ag(int ndim, const int* dims, const float* values) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, 1);
}

static tensor* full(int ndim, const int* dims, float value) {
    tensor* result = t_alloc(ndim, dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) result->storage->data[i] = value;
    return result;
}

TEST(test_sum_and_mean_backward_expand_reduced_axis) {
    int dims[2] = {2, 3}; float values[6] = {1,2,3,4,5,6};
    ag_tensor* input = make_ag(2, dims, values);
    ag_tensor* sum = ag_sum(input, 1, 0);
    ag_tensor* mean = ag_mean(input, 0, 1);
    tensor* sum_upstream = full(sum->value->ndim, sum->value->dims, 2.0f);
    tensor* mean_upstream = full(mean->value->ndim, mean->value->dims, 3.0f);
    tensor* sum_grad[1] = {NULL}, *mean_grad[1] = {NULL};

    ASSERT_EQ_INT(sum->creator->backward(sum->creator, sum_upstream, sum_grad), 0);
    ASSERT_EQ_INT(mean->creator->backward(mean->creator, mean_upstream, mean_grad), 0);
    ASSERT_TRUE(same_shape(sum_grad[0], input->value));
    ASSERT_TRUE(same_shape(mean_grad[0], input->value));
    for (int i = 0; i < 6; ++i) {
        int coords[2] = {i / 3, i % 3};
        ASSERT_EQ_FLOAT(sum_grad[0]->storage->data[get_flat_index_nd(sum_grad[0], coords)], 2.0f);
        ASSERT_EQ_FLOAT(mean_grad[0]->storage->data[get_flat_index_nd(mean_grad[0], coords)], 1.5f);
    }
    t_free(mean_grad[0]); t_free(sum_grad[0]); t_free(mean_upstream); t_free(sum_upstream);
    ag_tensor_release(mean); ag_tensor_release(sum); ag_tensor_release(input);
}

TEST(test_max_backward_splits_ties_equally) {
    int dims[2] = {2, 3}; float values[6] = {4,4,1,2,5,5};
    ag_tensor* input = make_ag(2, dims, values);
    ag_tensor* output = ag_max(input, 1, 0);
    float upstream_values[2] = {6,8};
    tensor* upstream = full(1, output->value->dims, 0.0f);
    upstream->storage->data[0] = upstream_values[0]; upstream->storage->data[1] = upstream_values[1];
    tensor* gradients[1] = {NULL};
    float expected[6] = {3,3,0,0,4,4};
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(gradients[0]->storage->data[i], expected[i]);
    t_free(gradients[0]); t_free(upstream); ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_max_nan_slice_produces_nan_gradients) {
    int dims[2] = {1, 3}; float values[3] = {1.0f, NAN, 2.0f};
    ag_tensor* input = make_ag(2, dims, values);
    ag_tensor* output = ag_max(input, 1, 1);
    tensor* upstream = full(output->value->ndim, output->value->dims, 1.0f);
    tensor* gradients[1] = {NULL};
    ASSERT_NAN(output->value->storage->data[0]);
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    for (int i = 0; i < 3; ++i) ASSERT_NAN(gradients[0]->storage->data[i]);
    t_free(gradients[0]); t_free(upstream); ag_tensor_release(output); ag_tensor_release(input);
}

TEST(test_reductions_reject_scalar_and_invalid_axis) {
    float scalar = 1.0f;
    ag_tensor* input = make_ag(0, NULL, &scalar);
    ASSERT_NULL(ag_sum(NULL, 0, 0));
    ASSERT_NULL(ag_sum(input, 0, 0));
    ASSERT_NULL(ag_mean(input, -1, 1));
    ASSERT_NULL(ag_max(input, 2, 0));
    ag_tensor_release(input);
}

int main(void) {
    printf("== autograd_reduc.c ==\n");
    RUN_TEST(test_sum_and_mean_backward_expand_reduced_axis);
    RUN_TEST(test_max_backward_splits_ties_equally);
    RUN_TEST(test_max_nan_slice_produces_nan_gradients);
    RUN_TEST(test_reductions_reject_scalar_and_invalid_axis);
    TEST_SUITE_SUMMARY();
}
