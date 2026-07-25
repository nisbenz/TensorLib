#include <stdlib.h>
#include "./../../../include/tensorlib/autograd.h"
#include "./../../fixtures/test_common.h"

static int contexts_freed = 0;

static tensor* make_scalar(float value) {
    tensor* result = t_alloc(0, NULL);
    if (result != NULL) result->storage->data[0] = value;
    return result;
}

static void count_context_free(void* context) {
    contexts_freed++;
    free(context);
}

TEST(test_owned_tensor_initializes_normalized_leaf) {
    tensor* raw = make_scalar(4.0f);
    ag_tensor* value = ag_from_owned_tensor(raw, -3);

    ASSERT_NOT_NULL(value);
    ASSERT_TRUE(value->value == raw);
    ASSERT_NULL(value->grad);
    ASSERT_NULL(value->creator);
    ASSERT_EQ_INT(value->requires_grad, 1);
    ASSERT_EQ_INT(value->ref_count, 1);
    ag_tensor_release(value);
}

TEST(test_owned_tensor_rejects_null) {
    ASSERT_NULL(ag_from_owned_tensor(NULL, 1));
}

TEST(test_tensor_retain_keeps_value_alive) {
    ag_tensor* value = ag_from_owned_tensor(make_scalar(3.0f), 0);
    ASSERT_NOT_NULL(value);

    ag_tensor_retain(value);
    ASSERT_EQ_INT(value->ref_count, 2);
    ag_tensor_release(value);
    ASSERT_EQ_INT(value->ref_count, 1);
    ASSERT_EQ_FLOAT(value->value->storage->data[0], 3.0f);
    ag_tensor_release(value);
}

TEST(test_output_release_destroys_creator_context_and_retained_input) {
    contexts_freed = 0;
    ag_tensor* input = ag_from_owned_tensor(make_scalar(2.0f), 1);
    ag_tensor* output = ag_from_owned_tensor(make_scalar(4.0f), 1);
    ag_node* node = (ag_node*)calloc(1, sizeof(*node));
    ASSERT_NOT_NULL(input);
    ASSERT_NOT_NULL(output);
    ASSERT_NOT_NULL(node);

    node->inputs = (ag_tensor**)malloc(sizeof(*node->inputs));
    node->context = malloc(1);
    ASSERT_NOT_NULL(node->inputs);
    ASSERT_NOT_NULL(node->context);
    node->input_count = 1;
    node->inputs[0] = input;
    node->output = output;
    node->free_context = count_context_free;
    node->ref_count = 1;
    ag_tensor_retain(input);
    output->creator = node;

    ag_tensor_release(input);
    ASSERT_EQ_INT(node->inputs[0]->ref_count, 1);
    ag_tensor_release(output);
    ASSERT_EQ_INT(contexts_freed, 1);
}

TEST(test_lifetime_functions_accept_null) {
    ag_tensor_retain(NULL);
    ag_tensor_release(NULL);
    ag_node_retain(NULL);
    ag_node_release(NULL);
    ASSERT_TRUE(1);
}

TEST(test_detach_shares_exact_view_metadata_and_outlives_source) {
    int dims[2] = {2, 3};
    tensor* raw = t_alloc(2, dims);
    for (int i = 0; i < 6; ++i) raw->storage->data[i] = (float)(i + 1);
    ag_tensor* base = ag_from_owned_tensor(raw, 1);
    ag_tensor* view = ag_transpose(base, 0, 1);
    ag_tensor* detached = ag_detach(view);
    ASSERT_NOT_NULL(base);
    ASSERT_NOT_NULL(view);
    ASSERT_NOT_NULL(detached);
    ASSERT_TRUE(detached->value->storage == view->value->storage);
    ASSERT_EQ_INT(detached->value->ndim, view->value->ndim);
    ASSERT_EQ_INT(detached->value->offset, view->value->offset);
    ASSERT_TRUE(same_shape(detached->value, view->value));
    ASSERT_TRUE(same_stride(detached->value, view->value));
    ASSERT_EQ_INT(detached->requires_grad, 0);
    ASSERT_NULL(detached->grad);
    ASSERT_NULL(detached->creator);

    ag_tensor_release(view);
    ag_tensor_release(base);
    ASSERT_EQ_FLOAT(detached->value->storage->data[0], 1.0f);
    tensor_mark_modified(detached->value);
    ASSERT_EQ_INT(detached->value->storage->version, 1);
    ag_tensor_release(detached);
}

TEST(test_detach_stops_gradient_tracking_and_detects_alias_mutation) {
    float value = 2.0f;
    ag_tensor* input = ag_from_owned_tensor(make_scalar(value), 1);
    ag_tensor* exponent = ag_exp(input);
    ag_tensor* detached = ag_detach(input);
    ag_tensor* product = ag_mul(exponent, detached);
    ASSERT_NOT_NULL(product);
    ASSERT_EQ_INT(product->requires_grad, 1);
    ASSERT_EQ_INT(ag_backward(product), 0);
    ASSERT_FLOAT_NEAR(input->grad->storage->data[0], expf(value) * value, 1e-5f);
    ASSERT_NULL(detached->grad);

    detached->value->storage->data[0] = 3.0f;
    tensor_mark_modified(detached->value);
    ASSERT_EQ_INT(ag_backward(product), 1);

    ag_tensor_release(product);
    ag_tensor_release(detached);
    ag_tensor_release(exponent);
    ag_tensor_release(input);
    ASSERT_NULL(ag_detach(NULL));
}

int main(void) {
    printf("== autograd_core.c ==\n");
    RUN_TEST(test_owned_tensor_initializes_normalized_leaf);
    RUN_TEST(test_owned_tensor_rejects_null);
    RUN_TEST(test_tensor_retain_keeps_value_alive);
    RUN_TEST(test_output_release_destroys_creator_context_and_retained_input);
    RUN_TEST(test_lifetime_functions_accept_null);
    RUN_TEST(test_detach_shares_exact_view_metadata_and_outlives_source);
    RUN_TEST(test_detach_stops_gradient_tracking_and_detects_alias_mutation);
    TEST_SUITE_SUMMARY();
}
