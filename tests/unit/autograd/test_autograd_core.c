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

int main(void) {
    printf("== autograd_core.c ==\n");
    RUN_TEST(test_owned_tensor_initializes_normalized_leaf);
    RUN_TEST(test_owned_tensor_rejects_null);
    RUN_TEST(test_tensor_retain_keeps_value_alive);
    RUN_TEST(test_output_release_destroys_creator_context_and_retained_input);
    RUN_TEST(test_lifetime_functions_accept_null);
    TEST_SUITE_SUMMARY();
}
