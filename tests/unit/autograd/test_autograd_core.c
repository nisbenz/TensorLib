#include "autograd.h"
#include "test_common.h"

static tensor* make_tensor(int ndim, const int* dims, const float* values) {
    tensor* result = t_alloc(ndim, dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) {
        result->storage->data[i] = values[i];
    }
    return result;
}

TEST(test_ag_from_owned_tensor_initializes_leaf) {
    int dims[2] = {2, 2};
    float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor* raw = make_tensor(2, dims, values);

    ag_tensor* value = ag_from_owned_tensor(raw, 7);

    ASSERT_NOT_NULL(value);
    ASSERT_TRUE(value->value == raw);
    ASSERT_NULL(value->grad);
    ASSERT_NULL(value->creator);
    ASSERT_EQ_INT(value->requires_grad, 1);
    ASSERT_EQ_INT(value->ref_count, 1);
    ag_tensor_release(value);
}

TEST(test_ag_from_owned_tensor_rejects_null) {
    ASSERT_NULL(ag_from_owned_tensor(NULL, 1));
}

TEST(test_ag_tensor_retain_and_release_manage_shared_owner) {
    float scalar = 3.0f;
    ag_tensor* value = ag_from_owned_tensor(make_tensor(0, NULL, &scalar), 0);
    ASSERT_NOT_NULL(value);

    ag_tensor_retain(value);
    ASSERT_EQ_INT(value->ref_count, 2);
    ag_tensor_release(value);
    ASSERT_EQ_INT(value->ref_count, 1);
    ASSERT_EQ_FLOAT(value->value->storage->data[0], 3.0f);
    ag_tensor_release(value);
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
    RUN_TEST(test_ag_from_owned_tensor_initializes_leaf);
    RUN_TEST(test_ag_from_owned_tensor_rejects_null);
    RUN_TEST(test_ag_tensor_retain_and_release_manage_shared_owner);
    RUN_TEST(test_lifetime_functions_accept_null);
    TEST_SUITE_SUMMARY();
}
