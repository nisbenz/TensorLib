#include "autograd.h"
#include "test_common.h"

static ag_tensor* make_ag(int ndim, const int* dims, const float* values, int requires_grad) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static void free_local_gradients(tensor** gradients, int count) {
    for (int i = 0; i < count; ++i) t_free(gradients[i]);
}

TEST(test_binary_forwards_create_typed_nodes_and_retain_inputs) {
    int dims[1] = {2};
    float av[2] = {6.0f, 8.0f};
    float bv[2] = {2.0f, 4.0f};
    ag_tensor* a = make_ag(1, dims, av, 1);
    ag_tensor* b = make_ag(1, dims, bv, 0);
    ag_tensor* results[4] = {ag_add(a, b), ag_sub(a, b), ag_mul(a, b), ag_div(a, b)};
    ag_op operations[4] = {AG_OP_ADD, AG_OP_SUB, AG_OP_MUL, AG_OP_DIV};
    float expected[4][2] = {{8, 12}, {4, 4}, {12, 32}, {3, 2}};

    for (int op = 0; op < 4; ++op) {
        ASSERT_NOT_NULL(results[op]);
        ASSERT_TRUE(results[op]->requires_grad);
        ASSERT_NOT_NULL(results[op]->creator);
        ASSERT_EQ_INT(results[op]->creator->operation, operations[op]);
        ASSERT_EQ_INT(results[op]->creator->input_count, 2);
        ASSERT_EQ_FLOAT(results[op]->value->storage->data[0], expected[op][0]);
        ASSERT_EQ_FLOAT(results[op]->value->storage->data[1], expected[op][1]);
    }
    ASSERT_EQ_INT(a->ref_count, 5);
    ASSERT_EQ_INT(b->ref_count, 5);
    for (int op = 0; op < 4; ++op) ag_tensor_release(results[op]);
    ag_tensor_release(b);
    ag_tensor_release(a);
}

TEST(test_binary_without_grad_omits_graph) {
    float av = 2.0f, bv = 3.0f;
    ag_tensor* a = make_ag(0, NULL, &av, 0);
    ag_tensor* b = make_ag(0, NULL, &bv, 0);
    ag_tensor* result = ag_add(a, b);
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->requires_grad, 0);
    ASSERT_NULL(result->creator);
    ASSERT_EQ_INT(a->ref_count, 1);
    ASSERT_EQ_INT(b->ref_count, 1);
    ag_tensor_release(result); ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_binary_local_gradients_match_derivatives) {
    int dims[1] = {2};
    float av[2] = {6.0f, 8.0f}, bv[2] = {2.0f, 4.0f}, gv[2] = {3.0f, 5.0f};
    ag_tensor* a = make_ag(1, dims, av, 1);
    ag_tensor* b = make_ag(1, dims, bv, 1);
    tensor* upstream = t_alloc(1, dims);
    upstream->storage->data[0] = gv[0]; upstream->storage->data[1] = gv[1];
    ag_tensor* results[4] = {ag_add(a, b), ag_sub(a, b), ag_mul(a, b), ag_div(a, b)};
    float expected_a[4][2] = {{3,5}, {3,5}, {6,20}, {1.5f,1.25f}};
    float expected_b[4][2] = {{3,5}, {-3,-5}, {18,40}, {-4.5f,-2.5f}};

    for (int op = 0; op < 4; ++op) {
        tensor* gradients[2] = {NULL, NULL};
        ASSERT_EQ_INT(results[op]->creator->backward(results[op]->creator, upstream, gradients), 0);
        ASSERT_NOT_NULL(gradients[0]); ASSERT_NOT_NULL(gradients[1]);
        for (int i = 0; i < 2; ++i) {
            ASSERT_FLOAT_NEAR(gradients[0]->storage->data[i], expected_a[op][i], 1e-6f);
            ASSERT_FLOAT_NEAR(gradients[1]->storage->data[i], expected_b[op][i], 1e-6f);
        }
        free_local_gradients(gradients, 2);
    }
    for (int op = 0; op < 4; ++op) ag_tensor_release(results[op]);
    t_free(upstream); ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_broadcast_forward_and_local_gradient_use_output_shape) {
    int adims[2] = {2, 3}, bdims[1] = {3};
    float av[6] = {1,2,3,4,5,6}, bv[3] = {10,20,30};
    ag_tensor* a = make_ag(2, adims, av, 1);
    ag_tensor* b = make_ag(1, bdims, bv, 1);
    ag_tensor* result = ag_mul(a, b);
    tensor* upstream = t_alloc(result->value->ndim, result->value->dims);
    for (int i = 0; i < tensor_numel(upstream); ++i) upstream->storage->data[i] = 1.0f;
    tensor* gradients[2] = {NULL, NULL};
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->creator->backward(result->creator, upstream, gradients), 0);
    ASSERT_EQ_INT(gradients[1]->ndim, 2);
    ASSERT_EQ_INT(gradients[1]->dims[0], 2);
    ASSERT_EQ_INT(gradients[1]->dims[1], 3);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[5], 6.0f);
    free_local_gradients(gradients, 2); t_free(upstream);
    ag_tensor_release(result); ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_binary_rejects_null_and_incompatible_shapes) {
    int adims[1] = {2}, bdims[1] = {3};
    float av[2] = {1,2}, bv[3] = {1,2,3};
    ag_tensor* a = make_ag(1, adims, av, 1);
    ag_tensor* b = make_ag(1, bdims, bv, 1);
    ASSERT_NULL(ag_add(NULL, b));
    ASSERT_NULL(ag_mul(a, NULL));
    ASSERT_NULL(ag_sub(a, b));
    ag_tensor_release(b); ag_tensor_release(a);
}

int main(void) {
    printf("== autograd_ops.c ==\n");
    RUN_TEST(test_binary_forwards_create_typed_nodes_and_retain_inputs);
    RUN_TEST(test_binary_without_grad_omits_graph);
    RUN_TEST(test_binary_local_gradients_match_derivatives);
    RUN_TEST(test_broadcast_forward_and_local_gradient_use_output_shape);
    RUN_TEST(test_binary_rejects_null_and_incompatible_shapes);
    TEST_SUITE_SUMMARY();
}
