#include "./../../../include/tensorlib/autograd.h"
#include "./../../fixtures/test_common.h"
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

TEST(test_unary_forwards_and_local_gradients) {
    int dims[1] = {3};
    float values[3] = {0.5f, 1.0f, 2.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* results[3] = {ag_neg(input), ag_exp(input), ag_log(input)};
    ag_op operations[3] = {AG_OP_NEG, AG_OP_EXP, AG_OP_LOG};
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 3; ++i) upstream->storage->data[i] = 2.0f;

    for (int op = 0; op < 3; ++op) {
        tensor* gradients[1] = {NULL};
        ASSERT_NOT_NULL(results[op]);
        ASSERT_EQ_INT(results[op]->creator->operation, operations[op]);
        ASSERT_EQ_INT(results[op]->creator->backward(results[op]->creator, upstream, gradients), 0);
        ASSERT_NOT_NULL(gradients[0]);
        for (int i = 0; i < 3; ++i) {
            float expected = op == 0 ? -2.0f
                           : op == 1 ? 2.0f * expf(values[i])
                                     : 2.0f / values[i];
            ASSERT_FLOAT_NEAR(gradients[0]->storage->data[i], expected, 1e-5f);
        }
        t_free(gradients[0]);
    }
    for (int op = 0; op < 3; ++op) ag_tensor_release(results[op]);
    t_free(upstream); ag_tensor_release(input);
}

TEST(test_unary_handles_scalar_view_and_null) {
    float scalar = 1.5f;
    ag_tensor* input = make_ag(0, NULL, &scalar, 0);
    ag_tensor* result = ag_exp(input);
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->value->ndim, 0);
    ASSERT_NULL(result->creator);
    ASSERT_FLOAT_NEAR(result->value->storage->data[0], expf(scalar), 1e-6f);
    ASSERT_NULL(ag_neg(NULL));
    ASSERT_NULL(ag_exp(NULL));
    ASSERT_NULL(ag_log(NULL));
    ag_tensor_release(result); ag_tensor_release(input);
}

TEST(test_pow_forward_backward_and_edge_exponents) {
    int dims[1] = {3};
    float values[3] = {1.0f, 2.0f, 3.0f};
    float upstream_values[3] = {1.0f, 2.0f, 0.5f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* cube = ag_pow(input, 3.0f);
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 3; ++i) upstream->storage->data[i] = upstream_values[i];
    ASSERT_NOT_NULL(cube);
    ASSERT_EQ_INT(cube->creator->operation, AG_OP_POW);
    ASSERT_EQ_INT(ag_backward_with_grad(cube, upstream), 0);
    for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_NEAR(cube->value->storage->data[i], values[i] * values[i] * values[i], 1e-6f);
        ASSERT_FLOAT_NEAR(input->grad->storage->data[i],
                          upstream_values[i] * 3.0f * values[i] * values[i], 1e-5f);
    }
    ag_zero_grad(input);

    ag_tensor* zero_power = ag_pow(input, 0.0f);
    ag_tensor* identity = ag_pow(input, 1.0f);
    ASSERT_EQ_INT(ag_backward_with_grad(zero_power, upstream), 0);
    for (int i = 0; i < 3; ++i) ASSERT_EQ_FLOAT(input->grad->storage->data[i], 0.0f);
    ag_zero_grad(input);
    ASSERT_EQ_INT(ag_backward_with_grad(identity, upstream), 0);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ_FLOAT(input->grad->storage->data[i], upstream_values[i]);
    }

    ag_tensor_release(identity);
    ag_tensor_release(zero_power);
    t_free(upstream);
    ag_tensor_release(cube);
    ag_tensor_release(input);
    ASSERT_NULL(ag_pow(NULL, 2.0f));
}

TEST(test_pow_scalar_and_view_gradient_matches_central_difference) {
    int dims[2] = {2, 2};
    float values[4] = {0.7f, 1.1f, 1.5f, 1.9f};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 4; ++i) base->storage->data[i] = values[i];
    tensor* view = t_transpose(base, 0, 1);
    t_free(base);
    ag_tensor* input = ag_from_owned_tensor(view, 1);
    ag_tensor* powered = ag_pow(input, 2.5f);
    ag_tensor* row = ag_sum(powered, 1, 0);
    ag_tensor* loss = ag_sum(row, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int i = 0; i < 4; ++i) {
        float x = input->value->storage->data[i];
        float numerical = (powf(x + epsilon, 2.5f) - powf(x - epsilon, 2.5f)) /
                          (2.0f * epsilon);
        int logical_index = i == 1 ? 2 : i == 2 ? 1 : i;
        ASSERT_FLOAT_NEAR(input->grad->storage->data[logical_index], numerical, 5e-4f);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row);
    ag_tensor_release(powered);
    ag_tensor_release(input);
}

int main(void) {
    printf("== autograd_ops.c ==\n");
    RUN_TEST(test_binary_forwards_create_typed_nodes_and_retain_inputs);
    RUN_TEST(test_binary_without_grad_omits_graph);
    RUN_TEST(test_binary_local_gradients_match_derivatives);
    RUN_TEST(test_broadcast_forward_and_local_gradient_use_output_shape);
    RUN_TEST(test_binary_rejects_null_and_incompatible_shapes);
    RUN_TEST(test_unary_forwards_and_local_gradients);
    RUN_TEST(test_unary_handles_scalar_view_and_null);
    RUN_TEST(test_pow_forward_backward_and_edge_exponents);
    RUN_TEST(test_pow_scalar_and_view_gradient_matches_central_difference);
    TEST_SUITE_SUMMARY();
}
