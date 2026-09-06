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

TEST(test_add_sub_backward_reduce_broadcast_at_source) {
    int matrix_dims[2] = {2, 3};
    int bias_dims[1] = {3};
    int owner_dims[2] = {3, 2};
    float matrix_values[6] = {1, 2, 3, 4, 5, 6};
    float bias_values[3] = {0.5f, 1.0f, 1.5f};
    ag_tensor* matrix = make_ag(2, matrix_dims, matrix_values, 1);
    ag_tensor* bias = make_ag(1, bias_dims, bias_values, 1);
    tensor* owner = t_alloc(2, owner_dims);
    for (int index = 0; index < 6; ++index) {
        owner->storage->data[index] = (float)(index + 1);
    }
    tensor* upstream = t_transpose(owner, 0, 1);
    tensor* gradients[2] = {NULL, NULL};
    ag_tensor* added = ag_add(matrix, bias);
    ASSERT_EQ_INT(added->creator->backward(
        added->creator, upstream, gradients), 0);
    ASSERT_EQ_INT(gradients[1]->ndim, 1);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[0], 3.0f);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[1], 7.0f);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[2], 11.0f);
    free_local_gradients(gradients, 2);
    ag_tensor* subtracted = ag_sub(matrix, bias);
    ASSERT_EQ_INT(subtracted->creator->backward(
        subtracted->creator, upstream, gradients), 0);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[0], -3.0f);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[1], -7.0f);
    ASSERT_EQ_FLOAT(gradients[1]->storage->data[2], -11.0f);
    free_local_gradients(gradients, 2);
    ag_tensor_release(subtracted); ag_tensor_release(added);
    t_free(upstream); t_free(owner);
    ag_tensor_release(bias); ag_tensor_release(matrix);
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

TEST(test_scalar_arithmetic_forward_backward_and_lifecycle) {
    int dims[1] = {3};
    float values[3] = {2.0f, -4.0f, 0.5f};
    float upstream_values[3] = {1.0f, 2.0f, -3.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    tensor* upstream = t_alloc(1, dims);
    ag_tensor* multiplied;
    ag_tensor* divided;
    tensor* gradients[1] = {NULL};

    for (int i = 0; i < 3; ++i) {
        upstream->storage->data[i] = upstream_values[i];
    }
    multiplied = ag_mul_scalar(input, -2.0f);
    divided = ag_div_scalar(input, 4.0f);
    ASSERT_NOT_NULL(multiplied);
    ASSERT_NOT_NULL(divided);
    ASSERT_EQ_INT(multiplied->creator->operation, AG_OP_MUL_SCALAR);
    ASSERT_EQ_INT(divided->creator->operation, AG_OP_DIV_SCALAR);
    ASSERT_EQ_INT(multiplied->creator->input_count, 1);
    ASSERT_EQ_INT(input->ref_count, 3);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ_FLOAT(multiplied->value->storage->data[i], values[i] * -2.0f);
        ASSERT_EQ_FLOAT(divided->value->storage->data[i], values[i] / 4.0f);
    }
    ASSERT_EQ_INT(multiplied->creator->backward(
        multiplied->creator, upstream, gradients), 0);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ_FLOAT(gradients[0]->storage->data[i],
                        upstream_values[i] * -2.0f);
    }
    t_free(gradients[0]);
    gradients[0] = NULL;
    ASSERT_EQ_INT(divided->creator->backward(
        divided->creator, upstream, gradients), 0);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ_FLOAT(gradients[0]->storage->data[i],
                        upstream_values[i] / 4.0f);
    }
    t_free(gradients[0]);
    ag_tensor_release(divided);
    ag_tensor_release(multiplied);
    ASSERT_EQ_INT(input->ref_count, 1);
    t_free(upstream);
    ag_tensor_release(input);
}

TEST(test_scalar_arithmetic_views_ieee_and_finite_difference) {
    int dims[2] = {2, 2};
    float values[4] = {1.5f, -2.0f, 3.0f, 4.0f};
    tensor* raw = t_alloc(2, dims);
    ag_tensor* input;
    ag_tensor* multiplied;
    ag_tensor* divided;
    ag_tensor* row_sum;
    ag_tensor* loss;
    const float epsilon = 1e-3f;

    for (int i = 0; i < 4; ++i) raw->storage->data[i] = values[i];
    input = ag_from_owned_tensor(t_transpose(raw, 0, 1), 1);
    t_free(raw);
    multiplied = ag_mul_scalar(input, 1.75f);
    divided = ag_div_scalar(multiplied, 2.5f);
    row_sum = ag_sum(divided, 1, 0);
    loss = ag_sum(row_sum, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    for (int i = 0; i < 4; ++i) {
        float x = values[i];
        float numerical = (((x + epsilon) * 1.75f / 2.5f) -
                           ((x - epsilon) * 1.75f / 2.5f)) /
                          (2.0f * epsilon);
        ASSERT_FLOAT_NEAR(input->grad->storage->data[i], numerical, 1e-4f);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row_sum);
    ag_tensor_release(divided);
    ag_tensor_release(multiplied);
    ag_tensor_release(input);

    {
        float scalar_value = 0.0f;
        ag_tensor* scalar = make_ag(0, NULL, &scalar_value, 0);
        ag_tensor* untracked = ag_mul_scalar(scalar, 0.0f);
        ag_tensor* ieee = ag_div_scalar(scalar, 0.0f);
        ASSERT_NOT_NULL(untracked);
        ASSERT_NULL(untracked->creator);
        ASSERT_NAN(ieee->value->storage->data[0]);
        ASSERT_NULL(ag_mul_scalar(NULL, 2.0f));
        ASSERT_NULL(ag_div_scalar(NULL, 2.0f));
        ag_tensor_release(ieee);
        ag_tensor_release(untracked);
        ag_tensor_release(scalar);
    }
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

TEST(test_sqrt_forward_backward_preserves_ieee_domains) {
    int dims[1] = {4};
    float values[4] = {0.0f, 4.0f, 9.0f, -1.0f};
    float upstream_values[4] = {1.0f, 2.0f, 3.0f, 1.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* result = ag_sqrt(input);
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 4; ++i) upstream->storage->data[i] = upstream_values[i];
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->creator->operation, AG_OP_SQRT);
    ASSERT_EQ_INT(ag_backward_with_grad(result, upstream), 0);
    ASSERT_POS_INF(input->grad->storage->data[0]);
    ASSERT_FLOAT_NEAR(input->grad->storage->data[1], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(input->grad->storage->data[2], 0.5f, 1e-6f);
    ASSERT_NAN(input->grad->storage->data[3]);
    ASSERT_NAN(result->value->storage->data[3]);
    t_free(upstream);
    ag_tensor_release(result);
    ag_tensor_release(input);
    ASSERT_NULL(ag_sqrt(NULL));
}

TEST(test_sqrt_scalar_gradient_matches_central_difference) {
    float value = 2.25f;
    ag_tensor* input = make_ag(0, NULL, &value, 1);
    ag_tensor* result = ag_sqrt(input);
    ASSERT_EQ_INT(ag_backward(result), 0);
    const float epsilon = 1e-3f;
    float numerical = (sqrtf(value + epsilon) - sqrtf(value - epsilon)) /
                      (2.0f * epsilon);
    ASSERT_FLOAT_NEAR(input->grad->storage->data[0], numerical, 5e-5f);
    ag_tensor_release(result);
    ag_tensor_release(input);
}

TEST(test_relu_backward_defines_zero_and_nan_behavior) {
    int dims[1] = {4};
    float values[4] = {-2.0f, 0.0f, 3.0f, NAN};
    float upstream_values[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* result = ag_relu(input);
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 4; ++i) upstream->storage->data[i] = upstream_values[i];
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->creator->operation, AG_OP_RELU);
    ASSERT_EQ_INT(ag_backward_with_grad(result, upstream), 0);
    ASSERT_EQ_FLOAT(input->grad->storage->data[0], 0.0f);
    ASSERT_EQ_FLOAT(input->grad->storage->data[1], 0.0f);
    ASSERT_EQ_FLOAT(input->grad->storage->data[2], 2.0f);
    ASSERT_NAN(input->grad->storage->data[3]);
    t_free(upstream);
    ag_tensor_release(result);
    ag_tensor_release(input);
    ASSERT_NULL(ag_relu(NULL));
}

TEST(test_relu_strided_and_scalar_gradients) {
    int dims[2] = {2, 2};
    float values[4] = {-1.0f, 2.0f, 3.0f, -4.0f};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 4; ++i) base->storage->data[i] = values[i];
    tensor* view = t_transpose(base, 0, 1);
    t_free(base);
    ag_tensor* input = ag_from_owned_tensor(view, 1);
    ag_tensor* result = ag_relu(input);
    ag_tensor* row = ag_sum(result, 1, 0);
    ag_tensor* loss = ag_sum(row, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    float expected[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ_FLOAT(input->grad->storage->data[i], expected[i]);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row);
    ag_tensor_release(result);
    ag_tensor_release(input);

    float scalar = 0.7f;
    ag_tensor* scalar_input = make_ag(0, NULL, &scalar, 1);
    ag_tensor* scalar_result = ag_relu(scalar_input);
    ASSERT_EQ_INT(ag_backward(scalar_result), 0);
    const float epsilon = 1e-3f;
    float numerical = (fmaxf(scalar + epsilon, 0.0f) -
                       fmaxf(scalar - epsilon, 0.0f)) / (2.0f * epsilon);
    ASSERT_FLOAT_NEAR(scalar_input->grad->storage->data[0], numerical, 2e-5f);
    ag_tensor_release(scalar_result);
    ag_tensor_release(scalar_input);
}

TEST(test_sigmoid_seeded_backward_and_saturation) {
    int dims[1] = {4};
    float values[4] = {-20.0f, -2.0f, 0.0f, 20.0f};
    float upstream_values[4] = {3.0f, 2.0f, 1.0f, 4.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* result = ag_sigmoid(input);
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 4; ++i) upstream->storage->data[i] = upstream_values[i];
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->creator->operation, AG_OP_SIGMOID);
    ASSERT_EQ_INT(ag_backward_with_grad(result, upstream), 0);
    for (int i = 0; i < 4; ++i) {
        float y = 1.0f / (1.0f + expf(-values[i]));
        ASSERT_FLOAT_NEAR(result->value->storage->data[i], y, 1e-6f);
        ASSERT_FLOAT_NEAR(input->grad->storage->data[i],
                          upstream_values[i] * y * (1.0f - y), 1e-6f);
    }
    t_free(upstream);
    ag_tensor_release(result);
    ag_tensor_release(input);
    ASSERT_NULL(ag_sigmoid(NULL));
}

TEST(test_sigmoid_view_gradient_matches_central_difference) {
    int dims[2] = {2, 2};
    float values[4] = {-3.0f, 0.5f, 2.0f, 4.0f};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 4; ++i) base->storage->data[i] = values[i];
    tensor* view = t_transpose(base, 0, 1);
    t_free(base);
    ag_tensor* input = ag_from_owned_tensor(view, 1);
    ag_tensor* result = ag_sigmoid(input);
    ag_tensor* row = ag_sum(result, 1, 0);
    ag_tensor* loss = ag_sum(row, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int logical = 0; logical < 4; ++logical) {
        int storage_index = logical == 1 ? 2 : logical == 2 ? 1 : logical;
        float x = values[storage_index];
        float plus = 1.0f / (1.0f + expf(-(x + epsilon)));
        float minus = 1.0f / (1.0f + expf(-(x - epsilon)));
        ASSERT_FLOAT_NEAR(input->grad->storage->data[logical],
                          (plus - minus) / (2.0f * epsilon), 3e-5f);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row);
    ag_tensor_release(result);
    ag_tensor_release(input);
}

TEST(test_tanh_seeded_backward_and_saturation) {
    int dims[1] = {5};
    float values[5] = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    float upstream_values[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* result = ag_tanh(input);
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 5; ++i) upstream->storage->data[i] = upstream_values[i];
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->creator->operation, AG_OP_TANH);
    ASSERT_EQ_INT(ag_backward_with_grad(result, upstream), 0);
    for (int i = 0; i < 5; ++i) {
        float y = tanhf(values[i]);
        ASSERT_FLOAT_NEAR(result->value->storage->data[i], y, 1e-6f);
        ASSERT_FLOAT_NEAR(input->grad->storage->data[i],
                          upstream_values[i] * (1.0f - y * y), 1e-6f);
    }
    t_free(upstream);
    ag_tensor_release(result);
    ag_tensor_release(input);
    ASSERT_NULL(ag_tanh(NULL));
}

TEST(test_tanh_view_gradient_matches_central_difference) {
    int dims[2] = {2, 2};
    float values[4] = {-3.0f, -0.5f, 0.7f, 3.0f};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 4; ++i) base->storage->data[i] = values[i];
    tensor* view = t_transpose(base, 0, 1);
    t_free(base);
    ag_tensor* input = ag_from_owned_tensor(view, 1);
    ag_tensor* result = ag_tanh(input);
    ag_tensor* row = ag_sum(result, 1, 0);
    ag_tensor* loss = ag_sum(row, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int logical = 0; logical < 4; ++logical) {
        int storage_index = logical == 1 ? 2 : logical == 2 ? 1 : logical;
        float x = values[storage_index];
        float numerical = (tanhf(x + epsilon) - tanhf(x - epsilon)) /
                          (2.0f * epsilon);
        ASSERT_FLOAT_NEAR(input->grad->storage->data[logical], numerical, 4e-5f);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row);
    ag_tensor_release(result);
    ag_tensor_release(input);
}

static float gelu_derivative_reference(float x) {
    const float c = 0.7978845608028654f;
    const float a = 0.044715f;
    float u = c * (x + a * x * x * x);
    float t = tanhf(u);
    return 0.5f * (1.0f + t) +
           0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * a * x * x);
}

TEST(test_gelu_backward_matches_tanh_approximation) {
    int dims[1] = {6};
    float values[6] = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f, NAN};
    float upstream_values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f};
    ag_tensor* input = make_ag(1, dims, values, 1);
    ag_tensor* result = ag_gelu(input);
    tensor* upstream = t_alloc(1, dims);
    for (int i = 0; i < 6; ++i) upstream->storage->data[i] = upstream_values[i];
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_INT(result->creator->operation, AG_OP_GELU);
    ASSERT_EQ_INT(ag_backward_with_grad(result, upstream), 0);
    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_NEAR(input->grad->storage->data[i],
                          upstream_values[i] * gelu_derivative_reference(values[i]), 2e-6f);
    }
    ASSERT_NAN(result->value->storage->data[5]);
    ASSERT_NAN(input->grad->storage->data[5]);
    t_free(upstream);
    ag_tensor_release(result);
    ag_tensor_release(input);
    ASSERT_NULL(ag_gelu(NULL));
}

TEST(test_gelu_view_gradient_matches_central_difference) {
    int dims[2] = {2, 2};
    float values[4] = {-2.0f, -0.4f, 0.8f, 2.5f};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 4; ++i) base->storage->data[i] = values[i];
    tensor* view = t_transpose(base, 0, 1);
    t_free(base);
    ag_tensor* input = ag_from_owned_tensor(view, 1);
    ag_tensor* result = ag_gelu(input);
    ag_tensor* row = ag_sum(result, 1, 0);
    ag_tensor* loss = ag_sum(row, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int logical = 0; logical < 4; ++logical) {
        int storage_index = logical == 1 ? 2 : logical == 2 ? 1 : logical;
        float x = values[storage_index];
        float plus = 0.5f * (x + epsilon) *
                     (1.0f + tanhf(0.7978845608028654f *
                     ((x + epsilon) + 0.044715f * (x + epsilon) *
                     (x + epsilon) * (x + epsilon))));
        float minus = 0.5f * (x - epsilon) *
                      (1.0f + tanhf(0.7978845608028654f *
                      ((x - epsilon) + 0.044715f * (x - epsilon) *
                      (x - epsilon) * (x - epsilon))));
        ASSERT_FLOAT_NEAR(input->grad->storage->data[logical],
                          (plus - minus) / (2.0f * epsilon), 2e-4f);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row);
    ag_tensor_release(result);
    ag_tensor_release(input);
}

static float evaluate_binary_scalar(ag_op operation, float a, float b) {
    switch (operation) {
        case AG_OP_ADD: return a + b;
        case AG_OP_SUB: return a - b;
        case AG_OP_MUL: return a * b;
        case AG_OP_DIV: return a / b;
        default: return NAN;
    }
}

TEST(test_existing_arithmetic_primitives_match_individual_finite_differences) {
    const float epsilon = 1e-3f;
    const float a_value = 1.7f;
    const float b_value = 0.8f;
    ag_op binary_ops[4] = {AG_OP_ADD, AG_OP_SUB, AG_OP_MUL, AG_OP_DIV};
    ag_tensor* (*binary_functions[4])(const ag_tensor*, const ag_tensor*) = {
        ag_add, ag_sub, ag_mul, ag_div
    };
    for (int op = 0; op < 4; ++op) {
        ag_tensor* a = make_ag(0, NULL, &a_value, 1);
        ag_tensor* b = make_ag(0, NULL, &b_value, 1);
        ag_tensor* result = binary_functions[op](a, b);
        ASSERT_EQ_INT(ag_backward(result), 0);
        float numerical_a =
            (evaluate_binary_scalar(binary_ops[op], a_value + epsilon, b_value) -
             evaluate_binary_scalar(binary_ops[op], a_value - epsilon, b_value)) /
            (2.0f * epsilon);
        float numerical_b =
            (evaluate_binary_scalar(binary_ops[op], a_value, b_value + epsilon) -
             evaluate_binary_scalar(binary_ops[op], a_value, b_value - epsilon)) /
            (2.0f * epsilon);
        ASSERT_FLOAT_NEAR(a->grad->storage->data[0], numerical_a, 8e-5f);
        ASSERT_FLOAT_NEAR(b->grad->storage->data[0], numerical_b, 2e-4f);
        ag_tensor_release(result);
        ag_tensor_release(b);
        ag_tensor_release(a);
    }

    ag_tensor* (*unary_functions[3])(const ag_tensor*) = {ag_neg, ag_exp, ag_log};
    for (int op = 0; op < 3; ++op) {
        ag_tensor* input = make_ag(0, NULL, &a_value, 1);
        ag_tensor* result = unary_functions[op](input);
        ASSERT_EQ_INT(ag_backward(result), 0);
        float plus = op == 0 ? -(a_value + epsilon)
                   : op == 1 ? expf(a_value + epsilon)
                             : logf(a_value + epsilon);
        float minus = op == 0 ? -(a_value - epsilon)
                    : op == 1 ? expf(a_value - epsilon)
                              : logf(a_value - epsilon);
        ASSERT_FLOAT_NEAR(input->grad->storage->data[0],
                          (plus - minus) / (2.0f * epsilon), 3e-4f);
        ag_tensor_release(result);
        ag_tensor_release(input);
    }
}

static float evaluate_broadcast_branch(const float* x, const float* weights) {
    float result = 0.0f;
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            result += 2.0f * x[row] * weights[column];
        }
    }
    return result;
}

TEST(test_broadcast_branched_arithmetic_matches_finite_difference) {
    int x_dims[2] = {2, 1};
    int weight_dims[2] = {1, 3};
    float x_values[2] = {0.4f, 1.1f};
    float weight_values[3] = {0.3f, 0.7f, 1.2f};
    ag_tensor* x = make_ag(2, x_dims, x_values, 1);
    ag_tensor* weights = make_ag(2, weight_dims, weight_values, 1);
    ag_tensor* product = ag_mul(x, weights);
    ag_tensor* branched = ag_add(product, product);
    ag_tensor* row_sum = ag_sum(branched, 1, 0);
    ag_tensor* loss = ag_sum(row_sum, 0, 0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int i = 0; i < 2; ++i) {
        float original = x_values[i];
        x_values[i] = original + epsilon;
        float plus = evaluate_broadcast_branch(x_values, weight_values);
        x_values[i] = original - epsilon;
        float minus = evaluate_broadcast_branch(x_values, weight_values);
        x_values[i] = original;
        ASSERT_FLOAT_NEAR(x->grad->storage->data[i],
                          (plus - minus) / (2.0f * epsilon), 3e-4f);
    }
    for (int i = 0; i < 3; ++i) {
        float original = weight_values[i];
        weight_values[i] = original + epsilon;
        float plus = evaluate_broadcast_branch(x_values, weight_values);
        weight_values[i] = original - epsilon;
        float minus = evaluate_broadcast_branch(x_values, weight_values);
        weight_values[i] = original;
        ASSERT_FLOAT_NEAR(weights->grad->storage->data[i],
                          (plus - minus) / (2.0f * epsilon), 3e-4f);
    }
    ag_tensor_release(loss);
    ag_tensor_release(row_sum);
    ag_tensor_release(branched);
    ag_tensor_release(product);
    ag_tensor_release(weights);
    ag_tensor_release(x);
}

int main(void) {
    printf("== autograd_ops.c ==\n");
    RUN_TEST(test_binary_forwards_create_typed_nodes_and_retain_inputs);
    RUN_TEST(test_binary_without_grad_omits_graph);
    RUN_TEST(test_binary_local_gradients_match_derivatives);
    RUN_TEST(test_broadcast_forward_and_local_gradient_use_output_shape);
    RUN_TEST(test_add_sub_backward_reduce_broadcast_at_source);
    RUN_TEST(test_binary_rejects_null_and_incompatible_shapes);
    RUN_TEST(test_scalar_arithmetic_forward_backward_and_lifecycle);
    RUN_TEST(test_scalar_arithmetic_views_ieee_and_finite_difference);
    RUN_TEST(test_unary_forwards_and_local_gradients);
    RUN_TEST(test_unary_handles_scalar_view_and_null);
    RUN_TEST(test_pow_forward_backward_and_edge_exponents);
    RUN_TEST(test_pow_scalar_and_view_gradient_matches_central_difference);
    RUN_TEST(test_sqrt_forward_backward_preserves_ieee_domains);
    RUN_TEST(test_sqrt_scalar_gradient_matches_central_difference);
    RUN_TEST(test_relu_backward_defines_zero_and_nan_behavior);
    RUN_TEST(test_relu_strided_and_scalar_gradients);
    RUN_TEST(test_sigmoid_seeded_backward_and_saturation);
    RUN_TEST(test_sigmoid_view_gradient_matches_central_difference);
    RUN_TEST(test_tanh_seeded_backward_and_saturation);
    RUN_TEST(test_tanh_view_gradient_matches_central_difference);
    RUN_TEST(test_gelu_backward_matches_tanh_approximation);
    RUN_TEST(test_gelu_view_gradient_matches_central_difference);
    RUN_TEST(test_existing_arithmetic_primitives_match_individual_finite_differences);
    RUN_TEST(test_broadcast_branched_arithmetic_matches_finite_difference);
    TEST_SUITE_SUMMARY();
}
