#include "../include/test_common.h"
#include "../include/tensor.h"

static tensor* make_vector(const float* values, int count) {
    tensor* t = t_alloc(1, &count);
    if (t != NULL) {
        for (int i = 0; i < count; ++i) t->storage->data[i] = values[i];
    }
    return t;
}

typedef tensor* (*unary_op)(tensor*);

TEST(test_t_add_contiguous_elementwise) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    tensor* b = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) {
        a->storage->data[i] = (float)i;
        b->storage->data[i] = (float)(10 * i);
    }

    tensor* c = t_add(a, b);
    ASSERT_NOT_NULL(c);
    for (int i = 0; i < 6; i++) ASSERT_EQ_FLOAT(c->storage->data[i], (float)(i + 10 * i));

    t_free(a); t_free(b); t_free(c);
}

TEST(test_t_add_result_is_independent_storage) {
    int dims[2] = {2, 2};
    tensor* a = t_alloc(2, dims);
    tensor* b = t_alloc(2, dims);
    for (int i = 0; i < 4; i++) { a->storage->data[i] = 1.0f; b->storage->data[i] = 1.0f; }

    tensor* c = t_add(a, b);
    ASSERT_TRUE(c->storage != a->storage);
    ASSERT_TRUE(c->storage != b->storage);

    t_free(a); t_free(b); t_free(c);
}

TEST(test_t_add_strided_input_via_transpose) {
    int dims_a[2] = {2, 3};
    tensor* a = t_alloc(2, dims_a);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    int dims_b[2] = {3, 2};
    tensor* b_base = t_alloc(2, dims_b);
    for (int i = 0; i < 6; i++) b_base->storage->data[i] = 100.0f;
    tensor* b = t_transpose(b_base, 0, 1);

    ASSERT_EQ_INT(same_shape(a, b), 1);
    ASSERT_EQ_INT(same_stride(a, b), 0);

    tensor* c = t_add(a, b);
    ASSERT_NOT_NULL(c);
    for (int i = 0; i < 6; i++) ASSERT_EQ_FLOAT(c->storage->data[i], (float)i + 100.0f);

    t_free(a); t_free(b); t_free(b_base); t_free(c);
}

TEST(test_t_add_shape_mismatch_returns_null) {
    int dims_a[2] = {2, 3};
    int dims_b[2] = {3, 2};
    tensor* a = t_alloc(2, dims_a);
    tensor* b = t_alloc(2, dims_b);

    ASSERT_NULL(t_add(a, b));

    t_free(a); t_free(b);
}

TEST(test_binary_operations_broadcast_singleton_dimensions) {
    int a_dims[2] = {2, 1};
    int b_dims[2] = {1, 3};
    tensor* a = t_alloc(2, a_dims);
    tensor* b = t_alloc(2, b_dims);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    a->storage->data[0] = 1.0f;
    a->storage->data[1] = 2.0f;
    b->storage->data[0] = 10.0f;
    b->storage->data[1] = 20.0f;
    b->storage->data[2] = 30.0f;

    tensor* sum = t_add(a, b);
    tensor* product = t_mul(a, b);
    ASSERT_NOT_NULL(sum);
    ASSERT_NOT_NULL(product);
    ASSERT_EQ_INT(sum->ndim, 2);
    ASSERT_EQ_INT(sum->dims[0], 2);
    ASSERT_EQ_INT(sum->dims[1], 3);

    const float expected_sum[6] = {11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f};
    const float expected_product[6] = {10.0f, 20.0f, 30.0f, 20.0f, 40.0f, 60.0f};
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ_FLOAT(sum->storage->data[i], expected_sum[i]);
        ASSERT_EQ_FLOAT(product->storage->data[i], expected_product[i]);
    }

    t_free(product); t_free(sum); t_free(b); t_free(a);
}

TEST(test_binary_operations_broadcast_strided_and_expanded_inputs) {
    int base_dims[2] = {2, 3};
    tensor* base = t_alloc(2, base_dims);
    ASSERT_NOT_NULL(base);
    for (int i = 0; i < 6; ++i) base->storage->data[i] = (float)(i + 1);

    tensor* transposed = t_transpose(base, 0, 1);
    int broadcast_dims[2] = {1, 2};
    tensor* broadcast = t_alloc(2, broadcast_dims);
    ASSERT_NOT_NULL(transposed);
    ASSERT_NOT_NULL(broadcast);
    broadcast->storage->data[0] = 100.0f;
    broadcast->storage->data[1] = 1000.0f;

    tensor* transposed_sum = t_add(transposed, broadcast);
    ASSERT_NOT_NULL(transposed_sum);
    const float expected_transposed[6] = {
        101.0f, 1004.0f, 102.0f, 1005.0f, 103.0f, 1006.0f
    };
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ_FLOAT(transposed_sum->storage->data[i], expected_transposed[i]);
    }

    int narrow_dims[2] = {2, 1};
    int expanded_dims[2] = {2, 3};
    tensor* narrow = t_alloc(2, narrow_dims);
    tensor* expanded = NULL;
    ASSERT_NOT_NULL(narrow);
    narrow->storage->data[0] = 5.0f;
    narrow->storage->data[1] = 7.0f;
    expanded = t_expand(narrow, 2, expanded_dims);
    ASSERT_NOT_NULL(expanded);

    int expanded_addend_dims[2] = {1, 3};
    tensor* expanded_addend = t_alloc(2, expanded_addend_dims);
    ASSERT_NOT_NULL(expanded_addend);
    expanded_addend->storage->data[0] = 100.0f;
    expanded_addend->storage->data[1] = 200.0f;
    expanded_addend->storage->data[2] = 300.0f;

    tensor* expanded_sum = t_add(expanded, expanded_addend);
    ASSERT_NOT_NULL(expanded_sum);
    ASSERT_EQ_FLOAT(expanded_sum->storage->data[0], 105.0f);
    ASSERT_EQ_FLOAT(expanded_sum->storage->data[1], 205.0f);
    ASSERT_EQ_FLOAT(expanded_sum->storage->data[2], 305.0f);
    ASSERT_EQ_FLOAT(expanded_sum->storage->data[3], 107.0f);
    ASSERT_EQ_FLOAT(expanded_sum->storage->data[4], 207.0f);
    ASSERT_EQ_FLOAT(expanded_sum->storage->data[5], 307.0f);

    t_free(expanded_sum); t_free(expanded_addend); t_free(expanded); t_free(narrow);
    t_free(transposed_sum); t_free(broadcast); t_free(transposed); t_free(base);
}

TEST(test_binary_operations_broadcast_rank_zero_tensor) {
    int dims[2] = {2, 2};
    tensor* matrix = t_alloc(2, dims);
    tensor* scalar = t_alloc(0, NULL);
    ASSERT_NOT_NULL(matrix);
    ASSERT_NOT_NULL(scalar);
    for (int i = 0; i < 4; ++i) matrix->storage->data[i] = (float)(i + 1);
    scalar->storage->data[0] = 10.0f;

    tensor* sum = t_add(matrix, scalar);
    ASSERT_NOT_NULL(sum);
    for (int i = 0; i < 4; ++i) ASSERT_EQ_FLOAT(sum->storage->data[i], (float)(i + 11));

    tensor* scalar_sum = t_add(scalar, scalar);
    ASSERT_NOT_NULL(scalar_sum);
    ASSERT_EQ_INT(scalar_sum->ndim, 0);
    ASSERT_EQ_FLOAT(scalar_sum->storage->data[0], 20.0f);

    t_free(scalar_sum); t_free(sum); t_free(scalar); t_free(matrix);
}

TEST(test_scalar_arithmetic_helpers) {
    const float values[4] = {2.0f, -4.0f, 0.5f, 8.0f};
    tensor* a = make_vector(values, 4);
    tensor* add = t_add_scalar(a, 2.0f);
    tensor* sub = t_sub_scalar(a, 2.0f);
    tensor* mul = t_mul_scalar(a, 2.0f);
    tensor* div = t_div_scalar(a, 2.0f);
    ASSERT_NOT_NULL(add);
    ASSERT_NOT_NULL(sub);
    ASSERT_NOT_NULL(mul);
    ASSERT_NOT_NULL(div);

    const float expected_add[4] = {4.0f, -2.0f, 2.5f, 10.0f};
    const float expected_sub[4] = {0.0f, -6.0f, -1.5f, 6.0f};
    const float expected_mul[4] = {4.0f, -8.0f, 1.0f, 16.0f};
    const float expected_div[4] = {1.0f, -2.0f, 0.25f, 4.0f};
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ_FLOAT(add->storage->data[i], expected_add[i]);
        ASSERT_EQ_FLOAT(sub->storage->data[i], expected_sub[i]);
        ASSERT_EQ_FLOAT(mul->storage->data[i], expected_mul[i]);
        ASSERT_EQ_FLOAT(div->storage->data[i], expected_div[i]);
    }

    t_free(div); t_free(mul); t_free(sub); t_free(add); t_free(a);
}

TEST(test_t_add_null_args_returns_null) {
    int dims[2] = {2, 2};
    tensor* a = t_alloc(2, dims);
    ASSERT_NULL(t_add(NULL, a));
    ASSERT_NULL(t_add(a, NULL));
    ASSERT_NULL(t_add(NULL, NULL));
    t_free(a);
}

TEST(test_sub_mul_div_contiguous_and_ieee_edges) {
    TEST_LOG("checking arithmetic results, NaN, and positive infinity");
    const float av[4] = {6.0f, -4.0f, 0.0f, 8.0f};
    const float bv[4] = {3.0f, 2.0f, 0.0f, 0.0f};
    tensor* a = make_vector(av, 4); tensor* b = make_vector(bv, 4);
    tensor* sub = t_sub(a, b); tensor* mul = t_mul(a, b); tensor* div = t_div(a, b);
    ASSERT_FLOAT_NEAR(sub->storage->data[0], 3.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(mul->storage->data[1], -8.0f, 1e-6f);
    ASSERT_NAN(div->storage->data[2]);
    ASSERT_POS_INF(div->storage->data[3]);
    t_free(div); t_free(mul); t_free(sub); t_free(b); t_free(a);
}

TEST(test_binary_operations_transposed_inputs) {
    TEST_LOG("checking logical element order for transposed operands");
    int dims[2] = {2, 3};
    tensor* ab = t_alloc(2, dims); tensor* bb = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) { ab->storage->data[i] = (float)(i + 1); bb->storage->data[i] = 2.0f; }
    tensor* a = t_transpose(ab, 0, 1); tensor* b = t_transpose(bb, 0, 1);
    tensor* sub = t_sub(a, b); tensor* mul = t_mul(a, b); tensor* div = t_div(a, b);
    const float logical[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_NEAR(sub->storage->data[i], logical[i] - 2.0f, 1e-6f);
        ASSERT_FLOAT_NEAR(mul->storage->data[i], logical[i] * 2.0f, 1e-6f);
        ASSERT_FLOAT_NEAR(div->storage->data[i], logical[i] / 2.0f, 1e-6f);
    }
    t_free(div); t_free(mul); t_free(sub); t_free(b); t_free(a); t_free(bb); t_free(ab);
}

TEST(test_all_operations_reject_null_inputs) {
    TEST_LOG("checking every P0 operation returns NULL for NULL tensor input");
    unary_op ops[] = {t_neg, t_sqrt, t_exp, t_log, t_relu, t_gelu, t_sigmoid, t_tanh};
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) ASSERT_NULL(ops[i](NULL));
    ASSERT_NULL(t_pow(NULL, 2.0f));
    ASSERT_NULL(t_sub(NULL, NULL)); ASSERT_NULL(t_mul(NULL, NULL)); ASSERT_NULL(t_div(NULL, NULL));
}

TEST(test_unary_operations_contiguous_values) {
    TEST_LOG("checking unary functions against libm/reference values");
    const float values[4] = {-2.0f, -1.0f, 0.0f, 2.0f};
    tensor* a = make_vector(values, 4);
    tensor* neg = t_neg(a); tensor* relu = t_relu(a); tensor* gelu = t_gelu(a);
    tensor* sigmoid = t_sigmoid(a); tensor* tanh_out = t_tanh(a); tensor* exp_out = t_exp(a);
    ASSERT_FLOAT_NEAR(neg->storage->data[0], 2.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(relu->storage->data[0], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(relu->storage->data[3], 2.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(gelu->storage->data[0], -0.0454023f, 1e-5f);
    ASSERT_FLOAT_NEAR(gelu->storage->data[2], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(gelu->storage->data[3], 1.9545977f, 1e-5f);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_NEAR(sigmoid->storage->data[i], 1.0f / (1.0f + expf(-values[i])), 1e-6f);
        ASSERT_FLOAT_NEAR(tanh_out->storage->data[i], tanhf(values[i]), 1e-6f);
        ASSERT_FLOAT_NEAR(exp_out->storage->data[i], expf(values[i]), 1e-6f);
    }
    t_free(exp_out); t_free(tanh_out); t_free(sigmoid); t_free(gelu); t_free(relu); t_free(neg); t_free(a);
}

TEST(test_unary_operations_transposed_inputs) {
    TEST_LOG("checking unary functions materialize transposed input in logical order");
    int dims[2] = {2, 3};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) base->storage->data[i] = (float)(i + 1);
    tensor* input = t_transpose(base, 0, 1);
    tensor* neg = t_neg(input); tensor* square = t_pow(input, 2.0f);
    const float logical[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_NEAR(neg->storage->data[i], -logical[i], 1e-6f);
        ASSERT_FLOAT_NEAR(square->storage->data[i], logical[i] * logical[i], 1e-6f);
    }
    t_free(square); t_free(neg); t_free(input); t_free(base);
}

TEST(test_unary_math_domain_uses_ieee_results) {
    TEST_LOG("checking log, sqrt, and pow preserve IEEE-754 domain results");
    const float values[3] = {-1.0f, 0.0f, 4.0f};
    tensor* a = make_vector(values, 3);
    tensor* log_out = t_log(a); tensor* sqrt_out = t_sqrt(a); tensor* pow_out = t_pow(a, 0.5f);
    ASSERT_NAN(log_out->storage->data[0]); ASSERT_NEG_INF(log_out->storage->data[1]);
    ASSERT_NAN(sqrt_out->storage->data[0]); ASSERT_FLOAT_NEAR(sqrt_out->storage->data[2], 2.0f, 1e-6f);
    ASSERT_NAN(pow_out->storage->data[0]); ASSERT_FLOAT_NEAR(pow_out->storage->data[2], 2.0f, 1e-6f);
    t_free(pow_out); t_free(sqrt_out); t_free(log_out); t_free(a);
}

int main(void) {
    printf("== tensor_ops.c ==\n");
    RUN_TEST(test_t_add_contiguous_elementwise);
    RUN_TEST(test_t_add_result_is_independent_storage);
    RUN_TEST(test_t_add_strided_input_via_transpose);
    RUN_TEST(test_t_add_shape_mismatch_returns_null);
    RUN_TEST(test_binary_operations_broadcast_singleton_dimensions);
    RUN_TEST(test_binary_operations_broadcast_strided_and_expanded_inputs);
    RUN_TEST(test_binary_operations_broadcast_rank_zero_tensor);
    RUN_TEST(test_scalar_arithmetic_helpers);
    RUN_TEST(test_t_add_null_args_returns_null);
    RUN_TEST(test_sub_mul_div_contiguous_and_ieee_edges);
    RUN_TEST(test_binary_operations_transposed_inputs);
    RUN_TEST(test_all_operations_reject_null_inputs);
    RUN_TEST(test_unary_operations_contiguous_values);
    RUN_TEST(test_unary_operations_transposed_inputs);
    RUN_TEST(test_unary_math_domain_uses_ieee_results);
    TEST_SUITE_SUMMARY();
}
