#include "test_common.h"
#include "../include/tensor.h"
static tensor* make_vector(const float* values, int count) {
    tensor* t = t_alloc(1, &count);
    if (t != NULL) for (int i = 0; i < count; i++) t->storage->data[i] = values[i];
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
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(c->storage->data[i], (float)(i + 10 * i));
    }

    t_free(a);
    t_free(b);
    t_free(c);
}

TEST(test_t_add_result_is_independent_storage) {
    int dims[2] = {2, 2};
    tensor* a = t_alloc(2, dims);
    tensor* b = t_alloc(2, dims);
    for (int i = 0; i < 4; i++) { a->storage->data[i] = 1.0f; b->storage->data[i] = 1.0f; }

    tensor* c = t_add(a, b);
    ASSERT_TRUE(c->storage != a->storage);
    ASSERT_TRUE(c->storage != b->storage);

    t_free(a);
    t_free(b);
    t_free(c);
}

TEST(test_t_add_strided_input_via_transpose) {
    /* a: 2x3 contiguous [[0,1,2],[3,4,5]]
       b: transpose of a 3x2 tensor filled with 100s -> logically 2x3 of 100s,
          but stored non-contiguously, exercising the slow coordinate path */
    int dims_a[2] = {2, 3};
    tensor* a = t_alloc(2, dims_a);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    int dims_b[2] = {3, 2};
    tensor* b_base = t_alloc(2, dims_b);
    for (int i = 0; i < 6; i++) b_base->storage->data[i] = 100.0f;
    tensor* b = t_transpose(b_base, 0, 1); /* now logically 2x3, strided */

    ASSERT_EQ_INT(same_shape(a, b), 1);
    ASSERT_EQ_INT(same_stride(a, b), 0); /* forces the non-fast-path branch */

    tensor* c = t_add(a, b);
    ASSERT_NOT_NULL(c);
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(c->storage->data[i], (float)i + 100.0f);
    }

    t_free(a);
    t_free(b);
    t_free(b_base);
    t_free(c);
}

TEST(test_t_add_shape_mismatch_returns_null) {
    int dims_a[2] = {2, 3};
    int dims_b[2] = {3, 2};
    tensor* a = t_alloc(2, dims_a);
    tensor* b = t_alloc(2, dims_b);

    ASSERT_NULL(t_add(a, b));

    t_free(a);
    t_free(b);
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
    unary_op ops[] = {t_neg, t_sqrt, t_exp, t_log, t_relu, t_gelu, t_sigmoid, t_tanh};
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) ASSERT_NULL(ops[i](NULL));
    ASSERT_NULL(t_pow(NULL, 2.0f));
    ASSERT_NULL(t_sub(NULL, NULL)); ASSERT_NULL(t_mul(NULL, NULL)); ASSERT_NULL(t_div(NULL, NULL));
}

TEST(test_unary_operations_contiguous_values) {
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
    RUN_TEST(test_t_add_null_args_returns_null);
    RUN_TEST(test_sub_mul_div_contiguous_and_ieee_edges);
    RUN_TEST(test_binary_operations_transposed_inputs);
    RUN_TEST(test_all_operations_reject_null_inputs);
    RUN_TEST(test_unary_operations_contiguous_values);
    RUN_TEST(test_unary_operations_transposed_inputs);
    RUN_TEST(test_unary_math_domain_uses_ieee_results);    TEST_SUITE_SUMMARY();
}
