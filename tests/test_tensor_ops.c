#include "../include/test_common.h"
#include "../include/tensor.h"
static tensor* make_vector(const float* values, int count) {
    tensor* t = t_alloc(1, &count);
    if (t != NULL) for (int i = 0; i < count; i++) t->storage->data[i] = values[i];
    return t;
}

static tensor* make_sequential_3d(void) {
    int dims[3] = {2, 3, 4};
    tensor* t = t_alloc(3, dims);
    if (t != NULL) {
        for (int i = 0; i < t->storage->size; ++i) {
            t->storage->data[i] = (float)i;
        }
    }
    return t;
}

static tensor* make_formula_3d(void) {
    int dims[3] = {2, 3, 4};
    tensor* t = t_alloc(3, dims);
    if (t != NULL) {
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                for (int k = 0; k < dims[2]; ++k) {
                    int coords[3] = {i, j, k};
                    t->storage->data[get_flat_index_nd(t, coords)] =
                        (float)(100 * i + 10 * j + k);
                }
            }
        }
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

TEST(test_t_sum_reduces_each_axis_of_contiguous_tensor) {
    TEST_LOG("checking output shape and values for every contiguous reduction axis");
    tensor* a = make_formula_3d();
    ASSERT_NOT_NULL(a);

    tensor* dim0 = t_sum(a, 0);
    int expected_dim0[12];
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 4; ++k) {
            expected_dim0[j * 4 + k] = 100 + 20 * j + 2 * k;
        }
    }
    ASSERT_NOT_NULL(dim0);
    ASSERT_EQ_INT(dim0->ndim, 2);
    ASSERT_EQ_INT(dim0->dims[0], 3);
    ASSERT_EQ_INT(dim0->dims[1], 4);
    for (int i = 0; i < 12; ++i) ASSERT_EQ_FLOAT(dim0->storage->data[i], expected_dim0[i]);

    tensor* dim1 = t_sum(a, 1);
    ASSERT_NOT_NULL(dim1);
    ASSERT_EQ_INT(dim1->dims[0], 2);
    ASSERT_EQ_INT(dim1->dims[1], 4);
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ_FLOAT(dim1->storage->data[i * 4 + k], 300 * i + 30 + 3 * k);
        }
    }

    tensor* dim2 = t_sum(a, 2);
    ASSERT_NOT_NULL(dim2);
    ASSERT_EQ_INT(dim2->dims[0], 2);
    ASSERT_EQ_INT(dim2->dims[1], 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ_FLOAT(dim2->storage->data[i * 3 + j], 400 * i + 40 * j + 6);
        }
    }

    t_free(dim2); t_free(dim1); t_free(dim0); t_free(a);
}

TEST(test_t_sum_one_dimensional_input_returns_scalar) {
    const float values[4] = {-2.0f, 1.5f, 0.5f, 4.0f};
    tensor* a = make_vector(values, 4);
    tensor* out = t_sum(a, 0);

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 0);
    ASSERT_EQ_INT(tensor_numel(out), 1);
    ASSERT_EQ_FLOAT(out->storage->data[0], 4.0f);

    t_free(out); t_free(a);
}

TEST(test_t_sum_handles_unit_reduction_dimension) {
    int dims[3] = {2, 1, 3};
    tensor* a = t_alloc(3, dims);
    for (int i = 0; i < 6; ++i) a->storage->data[i] = (float)(i + 1);

    tensor* out = t_sum(a, 1);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 2);
    ASSERT_EQ_INT(out->dims[0], 2);
    ASSERT_EQ_INT(out->dims[1], 3);
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(out->storage->data[i], (float)(i + 1));

    t_free(out); t_free(a);
}

TEST(test_t_sum_rejects_null_scalar_and_invalid_dimensions) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    tensor* scalar = t_alloc(0, NULL);

    ASSERT_NULL(t_sum(NULL, 0));
    ASSERT_NULL(t_sum(a, -1));
    ASSERT_NULL(t_sum(a, 2));
    ASSERT_NULL(t_sum(scalar, 0));

    t_free(scalar); t_free(a);
}

TEST(test_t_sum_transpose_view) {
    tensor* base = make_formula_3d();
    tensor* view = t_transpose(base, 0, 2); /* [4, 3, 2], strides [1, 4, 12] */
    tensor* out = t_sum(view, 2); /* reduce original dimension 0 */
    tensor* contiguous = t_contiguous(view);
    tensor* contiguous_out = t_sum(contiguous, 0);

    ASSERT_NOT_NULL(out); ASSERT_NOT_NULL(contiguous); ASSERT_NOT_NULL(contiguous_out);
    ASSERT_EQ_INT(out->ndim, 2);
    ASSERT_EQ_INT(out->dims[0], 4);
    ASSERT_EQ_INT(out->dims[1], 3);
    for (int k = 0; k < 4; ++k) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ_FLOAT(out->storage->data[k * 3 + j], 100 + 20 * j + 2 * k);
        }
    }
    ASSERT_EQ_INT(contiguous_out->dims[0], 3);
    ASSERT_EQ_INT(contiguous_out->dims[1], 2);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 2; ++i) {
            ASSERT_EQ_FLOAT(contiguous_out->storage->data[j * 2 + i],
                            6 + 4 * (100 * i + 10 * j));
        }
    }

    t_free(contiguous_out); t_free(contiguous); t_free(out); t_free(view); t_free(base);
}

TEST(test_t_sum_slice_views_preserve_offset_and_strides) {
    tensor* base = make_formula_3d();
    tensor* slice_dim0 = t_slice(base, 0, 1, 2);
    tensor* slice_dim1 = t_slice(base, 1, 1, 3);
    tensor* slice_dim2 = t_slice(base, 2, 1, 4);

    tensor* out0 = t_sum(slice_dim0, 0);
    tensor* out1 = t_sum(slice_dim1, 1);
    tensor* out2 = t_sum(slice_dim2, 2);

    ASSERT_NOT_NULL(out0); ASSERT_NOT_NULL(out1); ASSERT_NOT_NULL(out2);
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ_FLOAT(out0->storage->data[j * 4 + k], 100 + 10 * j + k);
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ_FLOAT(out1->storage->data[i * 4 + k], 30 + 200 * i + 2 * k);
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ_FLOAT(out2->storage->data[i * 3 + j], 6 + 3 * (100 * i + 10 * j));
        }
    }

    t_free(out2); t_free(out1); t_free(out0);
    t_free(slice_dim2); t_free(slice_dim1); t_free(slice_dim0); t_free(base);
}

TEST(test_t_sum_reshape_view) {
    tensor* base = make_sequential_3d();
    int reshaped_dims[2] = {4, 6};
    tensor* view = t_reshape(base, 2, reshaped_dims);
    tensor* out = t_sum(view, 1);
    const float expected[4] = {15.0f, 51.0f, 87.0f, 123.0f};

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 1);
    ASSERT_EQ_INT(out->dims[0], 4);
    for (int i = 0; i < 4; ++i) ASSERT_EQ_FLOAT(out->storage->data[i], expected[i]);

    t_free(out); t_free(view); t_free(base);
}

TEST(test_t_sum_unsqueeze_and_squeeze_views) {
    int dims[2] = {2, 3};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 6; ++i) base->storage->data[i] = (float)(i + 1);
    tensor* unsqueezed = t_unsqueeze(base, 1);
    tensor* unsqueezed_sum = t_sum(unsqueezed, 1);

    ASSERT_NOT_NULL(unsqueezed_sum);
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(unsqueezed_sum->storage->data[i], (float)(i + 1));

    int singleton_dims[3] = {2, 1, 3};
    tensor* singleton = t_alloc(3, singleton_dims);
    for (int i = 0; i < 6; ++i) singleton->storage->data[i] = (float)(i + 1);
    tensor* squeezed = t_squeeze(singleton, 1);
    tensor* squeezed_sum = t_sum(squeezed, 1);

    ASSERT_NOT_NULL(squeezed_sum);
    ASSERT_EQ_INT(squeezed_sum->ndim, 1);
    ASSERT_EQ_INT(squeezed_sum->dims[0], 2);
    ASSERT_EQ_FLOAT(squeezed_sum->storage->data[0], 6.0f);
    ASSERT_EQ_FLOAT(squeezed_sum->storage->data[1], 15.0f);

    t_free(squeezed_sum); t_free(squeezed); t_free(singleton);
    t_free(unsqueezed_sum); t_free(unsqueezed); t_free(base);
}

TEST(test_t_sum_expanded_broadcast_view) {
    int dims[3] = {2, 1, 3};
    int expanded_dims[3] = {2, 4, 3};
    tensor* base = t_alloc(3, dims);
    for (int i = 0; i < 6; ++i) base->storage->data[i] = (float)(i + 1);
    tensor* view = t_expand(base, 3, expanded_dims);
    tensor* out = t_sum(view, 1);
    const float expected[6] = {4, 8, 12, 16, 20, 24};

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 2);
    ASSERT_EQ_INT(out->dims[0], 2);
    ASSERT_EQ_INT(out->dims[1], 3);
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(out->storage->data[i], expected[i]);

    t_free(out); t_free(view); t_free(base);
}

TEST(test_t_sum_preserves_ieee_values) {
    const float values[3] = {NAN, 1.0f, INFINITY};
    tensor* a = make_vector(values, 3);
    tensor* out = t_sum(a, 0);

    ASSERT_NOT_NULL(out);
    ASSERT_NAN(out->storage->data[0]);

    t_free(out); t_free(a);
}

TEST(test_t_mean_uses_floating_point_division) {
    const float values[3] = {1.0f, 2.0f, 4.0f};
    tensor* a = make_vector(values, 3);
    tensor* out = t_mean(a, 0);

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 0);
    ASSERT_FLOAT_NEAR(out->storage->data[0], 7.0f / 3.0f, 1e-6f);

    t_free(out); t_free(a);
}

TEST(test_t_mean_reduces_strided_views) {
    int dims[2] = {2, 3};
    tensor* base = t_alloc(2, dims);
    for (int i = 0; i < 6; ++i) base->storage->data[i] = (float)(i + 1);
    tensor* view = t_transpose(base, 0, 1);
    tensor* out = t_mean(view, 1);

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 1);
    ASSERT_EQ_INT(out->dims[0], 3);
    ASSERT_FLOAT_NEAR(out->storage->data[0], 2.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(out->storage->data[1], 3.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(out->storage->data[2], 4.5f, 1e-6f);

    t_free(out); t_free(view); t_free(base);
}

TEST(test_t_mean_rejects_null_and_invalid_dimensions) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    tensor* scalar = t_alloc(0, NULL);

    ASSERT_NULL(t_mean(NULL, 0));
    ASSERT_NULL(t_mean(a, -1));
    ASSERT_NULL(t_mean(a, 2));
    ASSERT_NULL(t_mean(scalar, 0));

    t_free(scalar); t_free(a);
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
    RUN_TEST(test_unary_math_domain_uses_ieee_results);
    RUN_TEST(test_t_sum_reduces_each_axis_of_contiguous_tensor);
    RUN_TEST(test_t_sum_one_dimensional_input_returns_scalar);
    RUN_TEST(test_t_sum_handles_unit_reduction_dimension);
    RUN_TEST(test_t_sum_rejects_null_scalar_and_invalid_dimensions);
    RUN_TEST(test_t_sum_transpose_view);
    RUN_TEST(test_t_sum_slice_views_preserve_offset_and_strides);
    RUN_TEST(test_t_sum_reshape_view);
    RUN_TEST(test_t_sum_unsqueeze_and_squeeze_views);
    RUN_TEST(test_t_sum_expanded_broadcast_view);
    RUN_TEST(test_t_sum_preserves_ieee_values);
    RUN_TEST(test_t_mean_uses_floating_point_division);
    RUN_TEST(test_t_mean_reduces_strided_views);
    RUN_TEST(test_t_mean_rejects_null_and_invalid_dimensions);
    TEST_SUITE_SUMMARY();
}
