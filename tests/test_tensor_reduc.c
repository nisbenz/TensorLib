#include "../include/test_common.h"
#include "../include/tensor.h"

static tensor* make_vector(const float* values, int count) {
    tensor* t = t_alloc(1, &count);
    if (t != NULL) {
        for (int i = 0; i < count; ++i) t->storage->data[i] = values[i];
    }
    return t;
}

static tensor* make_sequential_3d(void) {
    int dims[3] = {2, 3, 4};
    tensor* t = t_alloc(3, dims);
    if (t != NULL) {
        for (int i = 0; i < t->storage->size; ++i) t->storage->data[i] = (float)i;
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

TEST(test_t_max_reduces_each_axis_of_contiguous_tensor) {
    TEST_LOG("checking output shape and values for every contiguous reduction axis");
    tensor* a = make_formula_3d();

    tensor* dim0 = t_max(a, 0);
    ASSERT_NOT_NULL(dim0);
    ASSERT_EQ_INT(dim0->ndim, 2);
    ASSERT_EQ_INT(dim0->dims[0], 3);
    ASSERT_EQ_INT(dim0->dims[1], 4);
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ_FLOAT(dim0->storage->data[j * 4 + k], 100 + 10 * j + k);
        }
    }

    tensor* dim1 = t_max(a, 1);
    ASSERT_NOT_NULL(dim1);
    ASSERT_EQ_INT(dim1->ndim, 2);
    ASSERT_EQ_INT(dim1->dims[0], 2);
    ASSERT_EQ_INT(dim1->dims[1], 4);
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ_FLOAT(dim1->storage->data[i * 4 + k], 100 * i + 20 + k);
        }
    }

    tensor* dim2 = t_max(a, 2);
    ASSERT_NOT_NULL(dim2);
    ASSERT_EQ_INT(dim2->ndim, 2);
    ASSERT_EQ_INT(dim2->dims[0], 2);
    ASSERT_EQ_INT(dim2->dims[1], 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ_FLOAT(dim2->storage->data[i * 3 + j], 100 * i + 10 * j + 3);
        }
    }

    t_free(dim2); t_free(dim1); t_free(dim0); t_free(a);
}

TEST(test_t_max_one_dimensional_input_returns_scalar_and_handles_negative_values) {
    const float values[4] = {-2.0f, -11.0f, -3.5f, -8.0f};
    tensor* a = make_vector(values, 4);
    tensor* out = t_max(a, 0);

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 0);
    ASSERT_EQ_INT(tensor_numel(out), 1);
    ASSERT_EQ_FLOAT(out->storage->data[0], -2.0f);

    t_free(out); t_free(a);
}

TEST(test_t_max_handles_unit_reduction_dimension) {
    int dims[3] = {2, 1, 3};
    tensor* a = t_alloc(3, dims);
    for (int i = 0; i < 6; ++i) a->storage->data[i] = (float)(-i - 1);

    tensor* out = t_max(a, 1);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 2);
    ASSERT_EQ_INT(out->dims[0], 2);
    ASSERT_EQ_INT(out->dims[1], 3);
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(out->storage->data[i], (float)(-i - 1));

    t_free(out); t_free(a);
}

TEST(test_t_max_rejects_null_scalar_invalid_dimensions_and_invalid_metadata) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    tensor* scalar = t_alloc(0, NULL);
    tensor invalid = {0};

    ASSERT_NULL(t_max(NULL, 0));
    ASSERT_NULL(t_max(a, -1));
    ASSERT_NULL(t_max(a, 2));
    ASSERT_NULL(t_max(scalar, 0));
    ASSERT_NULL(t_max(&invalid, 0));

    t_free(scalar); t_free(a);
}

TEST(test_t_max_reduces_transposed_and_offset_views) {
    tensor* base = make_formula_3d();
    tensor* transposed = t_transpose(base, 0, 2); /* [4, 3, 2] */
    tensor* transposed_out = t_max(transposed, 2);
    ASSERT_NOT_NULL(transposed_out);
    ASSERT_EQ_INT(transposed_out->ndim, 2);
    ASSERT_EQ_INT(transposed_out->dims[0], 4);
    ASSERT_EQ_INT(transposed_out->dims[1], 3);
    for (int k = 0; k < 4; ++k) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ_FLOAT(transposed_out->storage->data[k * 3 + j], 100 + 10 * j + k);
        }
    }

    tensor* slice = t_slice(base, 1, 1, 3); /* [2, 2, 4], non-zero offset */
    tensor* slice_out = t_max(slice, 1);
    ASSERT_NOT_NULL(slice_out);
    ASSERT_EQ_INT(slice_out->ndim, 2);
    ASSERT_EQ_INT(slice_out->dims[0], 2);
    ASSERT_EQ_INT(slice_out->dims[1], 4);
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ_FLOAT(slice_out->storage->data[i * 4 + k], 100 * i + 20 + k);
        }
    }

    t_free(slice_out); t_free(slice); t_free(transposed_out); t_free(transposed); t_free(base);
}

TEST(test_t_max_reduces_expanded_broadcast_view) {
    int dims[3] = {2, 1, 3};
    int expanded_dims[3] = {2, 4, 3};
    tensor* base = t_alloc(3, dims);
    for (int i = 0; i < 6; ++i) base->storage->data[i] = (float)(i + 1);
    tensor* view = t_expand(base, 3, expanded_dims);
    tensor* out = t_max(view, 1);

    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 2);
    ASSERT_EQ_INT(out->dims[0], 2);
    ASSERT_EQ_INT(out->dims[1], 3);
    for (int i = 0; i < 6; ++i) ASSERT_EQ_FLOAT(out->storage->data[i], (float)(i + 1));

    t_free(out); t_free(view); t_free(base);
}

TEST(test_t_max_propagates_nan_and_preserves_infinities) {
    int dims[4] = {4, 3};
    tensor* a = t_alloc(2, dims);
    const float values[12] = {
        -INFINITY, -2.0f, -3.0f,
        1.0f, 5.0f, NAN,
        NAN, -INFINITY, -2.0f,
        1.0f, NAN, 4.0f
    };
    for (int i = 0; i < 12; ++i) a->storage->data[i] = values[i];

    tensor* out = t_max(a, 1);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ_INT(out->ndim, 1);
    ASSERT_EQ_INT(out->dims[0], 4);
    ASSERT_EQ_FLOAT(out->storage->data[0], -2.0f);
    ASSERT_NAN(out->storage->data[1]);
    ASSERT_NAN(out->storage->data[2]);
    ASSERT_NAN(out->storage->data[3]);

    const float infinities[3] = {-INFINITY, INFINITY, -INFINITY};
    tensor* mixed_infinity = make_vector(infinities, 3);
    tensor* infinity_out = t_max(mixed_infinity, 0);
    ASSERT_NOT_NULL(infinity_out);
    ASSERT_POS_INF(infinity_out->storage->data[0]);

    const float all_negative_infinity_values[3] = {-INFINITY, -INFINITY, -INFINITY};
    tensor* negative_infinity = make_vector(all_negative_infinity_values, 3);
    tensor* negative_infinity_out = t_max(negative_infinity, 0);
    ASSERT_NOT_NULL(negative_infinity_out);
    ASSERT_NEG_INF(negative_infinity_out->storage->data[0]);

    t_free(negative_infinity_out); t_free(negative_infinity);
    t_free(infinity_out); t_free(mixed_infinity); t_free(out); t_free(a);
}

int main(void) {
    printf("== tensor_reduc.c ==\n");
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
    RUN_TEST(test_t_max_reduces_each_axis_of_contiguous_tensor);
    RUN_TEST(test_t_max_one_dimensional_input_returns_scalar_and_handles_negative_values);
    RUN_TEST(test_t_max_handles_unit_reduction_dimension);
    RUN_TEST(test_t_max_rejects_null_scalar_invalid_dimensions_and_invalid_metadata);
    RUN_TEST(test_t_max_reduces_transposed_and_offset_views);
    RUN_TEST(test_t_max_reduces_expanded_broadcast_view);
    RUN_TEST(test_t_max_propagates_nan_and_preserves_infinities);
    TEST_SUITE_SUMMARY();
}
