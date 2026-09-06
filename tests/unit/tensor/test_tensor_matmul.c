#include <stdio.h>

#include "../../fixtures/test_common.h"
#include "../../../include/tensorlib/tensor.h"

static void fill_tensor(tensor* t, const float* values, int count) {
    for (int i = 0; i < count; ++i) t->storage->data[i] = values[i];
}

static void assert_same_tensor(tensor* actual, tensor* expected) {
    ASSERT_NOT_NULL(actual);
    ASSERT_NOT_NULL(expected);
    ASSERT_EQ_INT(actual->ndim, expected->ndim);
    for (int axis = 0; axis < actual->ndim; ++axis) {
        ASSERT_EQ_INT(actual->dims[axis], expected->dims[axis]);
    }
    for (int index = 0; index < tensor_numel(actual); ++index) {
        ASSERT_FLOAT_NEAR(actual->storage->data[index],
                          expected->storage->data[index], 1e-4f);
    }
}

TEST(test_t_matmul_2d) {
    int a_dims[2] = {2, 3};
    int b_dims[2] = {3, 2};
    tensor* a = t_alloc(2, a_dims);
    tensor* b = t_alloc(2, b_dims);
    const float a_values[6] = {1, 2, 3, 4, 5, 6};
    const float b_values[6] = {7, 8, 9, 10, 11, 12};
    fill_tensor(a, a_values, 6);
    fill_tensor(b, b_values, 6);

    tensor* c = t_matmul(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_INT(c->ndim, 2);
    ASSERT_EQ_INT(c->dims[0], 2);
    ASSERT_EQ_INT(c->dims[1], 2);
    ASSERT_EQ_FLOAT(c->storage->data[0], 58.0f);
    ASSERT_EQ_FLOAT(c->storage->data[1], 64.0f);
    ASSERT_EQ_FLOAT(c->storage->data[2], 139.0f);
    ASSERT_EQ_FLOAT(c->storage->data[3], 154.0f);

    t_free(c);
    t_free(b);
    t_free(a);
}

TEST(test_t_matmul_batched_3d) {
    int a_dims[3] = {2, 2, 2};
    int b_dims[3] = {2, 2, 2};
    tensor* a = t_alloc(3, a_dims);
    tensor* b = t_alloc(3, b_dims);
    const float a_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float b_values[8] = {1, 0, 0, 1, 2, 0, 0, 2};
    fill_tensor(a, a_values, 8);
    fill_tensor(b, b_values, 8);

    tensor* c = t_matmul(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_INT(c->ndim, 3);
    ASSERT_EQ_INT(c->dims[0], 2);
    ASSERT_EQ_INT(c->dims[1], 2);
    ASSERT_EQ_INT(c->dims[2], 2);

    const float expected[8] = {1, 2, 3, 4, 10, 12, 14, 16};
    for (int i = 0; i < 8; ++i) ASSERT_EQ_FLOAT(c->storage->data[i], expected[i]);

    t_free(c);
    t_free(b);
    t_free(a);
}

TEST(test_t_matmul_broadcasts_batch_dimensions) {
    int a_dims[4] = {2, 1, 2, 2};
    int b_dims[4] = {1, 3, 2, 2};
    tensor* a = t_alloc(4, a_dims);
    tensor* b = t_alloc(4, b_dims);

    const float a_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float b_values[12] = {
        1, 0, 0, 1,
        2, 0, 0, 2,
        1, 1, 1, 1
    };
    fill_tensor(a, a_values, 8);
    fill_tensor(b, b_values, 12);

    tensor* c = t_matmul(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_INT(c->ndim, 4);
    ASSERT_EQ_INT(c->dims[0], 2);
    ASSERT_EQ_INT(c->dims[1], 3);
    ASSERT_EQ_INT(c->dims[2], 2);
    ASSERT_EQ_INT(c->dims[3], 2);

    const float expected[24] = {
        1, 2, 3, 4,
        2, 4, 6, 8,
        3, 3, 7, 7,
        5, 6, 7, 8,
        10, 12, 14, 16,
        11, 11, 15, 15
    };
    for (int i = 0; i < 24; ++i) ASSERT_EQ_FLOAT(c->storage->data[i], expected[i]);

    t_free(c);
    t_free(b);
    t_free(a);
}

TEST(test_t_matmul_accepts_transposed_views) {
    int a_base_dims[2] = {3, 2};
    int b_base_dims[2] = {2, 3};
    tensor* a_base = t_alloc(2, a_base_dims);
    tensor* b_base = t_alloc(2, b_base_dims);
    const float a_base_values[6] = {1, 4, 2, 5, 3, 6};
    const float b_base_values[6] = {7, 9, 11, 8, 10, 12};
    fill_tensor(a_base, a_base_values, 6);
    fill_tensor(b_base, b_base_values, 6);

    tensor* a = t_transpose(a_base, 0, 1);
    tensor* b = t_transpose(b_base, 0, 1);
    tensor* c = t_matmul(a, b);

    ASSERT_NOT_NULL(c);
    ASSERT_EQ_FLOAT(c->storage->data[0], 58.0f);
    ASSERT_EQ_FLOAT(c->storage->data[1], 64.0f);
    ASSERT_EQ_FLOAT(c->storage->data[2], 139.0f);
    ASSERT_EQ_FLOAT(c->storage->data[3], 154.0f);

    t_free(c);
    t_free(b);
    t_free(a);
    t_free(b_base);
    t_free(a_base);
}

TEST(test_t_matmul_transposed_2d_uneven_rows_and_columns) {
    int a_dims[] = {5, 3};
    int b_dims[] = {4, 3};
    tensor* a = t_alloc(2, a_dims);
    tensor* b_base = t_alloc(2, b_dims);
    tensor* b = NULL;
    tensor* actual = NULL;

    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b_base);
    for (int i = 0; i < tensor_numel(a); ++i) {
        a->storage->data[i] = (float)(i + 1);
    }
    for (int i = 0; i < tensor_numel(b_base); ++i) {
        b_base->storage->data[i] = (float)(i - 3);
    }
    b = t_transpose(b_base, 0, 1);
    actual = t_matmul(a, b);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(actual);
    ASSERT_EQ_INT(actual->ndim, 2);
    ASSERT_EQ_INT(actual->dims[0], 5);
    ASSERT_EQ_INT(actual->dims[1], 4);
    for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 4; ++column) {
            float expected = 0.0f;
            for (int inner = 0; inner < 3; ++inner) {
                expected += a->storage->data[row * 3 + inner] *
                            b_base->storage->data[column * 3 + inner];
            }
            ASSERT_EQ_FLOAT(actual->storage->data[row * 4 + column], expected);
        }
    }
    t_free(actual);
    t_free(b);
    t_free(b_base);
    t_free(a);
}

TEST(test_t_matmul_vector_cases) {
    int vector_dims[1] = {3};
    int matrix_dims[2] = {2, 3};
    int right_matrix_dims[2] = {3, 2};
    tensor* vector = t_alloc(1, vector_dims);
    tensor* matrix = t_alloc(2, matrix_dims);
    tensor* right_matrix = t_alloc(2, right_matrix_dims);
    const float vector_values[3] = {1, 2, 3};
    const float matrix_values[6] = {1, 2, 3, 4, 5, 6};
    const float right_matrix_values[6] = {1, 2, 3, 4, 5, 6};
    fill_tensor(vector, vector_values, 3);
    fill_tensor(matrix, matrix_values, 6);
    fill_tensor(right_matrix, right_matrix_values, 6);

    tensor* dot = t_matmul(vector, vector);
    tensor* matrix_vector = t_matmul(matrix, vector);
    tensor* vector_matrix = t_matmul(vector, right_matrix);

    ASSERT_NOT_NULL(dot);
    ASSERT_NOT_NULL(matrix_vector);
    ASSERT_NOT_NULL(vector_matrix);
    ASSERT_EQ_INT(dot->ndim, 0);
    ASSERT_EQ_FLOAT(dot->storage->data[0], 14.0f);
    ASSERT_EQ_INT(matrix_vector->ndim, 1);
    ASSERT_EQ_FLOAT(matrix_vector->storage->data[0], 14.0f);
    ASSERT_EQ_FLOAT(matrix_vector->storage->data[1], 32.0f);
    ASSERT_EQ_INT(vector_matrix->ndim, 1);
    ASSERT_EQ_FLOAT(vector_matrix->storage->data[0], 22.0f);
    ASSERT_EQ_FLOAT(vector_matrix->storage->data[1], 28.0f);

    t_free(vector_matrix);
    t_free(matrix_vector);
    t_free(dot);
    t_free(right_matrix);
    t_free(matrix);
    t_free(vector);
}

TEST(test_packed_rhs_snapshots_transposed_view) {
    int a_dims[2] = {2, 3};
    int b_base_dims[2] = {2, 3};
    tensor* a = t_alloc(2, a_dims);
    tensor* b_base = t_alloc(2, b_base_dims);
    const float a_values[6] = {1, 2, 3, 4, 5, 6};
    const float b_values[6] = {7, 9, 11, 8, 10, 12};
    fill_tensor(a, a_values, 6);
    fill_tensor(b_base, b_values, 6);

    tensor* b = t_transpose(b_base, 0, 1);
    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(b);
    ASSERT_NOT_NULL(packed);

    b_base->storage->data[0] = -99.0f;
    t_free(b);
    t_free(b_base);

    tensor* c = t_matmul_packed_rhs(a, packed);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_FLOAT(c->storage->data[0], 58.0f);
    ASSERT_EQ_FLOAT(c->storage->data[1], 64.0f);
    ASSERT_EQ_FLOAT(c->storage->data[2], 139.0f);
    ASSERT_EQ_FLOAT(c->storage->data[3], 154.0f);

    t_free(c);
    t_free_matmul_packed_rhs(packed);
    t_free(a);
}

TEST(test_packed_rhs_transpose_pack_retains_lifetime) {
    int left_dims[2] = {1, 3};
    int rhs_dims[2] = {2, 3};
    tensor* left = t_alloc(2, left_dims);
    tensor* rhs = t_alloc(2, rhs_dims);
    const float left_values[3] = {1, 2, 3};
    const float rhs_values[6] = {1, 2, 3, 4, 5, 6};
    fill_tensor(left, left_values, 3);
    fill_tensor(rhs, rhs_values, 6);

    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs_transposed(rhs);
    ASSERT_NOT_NULL(packed);
    t_retain_matmul_packed_rhs(packed);
    t_free_matmul_packed_rhs(packed);
    tensor* result = t_matmul_packed_rhs(left, packed);
    ASSERT_NOT_NULL(result);
    ASSERT_EQ_FLOAT(result->storage->data[0], 14.0f);
    ASSERT_EQ_FLOAT(result->storage->data[1], 32.0f);

    t_free(result);
    t_free_matmul_packed_rhs(packed);
    t_free(rhs);
    t_free(left);
}

TEST(test_packed_rhs_handles_transposed_slices_and_kernel_tails) {
    int left_dims[2] = {4, 4};
    int right_dims[2] = {4, 4};
    tensor* left_base = t_alloc(2, left_dims);
    tensor* right_base = t_alloc(2, right_dims);
    for (int index = 0; index < 16; ++index) {
        left_base->storage->data[index] = (float)((index % 7) - 3);
        right_base->storage->data[index] = (float)((index % 5) - 2);
    }

    tensor* left_slice = t_slice(left_base, 0, 1, 4);
    tensor* left = t_transpose(left_slice, 0, 1);
    tensor* right_slice = t_slice(right_base, 1, 1, 4);
    tensor* right = t_transpose(right_slice, 0, 1);
    tensor* expected = t_matmul(left, right);
    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(right);
    tensor* actual = t_matmul_packed_rhs(left, packed);

    assert_same_tensor(actual, expected);

    t_free(actual);
    t_free_matmul_packed_rhs(packed);
    t_free(expected);
    t_free(right);
    t_free(right_slice);
    t_free(left);
    t_free(left_slice);
    t_free(right_base);
    t_free(left_base);
}

TEST(test_packed_rhs_handles_non_multiple_kernel_dimensions) {
    int left_dims[2] = {5, 129};
    int right_dims[2] = {129, 17};
    tensor* left = t_alloc(2, left_dims);
    tensor* right = t_alloc(2, right_dims);
    for (int index = 0; index < tensor_numel(left); ++index) {
        left->storage->data[index] = (float)((index % 13) - 6) * 0.25f;
    }
    for (int index = 0; index < tensor_numel(right); ++index) {
        right->storage->data[index] = (float)((index % 11) - 5) * 0.125f;
    }

    tensor* expected = t_matmul(left, right);
    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(right);
    tensor* actual = t_matmul_packed_rhs(left, packed);

    assert_same_tensor(actual, expected);

    t_free(actual);
    t_free_matmul_packed_rhs(packed);
    t_free(expected);
    t_free(right);
    t_free(left);
}

TEST(test_packed_rhs_handles_reshape_contiguous_squeeze_and_unsqueeze) {
    int left_dims[2] = {3, 4};
    int right_base_dims[2] = {6, 4};
    int reshaped_dims[2] = {4, 6};
    tensor* left = t_alloc(2, left_dims);
    tensor* right_base = t_alloc(2, right_base_dims);
    for (int index = 0; index < tensor_numel(left); ++index) {
        left->storage->data[index] = (float)(index - 4);
    }
    for (int index = 0; index < tensor_numel(right_base); ++index) {
        right_base->storage->data[index] = (float)((index % 9) - 4);
    }

    tensor* right_transposed = t_transpose(right_base, 0, 1);
    tensor* right = t_reshape(right_transposed, 2, reshaped_dims);
    tensor* left_batched = t_unsqueeze(left, 0);
    tensor* right_batched = t_unsqueeze(right, 0);
    tensor* expected_batched = t_matmul(left_batched, right_batched);
    tensor_matmul_packed_rhs* packed_batched = t_pack_matmul_rhs(right_batched);
    tensor* actual_batched = t_matmul_packed_rhs(left_batched, packed_batched);
    assert_same_tensor(actual_batched, expected_batched);

    tensor* left_squeezed = t_squeeze(left_batched, 0);
    tensor* right_squeezed = t_squeeze(right_batched, 0);
    tensor* expected = t_matmul(left_squeezed, right_squeezed);
    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(right_squeezed);
    tensor* actual = t_matmul_packed_rhs(left_squeezed, packed);
    assert_same_tensor(actual, expected);

    t_free(actual);
    t_free_matmul_packed_rhs(packed);
    t_free(expected);
    t_free(right_squeezed);
    t_free(left_squeezed);
    t_free(actual_batched);
    t_free_matmul_packed_rhs(packed_batched);
    t_free(expected_batched);
    t_free(right_batched);
    t_free(left_batched);
    t_free(right);
    t_free(right_transposed);
    t_free(right_base);
    t_free(left);
}

TEST(test_packed_rhs_broadcasts_positive_stride_batches) {
    int left_dims[3] = {2, 3, 4};
    int right_base_dims[3] = {1, 6, 4};
    tensor* left = t_alloc(3, left_dims);
    tensor* right_base = t_alloc(3, right_base_dims);
    for (int index = 0; index < tensor_numel(left); ++index) {
        left->storage->data[index] = (float)((index % 7) - 3);
    }
    for (int index = 0; index < tensor_numel(right_base); ++index) {
        right_base->storage->data[index] = (float)((index % 5) - 2);
    }

    tensor* right = t_transpose(right_base, 1, 2);
    tensor* expected = t_matmul(left, right);
    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(right);
    tensor* actual = t_matmul_packed_rhs(left, packed);
    assert_same_tensor(actual, expected);

    t_free(actual);
    t_free_matmul_packed_rhs(packed);
    t_free(expected);
    t_free(right);
    t_free(right_base);
    t_free(left);
}

TEST(test_packed_rhs_handles_materialized_contiguous_view) {
    int left_dims[2] = {3, 4};
    int right_base_dims[2] = {6, 4};
    tensor* left = t_alloc(2, left_dims);
    tensor* right_base = t_alloc(2, right_base_dims);
    for (int index = 0; index < tensor_numel(left); ++index) {
        left->storage->data[index] = (float)((index % 5) - 2);
    }
    for (int index = 0; index < tensor_numel(right_base); ++index) {
        right_base->storage->data[index] = (float)((index % 7) - 3);
    }

    tensor* right_transposed = t_transpose(right_base, 0, 1);
    tensor* right = t_contiguous(right_transposed);
    tensor* expected = t_matmul(left, right);
    tensor_matmul_packed_rhs* packed = t_pack_matmul_rhs(right);
    tensor* actual = t_matmul_packed_rhs(left, packed);
    assert_same_tensor(actual, expected);

    t_free(actual);
    t_free_matmul_packed_rhs(packed);
    t_free(expected);
    t_free(right);
    t_free(right_transposed);
    t_free(right_base);
    t_free(left);
}

TEST(test_packed_rhs_rejects_zero_stride_and_non_matrix_rhs) {
    int left_dims[2] = {2, 3};
    int narrow_rhs_dims[2] = {3, 1};
    int expanded_rhs_dims[2] = {3, 4};
    int vector_dims[1] = {3};
    tensor* left = t_alloc(2, left_dims);
    tensor* narrow_rhs = t_alloc(2, narrow_rhs_dims);
    tensor* vector = t_alloc(1, vector_dims);
    tensor* scalar = t_alloc(0, NULL);
    for (int index = 0; index < tensor_numel(left); ++index) {
        left->storage->data[index] = (float)(index + 1);
    }
    for (int index = 0; index < tensor_numel(narrow_rhs); ++index) {
        narrow_rhs->storage->data[index] = (float)(index + 1);
    }

    tensor* expanded_rhs = t_expand(narrow_rhs, 2, expanded_rhs_dims);
    ASSERT_NULL(t_pack_matmul_rhs(expanded_rhs));
    ASSERT_NULL(t_pack_matmul_rhs(vector));
    ASSERT_NULL(t_pack_matmul_rhs(scalar));
    ASSERT_NULL(t_matmul_packed_rhs(vector, NULL));

    tensor* expected = t_matmul(left, expanded_rhs);
    ASSERT_NOT_NULL(expected);
    ASSERT_EQ_FLOAT(expected->storage->data[0], 14.0f);
    ASSERT_EQ_FLOAT(expected->storage->data[3], 14.0f);

    int narrow_left_dims[2] = {2, 1};
    int expanded_left_dims[2] = {2, 3};
    int regular_rhs_dims[2] = {3, 2};
    tensor* narrow_left = t_alloc(2, narrow_left_dims);
    tensor* regular_rhs = t_alloc(2, regular_rhs_dims);
    for (int index = 0; index < tensor_numel(narrow_left); ++index) {
        narrow_left->storage->data[index] = (float)(index + 1);
    }
    for (int index = 0; index < tensor_numel(regular_rhs); ++index) {
        regular_rhs->storage->data[index] = (float)((index % 3) - 1);
    }
    tensor* expanded_left = t_expand(narrow_left, 2, expanded_left_dims);
    tensor* fallback_expected = t_matmul(expanded_left, regular_rhs);
    tensor_matmul_packed_rhs* regular_packed = t_pack_matmul_rhs(regular_rhs);
    tensor* fallback_actual = t_matmul_packed_rhs(expanded_left, regular_packed);
    assert_same_tensor(fallback_actual, fallback_expected);

    t_free(fallback_actual);
    t_free_matmul_packed_rhs(regular_packed);
    t_free(fallback_expected);
    t_free(expanded_left);
    t_free(regular_rhs);
    t_free(narrow_left);
    t_free(expected);
    t_free(expanded_rhs);
    t_free(scalar);
    t_free(vector);
    t_free(narrow_rhs);
    t_free(left);
}

TEST(test_t_matmul_rejects_invalid_shapes) {
    int a_dims[2] = {2, 3};
    int b_dims[2] = {4, 2};
    int bad_batch_a_dims[3] = {2, 2, 3};
    int bad_batch_b_dims[3] = {3, 3, 2};
    tensor* a = t_alloc(2, a_dims);
    tensor* b = t_alloc(2, b_dims);
    tensor* bad_batch_a = t_alloc(3, bad_batch_a_dims);
    tensor* bad_batch_b = t_alloc(3, bad_batch_b_dims);
    tensor* scalar = t_alloc(0, NULL);

    ASSERT_NULL(t_matmul(NULL, a));
    ASSERT_NULL(t_matmul(a, NULL));
    ASSERT_NULL(t_matmul(a, b));
    ASSERT_NULL(t_matmul(bad_batch_a, bad_batch_b));
    ASSERT_NULL(t_matmul(scalar, a));

    t_free(scalar);
    t_free(bad_batch_b);
    t_free(bad_batch_a);
    t_free(b);
    t_free(a);
}

TEST(test_t_matmul_validates_vector_inner_dimension) {
    int a_dims[1] = {3};
    int b_dims[1] = {4};
    tensor* a = t_alloc(1, a_dims);
    tensor* b = t_alloc(1, b_dims);

    ASSERT_NULL(t_matmul(a, b));

    t_free(b);
    t_free(a);
}

TEST(test_t_matmul_allows_aliasing_and_returns_independent_storage) {
    int dims[2] = {2, 2};
    tensor* a = t_alloc(2, dims);
    const float values[4] = {1, 2, 3, 4};
    fill_tensor(a, values, 4);

    tensor* c = t_matmul(a, a);
    ASSERT_NOT_NULL(c);
    ASSERT_TRUE(c->storage != a->storage);
    ASSERT_EQ_FLOAT(c->storage->data[0], 7.0f);
    ASSERT_EQ_FLOAT(c->storage->data[1], 10.0f);
    ASSERT_EQ_FLOAT(c->storage->data[2], 15.0f);
    ASSERT_EQ_FLOAT(c->storage->data[3], 22.0f);

    c->storage->data[0] = 99.0f;
    ASSERT_EQ_FLOAT(a->storage->data[0], 1.0f);

    t_free(c);
    t_free(a);
}

int main(void) {
    printf("== tensor_matmul.c ==\n");
    RUN_TEST(test_t_matmul_2d);
    RUN_TEST(test_t_matmul_batched_3d);
    RUN_TEST(test_t_matmul_broadcasts_batch_dimensions);
    RUN_TEST(test_t_matmul_accepts_transposed_views);
    RUN_TEST(test_t_matmul_transposed_2d_uneven_rows_and_columns);
    RUN_TEST(test_t_matmul_vector_cases);
    RUN_TEST(test_packed_rhs_snapshots_transposed_view);
    RUN_TEST(test_packed_rhs_transpose_pack_retains_lifetime);
    RUN_TEST(test_packed_rhs_handles_transposed_slices_and_kernel_tails);
    RUN_TEST(test_packed_rhs_handles_non_multiple_kernel_dimensions);
    RUN_TEST(test_packed_rhs_handles_reshape_contiguous_squeeze_and_unsqueeze);
    RUN_TEST(test_packed_rhs_broadcasts_positive_stride_batches);
    RUN_TEST(test_packed_rhs_handles_materialized_contiguous_view);
    RUN_TEST(test_packed_rhs_rejects_zero_stride_and_non_matrix_rhs);
    RUN_TEST(test_t_matmul_rejects_invalid_shapes);
    RUN_TEST(test_t_matmul_validates_vector_inner_dimension);
    RUN_TEST(test_t_matmul_allows_aliasing_and_returns_independent_storage);
    TEST_SUITE_SUMMARY();
}
