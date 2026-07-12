#include "../include/test_common.h"
#include "../include/tensor.h"

TEST(test_calc_strides_3d) {
    int dims[3] = {2, 3, 4};
    int strides[3] = {0, 0, 0};
    calc_strides(3, dims, strides);
    ASSERT_EQ_INT(strides[0], 12);
    ASSERT_EQ_INT(strides[1], 4);
    ASSERT_EQ_INT(strides[2], 1);
}

TEST(test_calc_strides_1d) {
    int dims[1] = {5};
    int strides[1] = {0};
    calc_strides(1, dims, strides);
    ASSERT_EQ_INT(strides[0], 1);
}

TEST(test_calc_strides_ndim_zero_is_noop) {
    int strides[1] = {99};
    calc_strides(0, NULL, strides);
    ASSERT_EQ_INT(strides[0], 99); /* untouched */
}

TEST(test_advance_coords_increments_fastest_dim) {
    int dims[2] = {2, 3};
    int coords[2] = {0, 0};
    advance_coords(coords, dims, 2);
    ASSERT_EQ_INT(coords[0], 0);
    ASSERT_EQ_INT(coords[1], 1);
}

TEST(test_advance_coords_carries_into_next_dim) {
    int dims[2] = {2, 3};
    int coords[2] = {0, 2}; /* last position of fastest dim */
    advance_coords(coords, dims, 2);
    ASSERT_EQ_INT(coords[0], 1);
    ASSERT_EQ_INT(coords[1], 0);
}

TEST(test_advance_coords_wraps_at_full_end) {
    int dims[2] = {2, 3};
    int coords[2] = {1, 2}; /* last element overall */
    advance_coords(coords, dims, 2);
    ASSERT_EQ_INT(coords[0], 0);
    ASSERT_EQ_INT(coords[1], 0);
}

TEST(test_get_flat_index_nd_contiguous) {
    tensor t;
    int dims[2] = {2, 3};
    int strides[2] = {3, 1};
    t.dims = dims;
    t.strides = strides;
    t.ndim = 2;
    t.offset = 0;

    int coords[2] = {1, 2};
    ASSERT_EQ_INT(get_flat_index_nd(&t, coords), 5);
}

TEST(test_get_flat_index_nd_strided) {
    tensor t;
    int dims[2] = {3, 2};
    int strides[2] = {1, 3}; /* e.g. transposed view */
    t.dims = dims;
    t.strides = strides;
    t.ndim = 2;
    t.offset = 0;

    int coords[2] = {2, 1};
    ASSERT_EQ_INT(get_flat_index_nd(&t, coords), 2 * 1 + 1 * 3);
}

TEST(test_get_flat_index_nd_null_returns_zero) {
    ASSERT_EQ_INT(get_flat_index_nd(NULL, NULL), 0);
}

TEST(test_same_shape_true) {
    tensor a, b;
    int dims[2] = {2, 3};
    a.dims = dims; a.ndim = 2;
    b.dims = dims; b.ndim = 2;
    ASSERT_EQ_INT(same_shape(&a, &b), 1);
}

TEST(test_same_shape_false_different_dims) {
    tensor a, b;
    int dims_a[2] = {2, 3};
    int dims_b[2] = {3, 2};
    a.dims = dims_a; a.ndim = 2;
    b.dims = dims_b; b.ndim = 2;
    ASSERT_EQ_INT(same_shape(&a, &b), 0);
}

TEST(test_same_shape_false_different_ndim) {
    tensor a, b;
    int dims_a[2] = {2, 3};
    int dims_b[1] = {6};
    a.dims = dims_a; a.ndim = 2;
    b.dims = dims_b; b.ndim = 1;
    ASSERT_EQ_INT(same_shape(&a, &b), 0);
}

TEST(test_same_shape_null_args) {
    tensor a;
    ASSERT_EQ_INT(same_shape(NULL, &a), 0);
    ASSERT_EQ_INT(same_shape(&a, NULL), 0);
}

TEST(test_same_stride_true) {
    tensor a, b;
    int strides[2] = {3, 1};
    a.strides = strides; a.ndim = 2;
    b.strides = strides; b.ndim = 2;
    ASSERT_EQ_INT(same_stride(&a, &b), 1);
}

TEST(test_same_stride_false) {
    tensor a, b;
    int strides_a[2] = {3, 1};
    int strides_b[2] = {1, 3};
    a.strides = strides_a; a.ndim = 2;
    b.strides = strides_b; b.ndim = 2;
    ASSERT_EQ_INT(same_stride(&a, &b), 0);
}

TEST(test_is_contiguous_true_for_row_major) {
    tensor t;
    int dims[2] = {2, 3};
    int strides[2] = {3, 1};
    t.dims = dims; t.strides = strides; t.ndim = 2;
    ASSERT_EQ_INT(is_contiguous(&t), 1);
}

TEST(test_is_contiguous_false_for_transposed_layout) {
    tensor t;
    int dims[2] = {3, 2};
    int strides[2] = {1, 3}; /* transposed strides */
    t.dims = dims; t.strides = strides; t.ndim = 2;
    ASSERT_EQ_INT(is_contiguous(&t), 0);
}

TEST(test_is_contiguous_null) {
    ASSERT_EQ_INT(is_contiguous(NULL), 0);
}

TEST(test_tensor_numel_rejects_invalid_and_overflowing_shapes) {
    TEST_LOG("checking tensor_numel validation without performing allocation");
    tensor t = {0};
    int negative[2] = {2, -1};
    int zero[2] = {2, 0};
    int overflow[2] = {46341, 46341};
    t.ndim = -1;
    ASSERT_EQ_INT(tensor_numel(&t), 0);
    t.ndim = 2; t.dims = NULL;
    ASSERT_EQ_INT(tensor_numel(&t), 0);
    t.dims = negative;
    ASSERT_EQ_INT(tensor_numel(&t), 0);
    t.dims = zero;
    ASSERT_EQ_INT(tensor_numel(&t), 0);
    t.dims = overflow;
    ASSERT_EQ_INT(tensor_numel(&t), 0);
}
int main(void) {
    printf("== tensor_core.c ==\n");
    RUN_TEST(test_calc_strides_3d);
    RUN_TEST(test_calc_strides_1d);
    RUN_TEST(test_calc_strides_ndim_zero_is_noop);
    RUN_TEST(test_advance_coords_increments_fastest_dim);
    RUN_TEST(test_advance_coords_carries_into_next_dim);
    RUN_TEST(test_advance_coords_wraps_at_full_end);
    RUN_TEST(test_get_flat_index_nd_contiguous);
    RUN_TEST(test_get_flat_index_nd_strided);
    RUN_TEST(test_get_flat_index_nd_null_returns_zero);
    RUN_TEST(test_same_shape_true);
    RUN_TEST(test_same_shape_false_different_dims);
    RUN_TEST(test_same_shape_false_different_ndim);
    RUN_TEST(test_same_shape_null_args);
    RUN_TEST(test_same_stride_true);
    RUN_TEST(test_same_stride_false);
    RUN_TEST(test_is_contiguous_true_for_row_major);
    RUN_TEST(test_is_contiguous_false_for_transposed_layout);
    RUN_TEST(test_is_contiguous_null);
    RUN_TEST(test_tensor_numel_rejects_invalid_and_overflowing_shapes);
    TEST_SUITE_SUMMARY();
}


