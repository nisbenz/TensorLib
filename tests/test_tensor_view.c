#include <stdlib.h>
#include "test_common.h"
#include "../include/tensor.h"

TEST(test_t_transpose_swaps_dims_strides_and_preserves_offset) {
    int dims[3] = {2, 3, 4};
    tensor* a = t_alloc(3, dims);

    tensor* b = t_transpose(a, 0, 2);
    ASSERT_NOT_NULL(b);
    ASSERT_EQ_INT(b->dims[0], 4);
    ASSERT_EQ_INT(b->dims[1], 3);
    ASSERT_EQ_INT(b->dims[2], 2);
    ASSERT_EQ_INT(b->strides[0], 1);
    ASSERT_EQ_INT(b->strides[1], 4);
    ASSERT_EQ_INT(b->strides[2], 12);
    ASSERT_EQ_INT(b->offset, 0);

    t_free(b);
    t_free(a);
}

TEST(test_t_transpose_shares_storage_and_bumps_ref_count) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);

    tensor* b = t_transpose(a, 0, 1);
    ASSERT_TRUE(b->storage == a->storage);
    ASSERT_EQ_INT(a->storage->ref_count, 2);

    t_free(b);
    ASSERT_EQ_INT(a->storage->ref_count, 1);
    t_free(a);
}

TEST(test_t_transpose_result_is_not_contiguous) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    tensor* b = t_transpose(a, 0, 1);

    ASSERT_EQ_INT(is_contiguous(a), 1);
    ASSERT_EQ_INT(is_contiguous(b), 0);

    t_free(b);
    t_free(a);
}

TEST(test_t_transpose_out_of_bounds_returns_null) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);

    ASSERT_NULL(t_transpose(a, -1, 0));
    ASSERT_NULL(t_transpose(a, 0, 2));

    t_free(a);
}

TEST(test_t_transpose_null_input) {
    ASSERT_NULL(t_transpose(NULL, 0, 1));
}

TEST(test_t_contiguous_on_already_contiguous_returns_equivalent_view) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    tensor* v = t_contiguous(a);
    ASSERT_NOT_NULL(v);
    ASSERT_TRUE(v->storage == a->storage);
    ASSERT_EQ_INT(a->storage->ref_count, 2);
    ASSERT_EQ_INT(v->dims[0], 2);
    ASSERT_EQ_INT(v->dims[1], 3);
    ASSERT_EQ_INT(v->offset, 0);

    t_free(v);
    t_free(a);
}

TEST(test_t_contiguous_on_transposed_materializes_correct_order) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    tensor* t = t_transpose(a, 0, 1);
    tensor* c = t_contiguous(t);

    ASSERT_NOT_NULL(c);
    ASSERT_TRUE(c->storage != a->storage);
    ASSERT_EQ_INT(is_contiguous(c), 1);
    ASSERT_EQ_INT(c->offset, 0);

    float expected[6] = {0, 3, 1, 4, 2, 5};
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(c->storage->data[i], expected[i]);
    }

    t_free(c);
    t_free(t);
    t_free(a);
}

TEST(test_t_contiguous_null_input) {
    ASSERT_NULL(t_contiguous(NULL));
}

TEST(test_t_reshape_contiguous_input_returns_view) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    int new_dims[3] = {3, 2, 1};
    tensor* r = t_reshape(a, 3, new_dims);

    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->storage == a->storage);
    ASSERT_EQ_INT(a->storage->ref_count, 2);
    ASSERT_EQ_INT(r->ndim, 3);
    ASSERT_EQ_INT(r->dims[0], 3);
    ASSERT_EQ_INT(r->dims[1], 2);
    ASSERT_EQ_INT(r->dims[2], 1);
    ASSERT_EQ_INT(r->strides[0], 2);
    ASSERT_EQ_INT(r->strides[1], 1);
    ASSERT_EQ_INT(r->strides[2], 1);
    ASSERT_EQ_FLOAT(r->storage->data[r->offset + 4], 4.0f);

    t_free(r);
    t_free(a);
}

TEST(test_t_reshape_strided_input_materializes_contiguous_tensor) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    tensor* t = t_transpose(a, 0, 1);
    int new_dims[1] = {6};
    tensor* r = t_reshape(t, 1, new_dims);

    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->storage != a->storage);
    ASSERT_EQ_INT(is_contiguous(r), 1);
    ASSERT_EQ_INT(r->offset, 0);

    float expected[6] = {0, 3, 1, 4, 2, 5};
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(r->storage->data[i], expected[i]);
    }

    t_free(r);
    t_free(t);
    t_free(a);
}

TEST(test_t_reshape_rejects_mismatched_element_count) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    int bad_dims[2] = {2, 2};
    ASSERT_NULL(t_reshape(a, 2, bad_dims));
    t_free(a);
}

TEST(test_t_slice_returns_offset_view) {
    int dims[2] = {3, 4};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 12; i++) a->storage->data[i] = (float)i;

    tensor* s = t_slice(a, 0, 1, 3);
    ASSERT_NOT_NULL(s);
    ASSERT_TRUE(s->storage == a->storage);
    ASSERT_EQ_INT(a->storage->ref_count, 2);
    ASSERT_EQ_INT(s->dims[0], 2);
    ASSERT_EQ_INT(s->dims[1], 4);
    ASSERT_EQ_INT(s->strides[0], 4);
    ASSERT_EQ_INT(s->strides[1], 1);
    ASSERT_EQ_INT(s->offset, 4);

    int coords[2] = {1, 2};
    ASSERT_EQ_FLOAT(s->storage->data[get_flat_index_nd(s, coords)], 10.0f);

    t_free(s);
    t_free(a);
}

TEST(test_t_slice_invalid_args_return_null) {
    int dims[2] = {3, 4};
    tensor* a = t_alloc(2, dims);

    ASSERT_NULL(t_slice(a, -1, 0, 1));
    ASSERT_NULL(t_slice(a, 0, -1, 1));
    ASSERT_NULL(t_slice(a, 0, 2, 2));
    ASSERT_NULL(t_slice(a, 1, 0, 5));

    t_free(a);
}

TEST(test_t_reshape_rejects_invalid_and_overflowing_dimensions) {
    int dims[1] = {6};
    tensor* a = t_alloc(1, dims);
    int negative[2] = {2, -3};
    int zero[2] = {2, 0};
    int overflow[2] = {46341, 46341};
    ASSERT_NULL(t_reshape(a, -1, NULL));
    ASSERT_NULL(t_reshape(a, 1, NULL));
    ASSERT_NULL(t_reshape(a, 2, negative));
    ASSERT_NULL(t_reshape(a, 2, zero));
    ASSERT_NULL(t_reshape(a, 2, overflow));
    t_free(a);
}
int main(void) {
    printf("== tensor_view.c ==\n");
    RUN_TEST(test_t_transpose_swaps_dims_strides_and_preserves_offset);
    RUN_TEST(test_t_transpose_shares_storage_and_bumps_ref_count);
    RUN_TEST(test_t_transpose_result_is_not_contiguous);
    RUN_TEST(test_t_transpose_out_of_bounds_returns_null);
    RUN_TEST(test_t_transpose_null_input);
    RUN_TEST(test_t_contiguous_on_already_contiguous_returns_equivalent_view);
    RUN_TEST(test_t_contiguous_on_transposed_materializes_correct_order);
    RUN_TEST(test_t_contiguous_null_input);
    RUN_TEST(test_t_reshape_contiguous_input_returns_view);
    RUN_TEST(test_t_reshape_strided_input_materializes_contiguous_tensor);
    RUN_TEST(test_t_reshape_rejects_mismatched_element_count);
    RUN_TEST(test_t_reshape_rejects_invalid_and_overflowing_dimensions);
    RUN_TEST(test_t_slice_returns_offset_view);
    RUN_TEST(test_t_slice_invalid_args_return_null);
    TEST_SUITE_SUMMARY();
}
