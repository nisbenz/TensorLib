#include <stdlib.h>
#include "test_common.h"
#include "../include/tensor.h"

TEST(test_s_alloc_basic) {
    int dims[2] = {2, 3};
    Storage* s = s_alloc(2, dims);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ_INT(s->size, 6);
    ASSERT_EQ_INT(s->ref_count, 1);
    ASSERT_NOT_NULL(s->data);
    free(s->data);
    free(s);
}

TEST(test_s_alloc_ndim_zero_gives_size_one) {
    Storage* s = s_alloc(0, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ_INT(s->size, 1);
    free(s->data);
    free(s);
}

TEST(test_s_alloc_rejects_null_dims_with_positive_ndim) {
    Storage* s = s_alloc(2, NULL);
    ASSERT_NULL(s);
}

TEST(test_t_alloc_basic_shape_and_strides) {
    int dims[3] = {2, 3, 4};
    tensor* t = t_alloc(3, dims);
    ASSERT_NOT_NULL(t);
    ASSERT_EQ_INT(t->ndim, 3);
    ASSERT_EQ_INT(t->offset, 0);
    ASSERT_EQ_INT(t->dims[0], 2);
    ASSERT_EQ_INT(t->dims[1], 3);
    ASSERT_EQ_INT(t->dims[2], 4);
    ASSERT_EQ_INT(t->strides[0], 12);
    ASSERT_EQ_INT(t->strides[1], 4);
    ASSERT_EQ_INT(t->strides[2], 1);
    ASSERT_NOT_NULL(t->storage);
    ASSERT_EQ_INT(t->storage->size, 24);
    ASSERT_NOT_NULL(t->storage->data);
    t_free(t);
}

TEST(test_t_alloc_scalar_ndim_zero) {
    tensor* t = t_alloc(0, NULL);
    ASSERT_NOT_NULL(t);
    ASSERT_EQ_INT(t->ndim, 0);
    ASSERT_EQ_INT(t->offset, 0);
    ASSERT_NULL(t->dims);
    ASSERT_NULL(t->strides);
    ASSERT_NOT_NULL(t->storage);
    ASSERT_EQ_INT(t->storage->size, 1);
    t_free(t);
}

TEST(test_t_alloc_rejects_negative_ndim) {
    tensor* t = t_alloc(-1, NULL);
    ASSERT_NULL(t);
}

TEST(test_t_alloc_rejects_null_dims_with_positive_ndim) {
    tensor* t = t_alloc(2, NULL);
    ASSERT_NULL(t);
}

TEST(test_t_free_null_is_safe) {
    t_free(NULL);
    ASSERT_TRUE(1);
}

TEST(test_t_free_decrements_shared_storage_ref_count) {
    int dims[2] = {2, 2};
    tensor* a = t_alloc(2, dims);
    ASSERT_NOT_NULL(a);

    tensor* b = (tensor*)malloc(sizeof(tensor));
    b->dims = NULL;
    b->strides = NULL;
    b->ndim = 0;
    b->offset = 0;
    b->storage = NULL;
    add_ref_count(a->storage, b);

    ASSERT_EQ_INT(a->storage->ref_count, 2);

    t_free(b);
    ASSERT_EQ_INT(a->storage->ref_count, 1);

    t_free(a);
    ASSERT_TRUE(1);
}

TEST(test_init_t_copies_shape_and_zero_fills) {
    int dims[2] = {2, 3};
    int strides[2] = {3, 1};
    tensor ref;
    ref.dims = dims;
    ref.strides = strides;
    ref.ndim = 2;
    ref.offset = 0;

    tensor* c = (tensor*)malloc(sizeof(tensor));
    int rc = init_t(c, &ref);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(c->ndim, 2);
    ASSERT_EQ_INT(c->offset, 0);
    ASSERT_EQ_INT(c->dims[0], 2);
    ASSERT_EQ_INT(c->dims[1], 3);
    ASSERT_EQ_INT(c->strides[0], 3);
    ASSERT_EQ_INT(c->strides[1], 1);
    ASSERT_EQ_INT(c->storage->size, 6);
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(c->storage->data[i], 0.0f);
    }
    t_free(c);
}

TEST(test_init_t_null_args_returns_error) {
    tensor t;
    ASSERT_EQ_INT(init_t(NULL, &t), 1);
    ASSERT_EQ_INT(init_t(&t, NULL), 1);
}

TEST(test_t_clone_deep_copies_data) {
    int dims[2] = {2, 2};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 4; i++) a->storage->data[i] = (float)(i + 1);

    tensor* b = t_clone(a);
    ASSERT_NOT_NULL(b);
    ASSERT_TRUE(b->storage != a->storage);
    ASSERT_EQ_INT(is_contiguous(b), 1);
    ASSERT_EQ_INT(b->offset, 0);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ_FLOAT(b->storage->data[i], a->storage->data[i]);
    }

    b->storage->data[0] = 999.0f;
    ASSERT_EQ_FLOAT(a->storage->data[0], 1.0f);

    t_free(a);
    t_free(b);
}

TEST(test_t_clone_materializes_strided_view_contiguously) {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) a->storage->data[i] = (float)i;

    tensor* t = t_transpose(a, 0, 1);
    tensor* b = t_clone(t);

    ASSERT_NOT_NULL(b);
    ASSERT_TRUE(b->storage != a->storage);
    ASSERT_EQ_INT(is_contiguous(b), 1);
    ASSERT_EQ_INT(b->offset, 0);

    float expected[6] = {0, 3, 1, 4, 2, 5};
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(b->storage->data[i], expected[i]);
    }

    t_free(b);
    t_free(t);
    t_free(a);
}

TEST(test_add_ref_count_links_storage_and_bumps_count) {
    int dims[2] = {2, 2};
    Storage* s = s_alloc(2, dims);
    tensor* b = (tensor*)malloc(sizeof(tensor));
    b->storage = NULL;
    b->dims = NULL;
    b->strides = NULL;
    b->ndim = 0;
    b->offset = 0;

    add_ref_count(s, b);
    ASSERT_EQ_INT(s->ref_count, 2);
    ASSERT_TRUE(b->storage == s);

    free(b);
    free(s->data);
    free(s);
}

TEST(test_add_ref_count_null_args_is_safe) {
    add_ref_count(NULL, NULL);
    ASSERT_TRUE(1);
}

int main(void) {
    printf("== tensor_alloc.c ==\n");
    RUN_TEST(test_s_alloc_basic);
    RUN_TEST(test_s_alloc_ndim_zero_gives_size_one);
    RUN_TEST(test_s_alloc_rejects_null_dims_with_positive_ndim);
    RUN_TEST(test_t_alloc_basic_shape_and_strides);
    RUN_TEST(test_t_alloc_scalar_ndim_zero);
    RUN_TEST(test_t_alloc_rejects_negative_ndim);
    RUN_TEST(test_t_alloc_rejects_null_dims_with_positive_ndim);
    RUN_TEST(test_t_free_null_is_safe);
    RUN_TEST(test_t_free_decrements_shared_storage_ref_count);
    RUN_TEST(test_init_t_copies_shape_and_zero_fills);
    RUN_TEST(test_init_t_null_args_returns_error);
    RUN_TEST(test_t_clone_deep_copies_data);
    RUN_TEST(test_t_clone_materializes_strided_view_contiguously);
    RUN_TEST(test_add_ref_count_links_storage_and_bumps_count);
    RUN_TEST(test_add_ref_count_null_args_is_safe);
    TEST_SUITE_SUMMARY();
}
