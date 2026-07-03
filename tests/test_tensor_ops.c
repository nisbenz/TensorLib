#include "test_common.h"
#include "../include/tensor.h"

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

int main(void) {
    printf("== tensor_ops.c ==\n");
    RUN_TEST(test_t_add_contiguous_elementwise);
    RUN_TEST(test_t_add_result_is_independent_storage);
    RUN_TEST(test_t_add_strided_input_via_transpose);
    RUN_TEST(test_t_add_shape_mismatch_returns_null);
    RUN_TEST(test_t_add_null_args_returns_null);
    TEST_SUITE_SUMMARY();
}
