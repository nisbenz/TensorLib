#include <stdio.h>

#include "../include/test_common.h"
#include "../include/tensor.h"

static void fill_tensor(tensor* t, const float* values, int count) {
    for (int i = 0; i < count; ++i) t->storage->data[i] = values[i];
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

int main(void) {
    printf("== tensor_matmul.c ==\n");
    RUN_TEST(test_t_matmul_2d);
    RUN_TEST(test_t_matmul_batched_3d);
    RUN_TEST(test_t_matmul_broadcasts_batch_dimensions);
    RUN_TEST(test_t_matmul_accepts_transposed_views);
    RUN_TEST(test_t_matmul_vector_cases);
    RUN_TEST(test_t_matmul_rejects_invalid_shapes);
    TEST_SUITE_SUMMARY();
}
