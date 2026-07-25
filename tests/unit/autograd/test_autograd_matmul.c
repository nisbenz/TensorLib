#include "./../../../include/tensorlib/autograd.h"
#include <stdlib.h>
#include "./../../fixtures/test_common.h"
static ag_tensor* make_ag(int ndim, const int* dims, const float* values) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, 1);
}

static tensor* ones_like(tensor* value) {
    tensor* result = t_alloc(value->ndim, value->dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) result->storage->data[i] = 1.0f;
    return result;
}

static void run_local_backward(ag_tensor* output, tensor** gradients) {
    tensor* upstream = ones_like(output->value);
    ASSERT_EQ_INT(output->creator->backward(output->creator, upstream, gradients), 0);
    t_free(upstream);
}

TEST(test_matrix_matrix_backward) {
    int adims[2] = {2, 3}, bdims[2] = {3, 2};
    float av[6] = {1,2,3,4,5,6}, bv[6] = {1,2,3,4,5,6};
    ag_tensor* a = make_ag(2, adims, av), *b = make_ag(2, bdims, bv);
    ag_tensor* output = ag_matmul(a, b); tensor* gradients[2] = {NULL, NULL};
    float expected_a[6] = {3,7,11,3,7,11};
    float expected_b[6] = {5,5,7,7,9,9};
    ASSERT_NOT_NULL(output); run_local_backward(output, gradients);
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ_FLOAT(gradients[0]->storage->data[i], expected_a[i]);
        ASSERT_EQ_FLOAT(gradients[1]->storage->data[i], expected_b[i]);
    }
    t_free(gradients[1]); t_free(gradients[0]); ag_tensor_release(output);
    ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_transposed_view_matmul_backward_reaches_base_gradients) {
    int a_base_dims[2] = {3, 2}, b_base_dims[2] = {2, 3};
    float a_values[6] = {1,4,2,5,3,6};
    float b_values[6] = {7,9,11,8,10,12};
    ag_tensor* a_base = make_ag(2, a_base_dims, a_values);
    ag_tensor* b_base = make_ag(2, b_base_dims, b_values);
    ag_tensor* a = ag_transpose(a_base, 0, 1);
    ag_tensor* b = ag_transpose(b_base, 0, 1);
    ag_tensor* output = ag_matmul(a, b);
    ag_tensor* row_sum = ag_sum(output, 1, 0);
    ag_tensor* loss = ag_sum(row_sum, 0, 0);

    ASSERT_NOT_NULL(loss);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    ASSERT_NOT_NULL(a_base->grad);
    ASSERT_NOT_NULL(b_base->grad);

    const float expected_a[6] = {15,15,19,19,23,23};
    const float expected_b[6] = {5,7,9,5,7,9};
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ_FLOAT(a_base->grad->storage->data[i], expected_a[i]);
        ASSERT_EQ_FLOAT(b_base->grad->storage->data[i], expected_b[i]);
    }

    ag_tensor_release(loss); ag_tensor_release(row_sum); ag_tensor_release(output);
    ag_tensor_release(b); ag_tensor_release(a);
    ag_tensor_release(b_base); ag_tensor_release(a_base);
}

TEST(test_vector_vector_backward_returns_vectors) {
    int dims[1] = {3}; float av[3] = {1,2,3}, bv[3] = {4,5,6};
    ag_tensor* a = make_ag(1, dims, av), *b = make_ag(1, dims, bv);
    ag_tensor* output = ag_matmul(a, b); tensor* gradients[2] = {NULL, NULL};
    ASSERT_EQ_INT(output->value->ndim, 0); run_local_backward(output, gradients);
    ASSERT_TRUE(same_shape(gradients[0], a->value)); ASSERT_TRUE(same_shape(gradients[1], b->value));
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ_FLOAT(gradients[0]->storage->data[i], bv[i]);
        ASSERT_EQ_FLOAT(gradients[1]->storage->data[i], av[i]);
    }
    t_free(gradients[1]); t_free(gradients[0]); ag_tensor_release(output);
    ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_vector_matrix_and_matrix_vector_backward_shapes) {
    int vdims[1] = {3}, mdims[2] = {3, 2}, ldims[2] = {2, 3};
    float vv[3] = {1,2,3}, mv[6] = {1,2,3,4,5,6};
    ag_tensor* vector = make_ag(1, vdims, vv);
    ag_tensor* right = make_ag(2, mdims, mv);
    ag_tensor* left = make_ag(2, ldims, mv);
    ag_tensor* vm = ag_matmul(vector, right);
    ag_tensor* mvector = ag_matmul(left, vector);
    tensor* vm_grad[2] = {NULL,NULL}, *mv_grad[2] = {NULL,NULL};
    run_local_backward(vm, vm_grad); run_local_backward(mvector, mv_grad);
    ASSERT_TRUE(same_shape(vm_grad[0], vector->value));
    ASSERT_TRUE(same_shape(vm_grad[1], right->value));
    ASSERT_TRUE(same_shape(mv_grad[0], left->value));
    ASSERT_TRUE(same_shape(mv_grad[1], vector->value));
    t_free(mv_grad[1]); t_free(mv_grad[0]); t_free(vm_grad[1]); t_free(vm_grad[0]);
    ag_tensor_release(mvector); ag_tensor_release(vm); ag_tensor_release(left);
    ag_tensor_release(right); ag_tensor_release(vector);
}

TEST(test_batched_broadcast_backward_preserves_contribution_batch_shape) {
    int adims[3] = {2,2,3}, bdims[3] = {1,3,2};
    float av[12], bv[6]; for (int i=0;i<12;++i) av[i]=(float)(i+1); for(int i=0;i<6;++i) bv[i]=(float)(i+1);
    ag_tensor* a = make_ag(3, adims, av), *b = make_ag(3, bdims, bv);
    ag_tensor* output = ag_matmul(a,b); tensor* gradients[2]={NULL,NULL};
    ASSERT_NOT_NULL(output); run_local_backward(output, gradients);
    ASSERT_EQ_INT(gradients[0]->dims[0], 2);
    ASSERT_EQ_INT(gradients[1]->dims[0], 2);
    ASSERT_EQ_INT(gradients[1]->dims[1], 3);
    ASSERT_EQ_INT(gradients[1]->dims[2], 2);
    t_free(gradients[1]); t_free(gradients[0]); ag_tensor_release(output);
    ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_matmul_rejects_invalid_operands) {
    int adims[2]={2,3}, bdims[2]={4,2}; float av[6]={0}, bv[8]={0};
    ag_tensor* a=make_ag(2,adims,av), *b=make_ag(2,bdims,bv);
    ASSERT_NULL(ag_matmul(NULL,b)); ASSERT_NULL(ag_matmul(a,NULL)); ASSERT_NULL(ag_matmul(a,b));
    ag_tensor_release(b); ag_tensor_release(a);
}

static float logical_sum(tensor* value) {
    int* coords = value->ndim > 0
                ? (int*)calloc((size_t)value->ndim, sizeof(int)) : NULL;
    float sum = 0.0f;
    for (int i = 0; i < tensor_numel(value); ++i) {
        sum += value->storage->data[get_flat_index_nd(value, coords)];
        advance_coords(coords, value->dims, value->ndim);
    }
    free(coords);
    return sum;
}

static float evaluate_matmul_sum(int a_ndim, const int* a_dims, const float* a_values,
                                 int b_ndim, const int* b_dims, const float* b_values) {
    tensor* a = t_alloc(a_ndim, a_dims);
    tensor* b = t_alloc(b_ndim, b_dims);
    for (int i = 0; i < tensor_numel(a); ++i) a->storage->data[i] = a_values[i];
    for (int i = 0; i < tensor_numel(b); ++i) b->storage->data[i] = b_values[i];
    tensor* output = t_matmul(a, b);
    float result = logical_sum(output);
    t_free(output);
    t_free(b);
    t_free(a);
    return result;
}

static ag_tensor* sum_all(ag_tensor* value) {
    ag_tensor_retain(value);
    ag_tensor* current = value;
    while (current->value->ndim > 0) {
        ag_tensor* next = ag_sum(current, current->value->ndim - 1, 0);
        ag_tensor_release(current);
        current = next;
    }
    return current;
}

static void finite_check_matmul_case(int a_ndim, const int* a_dims, float* a_values,
                                     int b_ndim, const int* b_dims, float* b_values,
                                     float tolerance) {
    ag_tensor* a = make_ag(a_ndim, a_dims, a_values);
    ag_tensor* b = make_ag(b_ndim, b_dims, b_values);
    ag_tensor* output = ag_matmul(a, b);
    ag_tensor* loss = sum_all(output);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int i = 0; i < tensor_numel(a->value); ++i) {
        float original = a_values[i];
        a_values[i] = original + epsilon;
        float plus = evaluate_matmul_sum(a_ndim, a_dims, a_values,
                                         b_ndim, b_dims, b_values);
        a_values[i] = original - epsilon;
        float minus = evaluate_matmul_sum(a_ndim, a_dims, a_values,
                                          b_ndim, b_dims, b_values);
        a_values[i] = original;
        ASSERT_FLOAT_NEAR(a->grad->storage->data[i],
                          (plus - minus) / (2.0f * epsilon), tolerance);
    }
    for (int i = 0; i < tensor_numel(b->value); ++i) {
        float original = b_values[i];
        b_values[i] = original + epsilon;
        float plus = evaluate_matmul_sum(a_ndim, a_dims, a_values,
                                         b_ndim, b_dims, b_values);
        b_values[i] = original - epsilon;
        float minus = evaluate_matmul_sum(a_ndim, a_dims, a_values,
                                          b_ndim, b_dims, b_values);
        b_values[i] = original;
        ASSERT_FLOAT_NEAR(b->grad->storage->data[i],
                          (plus - minus) / (2.0f * epsilon), tolerance);
    }
    ag_tensor_release(loss);
    ag_tensor_release(output);
    ag_tensor_release(b);
    ag_tensor_release(a);
}

TEST(test_all_matmul_rank_forms_match_finite_differences) {
    int vector_dims[1] = {2};
    float vv_a[2] = {0.4f, 0.9f}, vv_b[2] = {0.3f, 1.1f};
    finite_check_matmul_case(1, vector_dims, vv_a, 1, vector_dims, vv_b, 2e-4f);

    int right_dims[2] = {2, 3};
    float vm_a[2] = {0.4f, 0.9f};
    float vm_b[6] = {0.2f, 0.5f, 0.7f, 1.0f, 1.2f, 1.5f};
    finite_check_matmul_case(1, vector_dims, vm_a, 2, right_dims, vm_b, 5e-4f);

    int left_dims[2] = {3, 2};
    float mv_a[6] = {0.2f, 0.5f, 0.7f, 1.0f, 1.2f, 1.5f};
    float mv_b[2] = {0.4f, 0.9f};
    finite_check_matmul_case(2, left_dims, mv_a, 1, vector_dims, mv_b, 5e-4f);

    int matrix_a_dims[2] = {2, 3};
    int matrix_b_dims[2] = {3, 2};
    float mm_a[6] = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f};
    float mm_b[6] = {0.3f, 0.5f, 0.7f, 0.9f, 1.1f, 1.3f};
    finite_check_matmul_case(2, matrix_a_dims, mm_a,
                             2, matrix_b_dims, mm_b, 1e-3f);

    int batch_a_dims[3] = {2, 2, 2};
    int batch_b_dims[3] = {1, 2, 2};
    float batch_a[8] = {0.2f,0.4f,0.6f,0.8f,1.0f,1.2f,1.4f,1.6f};
    float batch_b[4] = {0.3f,0.5f,0.7f,0.9f};
    finite_check_matmul_case(3, batch_a_dims, batch_a,
                             3, batch_b_dims, batch_b, 2e-3f);
}

static float evaluate_transposed_matmul_sum(const float* a_values,
                                            const float* b_values) {
    int a_dims[2] = {3, 2};
    int b_dims[2] = {2, 3};
    tensor* a_base = t_alloc(2, a_dims);
    tensor* b_base = t_alloc(2, b_dims);
    for (int i = 0; i < 6; ++i) {
        a_base->storage->data[i] = a_values[i];
        b_base->storage->data[i] = b_values[i];
    }
    tensor* a = t_transpose(a_base, 0, 1);
    tensor* b = t_transpose(b_base, 0, 1);
    tensor* output = t_matmul(a, b);
    float result = logical_sum(output);
    t_free(output); t_free(b); t_free(a); t_free(b_base); t_free(a_base);
    return result;
}

TEST(test_transposed_matmul_matches_finite_difference) {
    int a_dims[2] = {3, 2};
    int b_dims[2] = {2, 3};
    float a_values[6] = {0.2f,0.4f,0.6f,0.8f,1.0f,1.2f};
    float b_values[6] = {0.3f,0.5f,0.7f,0.9f,1.1f,1.3f};
    ag_tensor* a_base = make_ag(2, a_dims, a_values);
    ag_tensor* b_base = make_ag(2, b_dims, b_values);
    ag_tensor* a = ag_transpose(a_base, 0, 1);
    ag_tensor* b = ag_transpose(b_base, 0, 1);
    ag_tensor* output = ag_matmul(a, b);
    ag_tensor* loss = sum_all(output);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    const float epsilon = 1e-3f;
    for (int i = 0; i < 6; ++i) {
        float original = a_values[i];
        a_values[i] = original + epsilon;
        float plus = evaluate_transposed_matmul_sum(a_values, b_values);
        a_values[i] = original - epsilon;
        float minus = evaluate_transposed_matmul_sum(a_values, b_values);
        a_values[i] = original;
        ASSERT_FLOAT_NEAR(a_base->grad->storage->data[i],
                          (plus - minus) / (2.0f * epsilon), 1e-3f);
    }
    for (int i = 0; i < 6; ++i) {
        float original = b_values[i];
        b_values[i] = original + epsilon;
        float plus = evaluate_transposed_matmul_sum(a_values, b_values);
        b_values[i] = original - epsilon;
        float minus = evaluate_transposed_matmul_sum(a_values, b_values);
        b_values[i] = original;
        ASSERT_FLOAT_NEAR(b_base->grad->storage->data[i],
                          (plus - minus) / (2.0f * epsilon), 1e-3f);
    }
    ag_tensor_release(loss); ag_tensor_release(output);
    ag_tensor_release(b); ag_tensor_release(a);
    ag_tensor_release(b_base); ag_tensor_release(a_base);
}

int main(void) {
    printf("== autograd_matmul.c ==\n");
    RUN_TEST(test_matrix_matrix_backward);
    RUN_TEST(test_transposed_view_matmul_backward_reaches_base_gradients);
    RUN_TEST(test_vector_vector_backward_returns_vectors);
    RUN_TEST(test_vector_matrix_and_matrix_vector_backward_shapes);
    RUN_TEST(test_batched_broadcast_backward_preserves_contribution_batch_shape);
    RUN_TEST(test_matmul_rejects_invalid_operands);
    RUN_TEST(test_all_matmul_rank_forms_match_finite_differences);
    RUN_TEST(test_transposed_matmul_matches_finite_difference);
    TEST_SUITE_SUMMARY();
}
