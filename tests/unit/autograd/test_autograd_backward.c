#include "autograd.h"
#include "test_common.h"

static ag_tensor* make_ag(int ndim, const int* dims, const float* values, int requires_grad) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static tensor* make_tensor(int ndim, const int* dims, const float* values) {
    tensor* result = t_alloc(ndim, dims);
    if (result == NULL) return NULL;
    for (int i = 0; i < tensor_numel(result); ++i) result->storage->data[i] = values[i];
    return result;
}

TEST(test_backward_chain_computes_weight_gradients) {
    int dims[1] = {3}; float av[3] = {2,4,6}, bv[3] = {3,5,7};
    ag_tensor* a = make_ag(1,dims,av,1), *b = make_ag(1,dims,bv,1);
    ag_tensor* product = ag_mul(a,b), *sum = ag_add(product,a), *loss = ag_mean(sum,0,0);
    ASSERT_EQ_INT(ag_backward(loss), 0);
    for (int i=0;i<3;++i) {
        ASSERT_FLOAT_NEAR(a->grad->storage->data[i], (bv[i]+1.0f)/3.0f, 1e-6f);
        ASSERT_FLOAT_NEAR(b->grad->storage->data[i], av[i]/3.0f, 1e-6f);
    }
    ag_tensor_release(loss); ag_tensor_release(sum); ag_tensor_release(product);
    ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_shared_dag_accumulates_each_branch_once) {
    int dims[1]={3}; float values[3]={1,2,3};
    ag_tensor* x=make_ag(1,dims,values,1), *square=ag_mul(x,x), *joined=ag_add(square,x);
    ag_tensor* loss=ag_sum(joined,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);
    for(int i=0;i<3;++i) ASSERT_EQ_FLOAT(x->grad->storage->data[i],2.0f*values[i]+1.0f);
    ag_tensor_release(loss); ag_tensor_release(joined); ag_tensor_release(square); ag_tensor_release(x);
}

TEST(test_backward_unbroadcasts_gradient_to_operand_shape) {
    int adims[2]={2,3}, bdims[1]={3}; float av[6]={1,2,3,4,5,6}, bv[3]={10,20,30};
    ag_tensor* a=make_ag(2,adims,av,1), *b=make_ag(1,bdims,bv,1);
    ag_tensor* added=ag_add(a,b), *first=ag_sum(added,1,0), *loss=ag_sum(first,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);
    for(int i=0;i<6;++i) ASSERT_EQ_FLOAT(a->grad->storage->data[i],1.0f);
    for(int i=0;i<3;++i) ASSERT_EQ_FLOAT(b->grad->storage->data[i],2.0f);
    ag_tensor_release(loss); ag_tensor_release(first); ag_tensor_release(added);
    ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_seeded_backward_supports_non_scalar_output) {
    int dims[1]={3}; float values[3]={1,2,3}, seeds[3]={2,3,4};
    ag_tensor* x=make_ag(1,dims,values,1), *output=ag_mul(x,x);
    tensor* seed=make_tensor(1,dims,seeds);
    ASSERT_EQ_INT(ag_backward_with_grad(output,seed),0);
    for(int i=0;i<3;++i) ASSERT_EQ_FLOAT(x->grad->storage->data[i],2.0f*values[i]*seeds[i]);
    t_free(seed); ag_tensor_release(output); ag_tensor_release(x);
}

TEST(test_repeated_backward_accumulates_without_reusing_stale_intermediates) {
    float value=3.0f; ag_tensor* x=make_ag(0,NULL,&value,1), *square=ag_mul(x,x);
    ASSERT_EQ_INT(ag_backward(square),0); ASSERT_EQ_FLOAT(x->grad->storage->data[0],6.0f);
    ASSERT_EQ_INT(ag_backward(square),0); ASSERT_EQ_FLOAT(x->grad->storage->data[0],12.0f);
    ASSERT_EQ_FLOAT(square->grad->storage->data[0],2.0f);
    ag_tensor_release(square); ag_tensor_release(x);
}

TEST(test_backward_validation_leaves_existing_gradients_unchanged) {
    int dims[1]={2}, bad_dims[1]={3}; float values[2]={1,2}, bad_values[3]={1,1,1};
    ag_tensor* x=make_ag(1,dims,values,1), *output=ag_add(x,x);
    tensor* bad_seed=make_tensor(1,bad_dims,bad_values);
    ASSERT_EQ_INT(ag_backward(output),1);
    ASSERT_EQ_INT(ag_backward_with_grad(output,bad_seed),1);
    ASSERT_NULL(x->grad);
    ag_tensor* constant=make_ag(1,dims,values,0);
    tensor* valid_seed=make_tensor(1,dims,values);
    ASSERT_EQ_INT(ag_backward_with_grad(constant,valid_seed),1);
    t_free(valid_seed); ag_tensor_release(constant); t_free(bad_seed);
    ag_tensor_release(output); ag_tensor_release(x);
}

TEST(test_batched_matmul_reduces_broadcast_weight_gradient) {
    int adims[3]={2,2,3}, bdims[3]={1,3,2}; float av[12], bv[6];
    for(int i=0;i<12;++i) av[i]=(float)(i+1); for(int i=0;i<6;++i) bv[i]=1.0f;
    ag_tensor* a=make_ag(3,adims,av,0), *b=make_ag(3,bdims,bv,1);
    ag_tensor* product=ag_matmul(a,b), *r1=ag_sum(product,2,0), *r2=ag_sum(r1,1,0), *loss=ag_sum(r2,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);
    ASSERT_TRUE(same_shape(b->grad,b->value));
    float expected[3]={22,26,30};
    for(int k=0;k<3;++k) {
        ASSERT_EQ_FLOAT(b->grad->storage->data[k*2],expected[k]);
        ASSERT_EQ_FLOAT(b->grad->storage->data[k*2+1],expected[k]);
    }
    ag_tensor_release(loss); ag_tensor_release(r2); ag_tensor_release(r1); ag_tensor_release(product);
    ag_tensor_release(b); ag_tensor_release(a);
}

TEST(test_zero_grad_single_and_entire_graph) {
    float value=2.0f; ag_tensor* x=make_ag(0,NULL,&value,1), *square=ag_mul(x,x);
    ASSERT_EQ_INT(ag_backward(square),0);
    ASSERT_NOT_NULL(x->grad); ASSERT_NOT_NULL(square->grad);
    ag_zero_grad(x); ASSERT_NULL(x->grad); ASSERT_NOT_NULL(square->grad);
    ASSERT_EQ_INT(ag_backward(square),0); ASSERT_EQ_FLOAT(x->grad->storage->data[0],4.0f);
    ag_zero_grad_all(square); ASSERT_NULL(x->grad); ASSERT_NULL(square->grad);
    ag_zero_grad(NULL); ag_zero_grad_all(NULL);
    ag_tensor_release(square); ag_tensor_release(x);
}

int main(void) {
    printf("== autograd_backward.c ==\n");
    RUN_TEST(test_backward_chain_computes_weight_gradients);
    RUN_TEST(test_shared_dag_accumulates_each_branch_once);
    RUN_TEST(test_backward_unbroadcasts_gradient_to_operand_shape);
    RUN_TEST(test_seeded_backward_supports_non_scalar_output);
    RUN_TEST(test_repeated_backward_accumulates_without_reusing_stale_intermediates);
    RUN_TEST(test_backward_validation_leaves_existing_gradients_unchanged);
    RUN_TEST(test_batched_matmul_reduces_broadcast_weight_gradient);
    RUN_TEST(test_zero_grad_single_and_entire_graph);
    TEST_SUITE_SUMMARY();
}
