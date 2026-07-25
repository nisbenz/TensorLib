#include "./../../../include/tensorlib/autograd.h"
#include <stdlib.h>
#include "./../../fixtures/test_common.h"

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

static ag_tensor* make_ag(int ndim, const int* dims, const float* values, int requires_grad) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

static float evaluate_smooth_model(const float* x_values, const float* w_values) {
    int x_dims[2] = {2, 3}, w_dims[1] = {3};
    tensor* x = t_alloc(2, x_dims);
    tensor* w = t_alloc(1, w_dims);
    for (int i = 0; i < 6; ++i) x->storage->data[i] = x_values[i];
    for (int i = 0; i < 3; ++i) w->storage->data[i] = w_values[i];
    tensor* product = t_mul(x, w);
    tensor* exponent = t_exp(product);
    tensor* logged = t_log(exponent);
    tensor* row_sum = t_sum(logged, 1);
    tensor* loss = t_mean(row_sum, 0);
    float result = loss->storage->data[0];
    t_free(loss); t_free(row_sum); t_free(logged); t_free(exponent);
    t_free(product); t_free(w); t_free(x);
    return result;
}

TEST(test_composed_gradient_matches_central_difference) {
    int x_dims[2] = {2, 3}, w_dims[1] = {3};
    float x_values[6] = {0.2f,0.4f,0.6f,0.8f,1.0f,1.2f};
    float w_values[3] = {0.3f,0.5f,0.7f};
    ag_tensor* x = make_ag(2,x_dims,x_values,1), *w = make_ag(1,w_dims,w_values,1);
    ag_tensor* product=ag_mul(x,w), *exponent=ag_exp(product), *logged=ag_log(exponent);
    ag_tensor* row_sum=ag_sum(logged,1,0), *loss=ag_mean(row_sum,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);

    const float epsilon=1e-3f;
    for(int i=0;i<6;++i) {
        float original=x_values[i];
        x_values[i]=original+epsilon; float plus=evaluate_smooth_model(x_values,w_values);
        x_values[i]=original-epsilon; float minus=evaluate_smooth_model(x_values,w_values);
        x_values[i]=original;
        ASSERT_FLOAT_NEAR(x->grad->storage->data[i],(plus-minus)/(2.0f*epsilon),2e-4f);
    }
    for(int i=0;i<3;++i) {
        float original=w_values[i];
        w_values[i]=original+epsilon; float plus=evaluate_smooth_model(x_values,w_values);
        w_values[i]=original-epsilon; float minus=evaluate_smooth_model(x_values,w_values);
        w_values[i]=original;
        ASSERT_FLOAT_NEAR(w->grad->storage->data[i],(plus-minus)/(2.0f*epsilon),3e-4f);
    }
    ag_tensor_release(loss); ag_tensor_release(row_sum); ag_tensor_release(logged);
    ag_tensor_release(exponent); ag_tensor_release(product); ag_tensor_release(w); ag_tensor_release(x);
}

TEST(test_view_chain_backpropagates_to_original_layout) {
    int dims[2]={3,4}; float values[12]; for(int i=0;i<12;++i) values[i]=(float)i;
    int flat_dims[1]={8};
    ag_tensor* input=make_ag(2,dims,values,1), *slice=ag_slice(input,0,1,3);
    ag_tensor* transposed=ag_transpose(slice,0,1), *flat=ag_reshape(transposed,1,flat_dims);
    ag_tensor* loss=ag_mean(flat,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);
    for(int i=0;i<4;++i) ASSERT_EQ_FLOAT(input->grad->storage->data[i],0.0f);
    for(int i=4;i<12;++i) ASSERT_EQ_FLOAT(input->grad->storage->data[i],0.125f);
    ag_tensor_release(loss); ag_tensor_release(flat); ag_tensor_release(transposed);
    ag_tensor_release(slice); ag_tensor_release(input);
}

TEST(test_expand_backpropagates_repeated_values) {
    int input_dims[2]={1,3}, expanded_dims[3]={2,4,3}; float values[3]={1,2,3};
    ag_tensor* input=make_ag(2,input_dims,values,1), *expanded=ag_expand(input,3,expanded_dims);
    ag_tensor* r1=ag_sum(expanded,2,0), *r2=ag_sum(r1,1,0), *loss=ag_sum(r2,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);
    for(int i=0;i<3;++i) ASSERT_EQ_FLOAT(input->grad->storage->data[i],8.0f);
    ag_tensor_release(loss); ag_tensor_release(r2); ag_tensor_release(r1);
    ag_tensor_release(expanded); ag_tensor_release(input);
}

TEST(test_non_gradient_input_participates_without_receiving_grad) {
    int dims[1]={3}; float xv[3]={1,2,3}, wv[3]={4,5,6};
    ag_tensor* input=make_ag(1,dims,xv,0), *weight=make_ag(1,dims,wv,1);
    ag_tensor* product=ag_mul(input,weight), *loss=ag_sum(product,0,0);
    ASSERT_EQ_INT(ag_backward(loss),0);
    ASSERT_NULL(input->grad);
    for(int i=0;i<3;++i) ASSERT_EQ_FLOAT(weight->grad->storage->data[i],xv[i]);
    ag_tensor_release(loss); ag_tensor_release(product); ag_tensor_release(weight); ag_tensor_release(input);
}

TEST(test_graph_retains_released_leaves_and_intermediates) {
    int dims[1]={3}; float values[3]={1,2,3};
    ag_tensor* input=make_ag(1,dims,values,1), *exponent=ag_exp(input), *loss=ag_sum(exponent,0,0);
    ag_tensor* retained_exponent=loss->creator->inputs[0];
    ag_tensor* retained_input=retained_exponent->creator->inputs[0];
    ag_tensor_release(exponent); ag_tensor_release(input);
    ASSERT_EQ_INT(ag_backward(loss),0);
    for(int i=0;i<3;++i) ASSERT_FLOAT_NEAR(retained_input->grad->storage->data[i],expf(values[i]),1e-5f);
    ag_tensor_release(loss);
}

TEST(test_repeated_graph_construction_and_destruction) {
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemState before, after, difference;
    _CrtMemCheckpoint(&before);
#endif
    for(int iteration=0;iteration<500;++iteration) {
        float value=1.0f+(float)iteration/1000.0f;
        ag_tensor* x=make_ag(0,NULL,&value,1), *e=ag_exp(x), *p=ag_mul(e,x), *loss=ag_log(p);
        ASSERT_EQ_INT(ag_backward(loss),0);
        ag_tensor_release(loss); ag_tensor_release(p); ag_tensor_release(e); ag_tensor_release(x);
    }
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemCheckpoint(&after);
    ASSERT_TRUE(!_CrtMemDifference(&difference, &before, &after));
#endif
}

static float logical_weighted_sum(tensor* value, const float* weights) {
    int* coords = value->ndim > 0
                ? (int*)calloc((size_t)value->ndim, sizeof(int)) : NULL;
    float result = 0.0f;
    for (int i = 0; i < tensor_numel(value); ++i) {
        result += value->storage->data[get_flat_index_nd(value, coords)] * weights[i];
        advance_coords(coords, value->dims, value->ndim);
    }
    free(coords);
    return result;
}

static float evaluate_view_loss(int operation, const float* values) {
    tensor* input = NULL;
    tensor* view = NULL;
    float result = NAN;
    if (operation == 0 || operation == 1) {
        int dims[2] = {2, 3};
        int reshaped_dims[2] = {3, 2};
        float weights[6] = {0.2f, 0.4f, 0.7f, 1.1f, 1.3f, 1.8f};
        input = t_alloc(2, dims);
        for (int i = 0; i < 6; ++i) input->storage->data[i] = values[i];
        view = operation == 0 ? t_reshape(input, 2, reshaped_dims)
                              : t_transpose(input, 0, 1);
        result = logical_weighted_sum(view, weights);
    } else if (operation == 2) {
        int dims[1] = {4};
        float weights[2] = {0.6f, 1.4f};
        input = t_alloc(1, dims);
        for (int i = 0; i < 4; ++i) input->storage->data[i] = values[i];
        view = t_slice(input, 0, 1, 3);
        result = logical_weighted_sum(view, weights);
    } else {
        int dims[2] = {2, 1};
        int expanded_dims[2] = {2, 3};
        float weights[6] = {0.1f, 0.3f, 0.5f, 0.7f, 1.0f, 1.2f};
        input = t_alloc(2, dims);
        input->storage->data[0] = values[0];
        input->storage->data[1] = values[1];
        view = t_expand(input, 2, expanded_dims);
        result = logical_weighted_sum(view, weights);
    }
    t_free(view);
    t_free(input);
    return result;
}

static ag_tensor* weighted_view_loss(ag_tensor* input, int operation) {
    ag_tensor* view = NULL;
    ag_tensor* weights = NULL;
    ag_tensor* product = NULL;
    ag_tensor* first = NULL;
    ag_tensor* loss = NULL;
    if (operation == 0 || operation == 1) {
        int output_dims[2] = {3, 2};
        float weight_values[6] = {0.2f, 0.4f, 0.7f, 1.1f, 1.3f, 1.8f};
        view = operation == 0 ? ag_reshape(input, 2, output_dims)
                              : ag_transpose(input, 0, 1);
        weights = make_ag(2, output_dims, weight_values, 0);
    } else if (operation == 2) {
        int output_dims[1] = {2};
        float weight_values[2] = {0.6f, 1.4f};
        view = ag_slice(input, 0, 1, 3);
        weights = make_ag(1, output_dims, weight_values, 0);
    } else {
        int output_dims[2] = {2, 3};
        float weight_values[6] = {0.1f, 0.3f, 0.5f, 0.7f, 1.0f, 1.2f};
        view = ag_expand(input, 2, output_dims);
        weights = make_ag(2, output_dims, weight_values, 0);
    }
    product = ag_mul(view, weights);
    if (product->value->ndim == 1) {
        loss = ag_sum(product, 0, 0);
    } else {
        first = ag_sum(product, 1, 0);
        loss = ag_sum(first, 0, 0);
    }
    ag_tensor_release(first);
    ag_tensor_release(product);
    ag_tensor_release(weights);
    ag_tensor_release(view);
    return loss;
}

TEST(test_each_view_primitive_matches_finite_difference) {
    float matrix_values[6] = {0.4f, 0.7f, 1.0f, 1.3f, 1.6f, 1.9f};
    float vector_values[4] = {0.2f, 0.5f, 0.9f, 1.4f};
    float expand_values[2] = {0.8f, 1.5f};
    const float epsilon = 1e-3f;
    for (int operation = 0; operation < 4; ++operation) {
        float* values = operation < 2 ? matrix_values
                      : operation == 2 ? vector_values : expand_values;
        int count = operation < 2 ? 6 : operation == 2 ? 4 : 2;
        int matrix_dims[2] = {2, 3};
        int vector_dims[1] = {4};
        int expand_dims[2] = {2, 1};
        ag_tensor* input = operation < 2
                         ? make_ag(2, matrix_dims, values, 1)
                         : operation == 2
                         ? make_ag(1, vector_dims, values, 1)
                         : make_ag(2, expand_dims, values, 1);
        ag_tensor* loss = weighted_view_loss(input, operation);
        ASSERT_EQ_INT(ag_backward(loss), 0);
        for (int i = 0; i < count; ++i) {
            float original = values[i];
            values[i] = original + epsilon;
            float plus = evaluate_view_loss(operation, values);
            values[i] = original - epsilon;
            float minus = evaluate_view_loss(operation, values);
            values[i] = original;
            ASSERT_FLOAT_NEAR(input->grad->storage->data[i],
                              (plus - minus) / (2.0f * epsilon), 6e-4f);
        }
        ag_tensor_release(loss);
        ag_tensor_release(input);
    }
}

static float evaluate_reduction_loss(int operation, const float* values) {
    float result = 0.0f;
    for (int row = 0; row < 2; ++row) {
        if (operation == 0) {
            for (int column = 0; column < 3; ++column) {
                result += values[row * 3 + column];
            }
        } else if (operation == 1) {
            for (int column = 0; column < 3; ++column) {
                result += values[row * 3 + column] / 3.0f;
            }
        } else {
            float maximum = values[row * 3];
            for (int column = 1; column < 3; ++column) {
                if (values[row * 3 + column] > maximum) {
                    maximum = values[row * 3 + column];
                }
            }
            result += maximum;
        }
    }
    return result;
}

TEST(test_each_reduction_primitive_matches_finite_difference) {
    int dims[2] = {2, 3};
    const float initial[6] = {0.2f, 1.1f, 0.5f, 1.7f, 0.3f, 0.8f};
    const float epsilon = 1e-3f;
    for (int operation = 0; operation < 3; ++operation) {
        float values[6];
        for (int i = 0; i < 6; ++i) values[i] = initial[i];
        ag_tensor* input = make_ag(2, dims, values, 1);
        ag_tensor* reduced = operation == 0 ? ag_sum(input, 1, 0)
                           : operation == 1 ? ag_mean(input, 1, 0)
                                            : ag_max(input, 1, 0);
        ag_tensor* loss = ag_sum(reduced, 0, 0);
        ASSERT_EQ_INT(ag_backward(loss), 0);
        for (int i = 0; i < 6; ++i) {
            float original = values[i];
            values[i] = original + epsilon;
            float plus = evaluate_reduction_loss(operation, values);
            values[i] = original - epsilon;
            float minus = evaluate_reduction_loss(operation, values);
            values[i] = original;
            ASSERT_FLOAT_NEAR(input->grad->storage->data[i],
                              (plus - minus) / (2.0f * epsilon), 2e-4f);
        }
        ag_tensor_release(loss);
        ag_tensor_release(reduced);
        ag_tensor_release(input);
    }
}

int main(void) {
    printf("== autograd_integration.c ==\n");
    RUN_TEST(test_composed_gradient_matches_central_difference);
    RUN_TEST(test_view_chain_backpropagates_to_original_layout);
    RUN_TEST(test_expand_backpropagates_repeated_values);
    RUN_TEST(test_non_gradient_input_participates_without_receiving_grad);
    RUN_TEST(test_graph_retains_released_leaves_and_intermediates);
    RUN_TEST(test_repeated_graph_construction_and_destruction);
    RUN_TEST(test_each_view_primitive_matches_finite_difference);
    RUN_TEST(test_each_reduction_primitive_matches_finite_difference);
    TEST_SUITE_SUMMARY();
}
