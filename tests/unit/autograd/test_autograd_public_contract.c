#include <tensorlib/autograd.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int assertions;
static int current_failed;

#define CHECK(cond) do { \
    ++assertions; \
    if (!(cond)) { \
        fprintf(stderr, "    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        current_failed = 1; \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tol) do { \
    float check_a_ = (actual), check_e_ = (expected); \
    ++assertions; \
    if (!(fabsf(check_a_ - check_e_) <= (tol))) { \
        fprintf(stderr, "    FAIL %s:%d: %s = %.9g, expected %.9g (+/-%g)\n", \
                __FILE__, __LINE__, #actual, check_a_, check_e_, (double)(tol)); \
        current_failed = 1; \
    } \
} while (0)

static tensor* make_tensor(int ndim, const int* dims, const float* values)
{
    tensor* t = t_alloc(ndim, dims);
    int i;
    if (!t) return NULL;
    for (i = 0; i < tensor_numel(t); ++i) t->storage->data[t->offset + i] = values[i];
    return t;
}

static ag_tensor* make_ag(int ndim, const int* dims, const float* values, int requires_grad)
{
    return ag_from_owned_tensor(make_tensor(ndim, dims, values), requires_grad);
}

static float at(const tensor* t, int flat)
{
    int i;
    int rem = flat;
    int index = t->offset;
    for (i = t->ndim - 1; i >= 0; --i) {
        int coord = rem % t->dims[i];
        rem /= t->dims[i];
        index += coord * t->strides[i];
    }
    return t->storage->data[index];
}

static int shape_is(const tensor* t, int ndim, const int* dims)
{
    int i;
    if (!t || t->ndim != ndim) return 0;
    for (i = 0; i < ndim; ++i) if (t->dims[i] != dims[i]) return 0;
    return 1;
}

static void check_values(const tensor* t, const float* expected, int count, float tol)
{
    int i;
    CHECK(t != NULL);
    if (!t) return;
    CHECK(tensor_numel((tensor*)t) == count);
    if (tensor_numel((tensor*)t) != count) return;
    for (i = 0; i < count; ++i) CHECK_NEAR(at(t, i), expected[i], tol);
}

static void check_node(const ag_tensor* out, ag_op op, int input_count)
{
    int i;
    CHECK(out != NULL);
    if (!out) return;
    CHECK(out->requires_grad == 1);
    CHECK(out->creator != NULL);
    if (!out->creator) return;
    CHECK(out->creator->operation == op);
    CHECK(out->creator->input_count == input_count);
    CHECK(out->creator->output == out);
    CHECK(out->creator->backward != NULL);
    CHECK(out->creator->input_versions != NULL);
    CHECK(out->creator->ref_count == 1);
    for (i = 0; i < out->creator->input_count; ++i) {
        CHECK(out->creator->input_versions[i] ==
              out->creator->inputs[i]->value->storage->version);
    }
    CHECK(out->creator->output_version == out->value->storage->version);
}

static int test_construction_detach_and_retain(void)
{
    int d[1] = {2};
    float v[2] = {3, 4};
    ag_tensor* x = make_ag(1, d, v, 7);
    ag_tensor* detached;
    int storage_refs;
    CHECK(x != NULL);
    CHECK(x->value != NULL);
    CHECK(x->requires_grad == 1);
    CHECK(x->grad == NULL);
    CHECK(x->creator == NULL);
    CHECK(x->ref_count == 1);
    ag_tensor_retain(x);
    CHECK(x->ref_count == 2);
    ag_tensor_release(x);
    CHECK(x->ref_count == 1);
    storage_refs = x->value->storage->ref_count;
    detached = ag_detach(x);
    CHECK(detached != NULL);
    CHECK(detached->requires_grad == 0);
    CHECK(detached->grad == NULL);
    CHECK(detached->creator == NULL);
    CHECK(detached->ref_count == 1);
    CHECK(detached->value != x->value);
    CHECK(detached->value->storage == x->value->storage);
    CHECK(detached->value->dims[0] == x->value->dims[0]);
    CHECK(detached->value->strides[0] == x->value->strides[0]);
    CHECK(detached->value->offset == x->value->offset);
    CHECK(detached->value->storage->ref_count == storage_refs + 1);
    tensor_mark_modified(detached->value);
    CHECK(detached->value->storage->version == x->value->storage->version);
    ag_tensor_release(detached);
    CHECK(x->value->storage->ref_count == storage_refs);
    ag_tensor_release(x);
    CHECK(ag_from_owned_tensor(NULL, 1) == NULL);
    CHECK(ag_detach(NULL) == NULL);
    ag_tensor_retain(NULL);
    ag_tensor_release(NULL);
    ag_node_retain(NULL);
    ag_node_release(NULL);
    return current_failed;
}

typedef ag_tensor* (*binary_fn)(const ag_tensor*, const ag_tensor*);

static void exercise_binary(binary_fn fn, ag_op op, float expected,
                            float grad_a, float grad_b)
{
    float av = 6, bv = 2, seedv = 1;
    ag_tensor* a = make_ag(0, NULL, &av, 1);
    ag_tensor* b = make_ag(0, NULL, &bv, 1);
    ag_tensor* y;
    tensor* seed;
    int a_ref = a->ref_count, b_ref = b->ref_count;
    y = fn(a, b);
    check_node(y, op, 2);
    if (!y) goto done;
    CHECK(y->creator->inputs[0] == a);
    CHECK(y->creator->inputs[1] == b);
    CHECK(a->ref_count == a_ref + 1);
    CHECK(b->ref_count == b_ref + 1);
    CHECK(y->creator->input_versions[0] == a->value->storage->version);
    CHECK(y->creator->input_versions[1] == b->value->storage->version);
    CHECK_NEAR(at(y->value, 0), expected, 1e-6f);
    seed = make_tensor(0, NULL, &seedv);
    CHECK(ag_backward_with_grad(y, seed) == 0);
    t_free(seed);
    CHECK_NEAR(at(a->grad, 0), grad_a, 1e-5f);
    CHECK_NEAR(at(b->grad, 0), grad_b, 1e-5f);
done:
    if (y) ag_tensor_release(y);
    CHECK(a->ref_count == a_ref);
    CHECK(b->ref_count == b_ref);
    ag_tensor_release(a);
    ag_tensor_release(b);
}

static int test_binary_ops_and_node_ownership(void)
{
    exercise_binary(ag_add, AG_OP_ADD, 8, 1, 1);
    exercise_binary(ag_sub, AG_OP_SUB, 4, 1, -1);
    exercise_binary(ag_mul, AG_OP_MUL, 12, 2, 6);
    exercise_binary(ag_div, AG_OP_DIV, 3, 0.5f, -1.5f);
    return current_failed;
}

static int test_graph_omission_and_node_retain(void)
{
    float a0 = 2, b0 = 5;
    ag_tensor* a = make_ag(0, NULL, &a0, 0);
    ag_tensor* b = make_ag(0, NULL, &b0, 0);
    ag_tensor* plain = ag_add(a, b);
    ag_tensor* tracked;
    CHECK(plain != NULL);
    CHECK(plain->requires_grad == 0);
    CHECK(plain->creator == NULL);
    ag_tensor_release(plain);
    a->requires_grad = 1;
    tracked = ag_add(a, b);
    check_node(tracked, AG_OP_ADD, 2);
    if (tracked && tracked->creator) {
        int refs = tracked->creator->ref_count;
        ag_node_retain(tracked->creator);
        CHECK(tracked->creator->ref_count == refs + 1);
        ag_node_release(tracked->creator);
        CHECK(tracked->creator->ref_count == refs);
    }
    ag_tensor_release(tracked);
    ag_tensor_release(a);
    ag_tensor_release(b);
    return current_failed;
}

typedef ag_tensor* (*unary_fn)(const ag_tensor*);

static void exercise_unary(unary_fn fn, ag_op op, float xval,
                           float expected, float derivative, float tol)
{
    ag_tensor* x = make_ag(0, NULL, &xval, 1);
    ag_tensor* y = fn(x);
    check_node(y, op, 1);
    if (y) {
        CHECK_NEAR(at(y->value, 0), expected, tol);
        CHECK(ag_backward(y) == 0);
        CHECK(x->grad != NULL);
        if (x->grad) CHECK_NEAR(at(x->grad, 0), derivative, tol * 5 + 1e-6f);
    }
    ag_tensor_release(y);
    ag_tensor_release(x);
}

static int test_unary_ops(void)
{
    const float x = 0.7f;
    const float sig = 1.0f / (1.0f + expf(-x));
    const float th = tanhf(x);
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    const float u = k * (x + c * x * x * x);
    const float gelu = 0.5f * x * (1.0f + tanhf(u));
    const float gelu_d = 0.5f * (1.0f + tanhf(u)) +
        0.5f * x * (1.0f - tanhf(u) * tanhf(u)) * k * (1.0f + 3.0f * c * x * x);
    exercise_unary(ag_neg, AG_OP_NEG, x, -x, -1, 1e-6f);
    exercise_unary(ag_exp, AG_OP_EXP, x, expf(x), expf(x), 2e-6f);
    exercise_unary(ag_log, AG_OP_LOG, 2.0f, logf(2.0f), 0.5f, 2e-6f);
    exercise_unary(ag_sqrt, AG_OP_SQRT, 4.0f, 2.0f, 0.25f, 2e-6f);
    exercise_unary(ag_relu, AG_OP_RELU, -2.0f, 0.0f, 0.0f, 1e-6f);
    exercise_unary(ag_sigmoid, AG_OP_SIGMOID, x, sig, sig * (1 - sig), 2e-6f);
    exercise_unary(ag_tanh, AG_OP_TANH, x, th, 1 - th * th, 2e-6f);
    exercise_unary(ag_gelu, AG_OP_GELU, x, gelu, gelu_d, 3e-5f);
    return current_failed;
}

static int test_pow_special_cases(void)
{
    float xv = 0;
    ag_tensor* x = make_ag(0, NULL, &xv, 1);
    ag_tensor* p0 = ag_pow(x, 0);
    ag_tensor* p1 = ag_pow(x, 1);
    check_node(p0, AG_OP_POW, 1);
    check_node(p1, AG_OP_POW, 1);
    CHECK_NEAR(at(p0->value, 0), 1, 0);
    CHECK_NEAR(at(p1->value, 0), 0, 0);
    CHECK(ag_backward(p0) == 0);
    CHECK_NEAR(at(x->grad, 0), 0, 0);
    ag_zero_grad(x);
    CHECK(ag_backward(p1) == 0);
    CHECK_NEAR(at(x->grad, 0), 1, 0);
    ag_tensor_release(p0);
    ag_tensor_release(p1);
    ag_tensor_release(x);
    xv = 3;
    x = make_ag(0, NULL, &xv, 1);
    p0 = ag_pow(x, 2.5f);
    CHECK_NEAR(at(p0->value, 0), powf(3, 2.5f), 2e-5f);
    CHECK(ag_backward(p0) == 0);
    CHECK_NEAR(at(x->grad, 0), 2.5f * powf(3, 1.5f), 3e-5f);
    ag_tensor_release(p0);
    ag_tensor_release(x);
    return current_failed;
}

static int test_relu_zero_and_nan_contract(void)
{
    float vals[2] = {0.0f, NAN};
    int d[1] = {2};
    float seeds[2] = {1, 1};
    ag_tensor* x = make_ag(1, d, vals, 1);
    ag_tensor* y = ag_relu(x);
    tensor* seed = make_tensor(1, d, seeds);
    CHECK(ag_backward_with_grad(y, seed) == 0);
    CHECK_NEAR(at(x->grad, 0), 0, 0);
    CHECK(isnan(at(x->grad, 1)));
    t_free(seed);
    ag_tensor_release(y);
    ag_tensor_release(x);
    return current_failed;
}

static int test_sqrt_ieee_domain(void)
{
    float vals[2] = {0, -1};
    float seeds[2] = {1, 1};
    int d[1] = {2};
    ag_tensor* x = make_ag(1, d, vals, 1);
    ag_tensor* y = ag_sqrt(x);
    tensor* seed = make_tensor(1, d, seeds);
    CHECK(isnan(at(y->value, 1)));
    CHECK(ag_backward_with_grad(y, seed) == 0);
    CHECK(isinf(at(x->grad, 0)) && at(x->grad, 0) > 0);
    CHECK(isnan(at(x->grad, 1)));
    t_free(seed);
    ag_tensor_release(y);
    ag_tensor_release(x);
    return current_failed;
}

static int test_broadcasting_forward_and_gradients(void)
{
    int ad[2] = {2, 1}, bd[2] = {1, 3}, yd[2] = {2, 3};
    float av[2] = {1, 2}, bv[3] = {10, 20, 30};
    float expected[6] = {11, 21, 31, 12, 22, 32};
    float seedv[6] = {1, 2, 3, 4, 5, 6};
    float ga[2] = {6, 15}, gb[3] = {5, 7, 9};
    ag_tensor* a = make_ag(2, ad, av, 1);
    ag_tensor* b = make_ag(2, bd, bv, 1);
    ag_tensor* y = ag_add(a, b);
    tensor* seed = make_tensor(2, yd, seedv);
    CHECK(shape_is(y->value, 2, yd));
    check_values(y->value, expected, 6, 1e-6f);
    CHECK(ag_backward_with_grad(y, seed) == 0);
    CHECK(shape_is(a->grad, 2, ad));
    CHECK(shape_is(b->grad, 2, bd));
    check_values(a->grad, ga, 2, 1e-6f);
    check_values(b->grad, gb, 3, 1e-6f);
    t_free(seed);
    ag_tensor_release(y);
    ag_tensor_release(a);
    ag_tensor_release(b);
    return current_failed;
}

static int test_local_backward_contract(void)
{
    int ad[2] = {2, 1}, bd[2] = {1, 3}, yd[2] = {2, 3};
    float av[2] = {1, 2}, bv[3] = {10, 20, 30};
    float ones[6] = {1, 1, 1, 1, 1, 1};
    ag_tensor* a = make_ag(2, ad, av, 1);
    ag_tensor* b = make_ag(2, bd, bv, 1);
    ag_tensor* y = ag_mul(a, b);
    tensor* upstream = make_tensor(2, yd, ones);
    tensor* contributions[2] = {NULL, NULL};
    CHECK(y->creator->backward(y->creator, upstream, contributions) == 0);
    CHECK(contributions[0] != NULL);
    CHECK(contributions[1] != NULL);
    /*
     * The public contract permits reduction in the local rule or later,
     * provided accumulation receives each input's original shape.
     */
    CHECK(shape_is(contributions[0], 2, ad) || shape_is(contributions[0], 2, yd));
    CHECK(shape_is(contributions[1], 2, bd) || shape_is(contributions[1], 2, yd));
    if (shape_is(contributions[0], 2, ad)) {
        float reduced_a[2] = {60, 60};
        check_values(contributions[0], reduced_a, 2, 1e-6f);
    } else {
        float raw_a[6] = {10, 20, 30, 10, 20, 30};
        check_values(contributions[0], raw_a, 6, 1e-6f);
    }
    if (shape_is(contributions[1], 2, bd)) {
        float reduced_b[3] = {3, 3, 3};
        check_values(contributions[1], reduced_b, 3, 1e-6f);
    } else {
        float raw_b[6] = {1, 1, 1, 2, 2, 2};
        check_values(contributions[1], raw_b, 6, 1e-6f);
    }
    CHECK(a->grad == NULL && b->grad == NULL);
    t_free(contributions[0]);
    t_free(contributions[1]);
    t_free(upstream);
    ag_tensor_release(y);
    ag_tensor_release(a);
    ag_tensor_release(b);
    return current_failed;
}

static int test_views_and_view_gradients(void)
{
    int d23[2] = {2, 3}, d32[2] = {3, 2}, d6[1] = {6};
    float v[6] = {1, 2, 3, 4, 5, 6};
    float ones6[6] = {1, 1, 1, 1, 1, 1};
    ag_tensor* x;
    ag_tensor* y;
    tensor* seed;

    x = make_ag(2, d23, v, 1);
    y = ag_reshape(x, 1, d6);
    check_node(y, AG_OP_RESHAPE, 1);
    CHECK(shape_is(y->value, 1, d6));
    CHECK(y->value->storage == x->value->storage);
    CHECK(ag_backward_with_grad(y, (seed = make_tensor(1, d6, ones6))) == 0);
    check_values(x->grad, ones6, 6, 0);
    t_free(seed); ag_tensor_release(y); ag_tensor_release(x);

    x = make_ag(2, d23, v, 1);
    y = ag_transpose(x, 0, 1);
    check_node(y, AG_OP_TRANSPOSE, 1);
    CHECK(shape_is(y->value, 2, d32));
    CHECK(y->value->storage == x->value->storage);
    {
        float trans_expected[6] = {1, 4, 2, 5, 3, 6};
        float trans_seed[6] = {1, 2, 3, 4, 5, 6};
        float trans_grad[6] = {1, 3, 5, 2, 4, 6};
        check_values(y->value, trans_expected, 6, 0);
        seed = make_tensor(2, d32, trans_seed);
        CHECK(ag_backward_with_grad(y, seed) == 0);
        check_values(x->grad, trans_grad, 6, 0);
        t_free(seed);
    }
    ag_tensor_release(y); ag_tensor_release(x);

    x = make_ag(2, d23, v, 1);
    y = ag_slice(x, 1, 1, 3);
    {
        int sd[2] = {2, 2};
        float sliced[4] = {2, 3, 5, 6};
        float sseed[4] = {10, 20, 30, 40};
        float sgrad[6] = {0, 10, 20, 0, 30, 40};
        check_node(y, AG_OP_SLICE, 1);
        CHECK(shape_is(y->value, 2, sd));
        CHECK(y->value->storage == x->value->storage);
        check_values(y->value, sliced, 4, 0);
        seed = make_tensor(2, sd, sseed);
        CHECK(ag_backward_with_grad(y, seed) == 0);
        check_values(x->grad, sgrad, 6, 0);
        t_free(seed);
    }
    ag_tensor_release(y); ag_tensor_release(x);

    {
        int xd[2] = {2, 1}, ed[2] = {2, 3};
        float xv[2] = {7, 8};
        float eg[2] = {3, 3};
        x = make_ag(2, xd, xv, 1);
        y = ag_expand(x, 2, ed);
        check_node(y, AG_OP_EXPAND, 1);
        CHECK(shape_is(y->value, 2, ed));
        CHECK(y->value->storage == x->value->storage);
        CHECK(y->value->strides[1] == 0);
        CHECK(ag_backward_with_grad(y, (seed = make_tensor(2, ed, ones6))) == 0);
        check_values(x->grad, eg, 2, 0);
        t_free(seed); ag_tensor_release(y); ag_tensor_release(x);
    }
    return current_failed;
}

static int test_reductions(void)
{
    int d[2] = {2, 3}, outd[1] = {2}, keepd[2] = {2, 1};
    float v[6] = {1, 5, 2, 7, 3, 4};
    float seedv[2] = {2, 3};
    ag_tensor* x;
    ag_tensor* y;
    tensor* seed;

    x = make_ag(2, d, v, 1); y = ag_sum(x, 1, 0);
    {
        float out[2] = {8, 14}, grad[6] = {2, 2, 2, 3, 3, 3};
        check_node(y, AG_OP_SUM, 1); CHECK(shape_is(y->value, 1, outd));
        check_values(y->value, out, 2, 0);
        seed = make_tensor(1, outd, seedv); CHECK(ag_backward_with_grad(y, seed) == 0);
        check_values(x->grad, grad, 6, 0); t_free(seed);
    }
    ag_tensor_release(y); ag_tensor_release(x);

    x = make_ag(2, d, v, 1); y = ag_mean(x, 1, 1);
    {
        float out[2] = {8.0f/3.0f, 14.0f/3.0f};
        float grad[6] = {2.0f/3, 2.0f/3, 2.0f/3, 1, 1, 1};
        check_node(y, AG_OP_MEAN, 1); CHECK(shape_is(y->value, 2, keepd));
        check_values(y->value, out, 2, 1e-6f);
        seed = make_tensor(2, keepd, seedv); CHECK(ag_backward_with_grad(y, seed) == 0);
        check_values(x->grad, grad, 6, 1e-6f); t_free(seed);
    }
    ag_tensor_release(y); ag_tensor_release(x);

    x = make_ag(2, d, v, 1); y = ag_max(x, 1, 0);
    {
        float out[2] = {5, 7}, grad[6] = {0, 2, 0, 3, 0, 0};
        check_node(y, AG_OP_MAX, 1); CHECK(shape_is(y->value, 1, outd));
        check_values(y->value, out, 2, 0);
        seed = make_tensor(1, outd, seedv); CHECK(ag_backward_with_grad(y, seed) == 0);
        check_values(x->grad, grad, 6, 0); t_free(seed);
    }
    ag_tensor_release(y); ag_tensor_release(x);
    return current_failed;
}

static int test_matmul(void)
{
    int ad[2] = {2, 3}, bd[2] = {3, 2}, yd[2] = {2, 2};
    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[6] = {7, 8, 9, 10, 11, 12};
    float out[4] = {58, 64, 139, 154};
    float seedv[4] = {1, 2, 3, 4};
    float ga[6] = {23, 29, 35, 53, 67, 81};
    float gb[6] = {13, 18, 17, 24, 21, 30};
    ag_tensor* a = make_ag(2, ad, av, 1);
    ag_tensor* b = make_ag(2, bd, bv, 1);
    ag_tensor* y = ag_matmul(a, b);
    tensor* seed = make_tensor(2, yd, seedv);
    check_node(y, AG_OP_MATMUL, 2);
    CHECK(shape_is(y->value, 2, yd));
    check_values(y->value, out, 4, 0);
    CHECK(ag_backward_with_grad(y, seed) == 0);
    check_values(a->grad, ga, 6, 0);
    check_values(b->grad, gb, 6, 0);
    t_free(seed); ag_tensor_release(y); ag_tensor_release(a); ag_tensor_release(b);
    return current_failed;
}

static int test_scalar_seeded_accumulation_and_zero(void)
{
    float xv = 3;
    float seedv = 4;
    ag_tensor* x = make_ag(0, NULL, &xv, 1);
    ag_tensor* square = ag_mul(x, x);
    tensor* seed;
    CHECK(ag_backward(square) == 0);
    CHECK_NEAR(at(x->grad, 0), 6, 0);
    CHECK(ag_backward(square) == 0);
    CHECK_NEAR(at(x->grad, 0), 12, 0);
    seed = make_tensor(0, NULL, &seedv);
    CHECK(ag_backward_with_grad(square, seed) == 0);
    CHECK_NEAR(at(x->grad, 0), 36, 0);
    t_free(seed);
    CHECK(square->grad != NULL);
    ag_zero_grad(x);
    CHECK(x->grad == NULL);
    ag_zero_grad(x);
    ag_zero_grad(NULL);
    ag_zero_grad_all(square);
    CHECK(square->grad == NULL);
    CHECK(x->grad == NULL);
    ag_zero_grad_all(NULL);
    ag_tensor_release(square);
    ag_tensor_release(x);
    return current_failed;
}

static int test_non_scalar_seed_requirement_and_transaction(void)
{
    int d2[1] = {2}, d1[1] = {1};
    float xv[2] = {2, 3}, badv = 9;
    ag_tensor* x = make_ag(1, d2, xv, 1);
    ag_tensor* y = ag_mul(x, x);
    tensor* bad = make_tensor(1, d1, &badv);
    CHECK(ag_backward(y) != 0);
    CHECK(x->grad == NULL);
    CHECK(y->grad == NULL);
    CHECK(ag_backward_with_grad(y, bad) != 0);
    CHECK(x->grad == NULL);
    CHECK(y->grad == NULL);
    CHECK(ag_backward_with_grad(y, NULL) != 0);
    CHECK(x->grad == NULL);
    t_free(bad);
    ag_tensor_release(y);
    ag_tensor_release(x);
    CHECK(ag_backward(NULL) != 0);
    CHECK(ag_backward_with_grad(NULL, NULL) != 0);
    return current_failed;
}

static int test_stale_graph_transactional_rejection(void)
{
    float xv = 2;
    ag_tensor* x = make_ag(0, NULL, &xv, 1);
    ag_tensor* y = ag_mul(x, x);
    float prior, output_prior;
    uint64_t captured = y->creator->input_versions[0];
    CHECK(ag_backward(y) == 0);
    prior = at(x->grad, 0);
    output_prior = at(y->grad, 0);
    x->value->storage->data[x->value->offset] = 5;
    tensor_mark_modified(x->value);
    CHECK(x->value->storage->version != captured);
    CHECK(ag_backward(y) != 0);
    CHECK_NEAR(at(x->grad, 0), prior, 0);
    CHECK_NEAR(at(y->grad, 0), output_prior, 0);
    ag_tensor_release(y); ag_tensor_release(x);

    xv = 3;
    x = make_ag(0, NULL, &xv, 1);
    y = ag_exp(x);
    CHECK(ag_backward(y) == 0);
    prior = at(x->grad, 0);
    output_prior = at(y->grad, 0);
    y->value->storage->data[y->value->offset] += 1;
    tensor_mark_modified(y->value);
    CHECK(y->value->storage->version != y->creator->output_version);
    CHECK(ag_backward(y) != 0);
    CHECK_NEAR(at(x->grad, 0), prior, 0);
    CHECK_NEAR(at(y->grad, 0), output_prior, 0);
    ag_tensor_release(y); ag_tensor_release(x);
    return current_failed;
}

static int test_invalid_operation_arguments(void)
{
    int d2[1] = {2}, bad0[1] = {0}, badreshape[2] = {3, 1};
    int a2d[2] = {2, 2}, b2d[2] = {3, 1};
    float v2[2] = {1, 2}, va[4] = {1, 2, 3, 4}, vb[3] = {1, 2, 3};
    ag_tensor* x = make_ag(1, d2, v2, 1);
    ag_tensor* a = make_ag(2, a2d, va, 1);
    ag_tensor* b = make_ag(2, b2d, vb, 1);
    CHECK(ag_add(NULL, x) == NULL);
    CHECK(ag_sub(x, NULL) == NULL);
    CHECK(ag_mul(NULL, NULL) == NULL);
    CHECK(ag_div(x, NULL) == NULL);
    CHECK(ag_neg(NULL) == NULL);
    CHECK(ag_exp(NULL) == NULL);
    CHECK(ag_log(NULL) == NULL);
    CHECK(ag_pow(NULL, 2) == NULL);
    CHECK(ag_sqrt(NULL) == NULL);
    CHECK(ag_relu(NULL) == NULL);
    CHECK(ag_sigmoid(NULL) == NULL);
    CHECK(ag_tanh(NULL) == NULL);
    CHECK(ag_gelu(NULL) == NULL);
    CHECK(ag_reshape(NULL, 1, d2) == NULL);
    CHECK(ag_reshape(x, 2, badreshape) == NULL);
    CHECK(ag_reshape(x, 1, bad0) == NULL);
    CHECK(ag_transpose(NULL, 0, 0) == NULL);
    CHECK(ag_transpose(x, -1, 0) == NULL);
    CHECK(ag_transpose(x, 0, 1) == NULL);
    CHECK(ag_slice(NULL, 0, 0, 1) == NULL);
    CHECK(ag_slice(x, 1, 0, 1) == NULL);
    CHECK(ag_slice(x, 0, 1, 1) == NULL);
    CHECK(ag_expand(NULL, 1, d2) == NULL);
    CHECK(ag_expand(x, 1, bad0) == NULL);
    CHECK(ag_sum(NULL, 0, 0) == NULL);
    CHECK(ag_mean(NULL, 0, 0) == NULL);
    CHECK(ag_max(NULL, 0, 0) == NULL);
    CHECK(ag_sum(x, 1, 0) == NULL);
    CHECK(ag_mean(x, -1, 0) == NULL);
    CHECK(ag_matmul(NULL, b) == NULL);
    CHECK(ag_matmul(a, NULL) == NULL);
    CHECK(ag_matmul(a, b) == NULL);
    ag_tensor_release(x); ag_tensor_release(a); ag_tensor_release(b);
    return current_failed;
}

typedef int (*test_fn)(void);
typedef struct { const char* name; test_fn fn; } test_case;

int main(void)
{
    const test_case tests[] = {
        {"construction, detach, retain/release", test_construction_detach_and_retain},
        {"binary ops and node ownership", test_binary_ops_and_node_ownership},
        {"graph omission and node retain/release", test_graph_omission_and_node_retain},
        {"unary operations", test_unary_ops},
        {"pow exact special cases", test_pow_special_cases},
        {"ReLU zero and NaN derivative", test_relu_zero_and_nan_contract},
        {"sqrt IEEE domains", test_sqrt_ieee_domain},
        {"broadcast forward/backward", test_broadcasting_forward_and_gradients},
        {"local backward ownership contract", test_local_backward_contract},
        {"reshape/transpose/slice/expand views", test_views_and_view_gradients},
        {"sum/mean/max reductions", test_reductions},
        {"matmul forward/backward", test_matmul},
        {"backward accumulation and zeroing", test_scalar_seeded_accumulation_and_zero},
        {"seed validation and transactionality", test_non_scalar_seed_requirement_and_transaction},
        {"stale graph transactional rejection", test_stale_graph_transactional_rejection},
        {"null and invalid arguments", test_invalid_operation_arguments}
    };
    int passed = 0, failed = 0;
    size_t i;
    for (i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        current_failed = 0;
        printf("[ RUN      ] %s\n", tests[i].name);
        tests[i].fn();
        if (current_failed) {
            ++failed;
            printf("[  FAILED  ] %s\n", tests[i].name);
        } else {
            ++passed;
            printf("[       OK ] %s\n", tests[i].name);
        }
    }
    printf("\n%d passed, %d failed, %d assertions\n", passed, failed, assertions);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
