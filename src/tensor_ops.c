#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include <math.h>

typedef float (*binary_fn)(float, float);
typedef void (*contiguous_binary_fn)(float* restrict, const float* restrict, const float* restrict, int);

static inline void add_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

static inline void sub_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] - b[i];
}

static inline void mul_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] * b[i];
}

static inline void div_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] / b[i];
}

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }
static float op_div(float a, float b) { return a / b; }

static tensor* apply_binary(tensor* a,
                            tensor* b,
                            binary_fn scalar_op,
                            contiguous_binary_fn contiguous_op,
                            const char* op_name) {
    if (!tensor_has_valid_metadata(a) || !tensor_has_valid_metadata(b)) return NULL;
    if (same_shape(a, b) == 0) {
        fprintf(stderr, "ERROR: Tensors must have identical shapes to %s.\n", op_name);
        return NULL;
    }

    tensor* c = (tensor*)calloc(1, sizeof(tensor));
    if (c == NULL) return NULL;

    if (init_t(c, a) != 0) {
        t_free(c);
        return NULL;
    }

    int total_elements = c->storage->size;
    if (same_stride(a, b) == 1 && same_stride(a, c) == 1) {
        contiguous_op(c->storage->data + c->offset,
                      a->storage->data + a->offset,
                      b->storage->data + b->offset,
                      total_elements);
    } else {
        int* coords = (int*)calloc((size_t)a->ndim, sizeof(int));
        if (coords == NULL) {
            t_free(c);
            return NULL;
        }

        for (int i = 0; i < total_elements; i++) {
            int idx_a = get_flat_index_nd(a, coords);
            int idx_b = get_flat_index_nd(b, coords);
            int idx_c = get_flat_index_nd(c, coords);
            c->storage->data[idx_c] = scalar_op(a->storage->data[idx_a], b->storage->data[idx_b]);
            advance_coords(coords, a->dims, a->ndim);
        }
        free(coords);
    }

    return c;
}

tensor* t_add(tensor* a, tensor* b) { return apply_binary(a, b, op_add, add_contiguous, "add"); }
tensor* t_sub(tensor* a, tensor* b) { return apply_binary(a, b, op_sub, sub_contiguous, "subtract"); }
tensor* t_mul(tensor* a, tensor* b) { return apply_binary(a, b, op_mul, mul_contiguous, "multiply"); }
tensor* t_div(tensor* a, tensor* b) { return apply_binary(a, b, op_div, div_contiguous, "divide"); }

typedef float (*unary_fn)(float);

static tensor* apply_unary(tensor* input, unary_fn fn) {
    if (input == NULL || fn == NULL) return NULL;

    tensor* contiguous = t_contiguous(input);
    if (contiguous == NULL) return NULL;

    tensor* out = t_alloc(contiguous->ndim, contiguous->dims);
    if (out == NULL) {
        t_free(contiguous);
        return NULL;
    }

    for (int i = 0; i < out->storage->size; ++i) {
        out->storage->data[i] = fn(contiguous->storage->data[contiguous->offset + i]);
    }

    t_free(contiguous);
    return out;
}

static float op_exp(float x) { return expf(x); }
static float op_log(float x) { return logf(x); }
static float op_relu(float x) { return x < 0.0f ? 0.0f : x; }
static float op_tanh(float x) { return tanhf(x); }
static float op_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float op_neg(float x) { return -x; }
static float op_sqrt(float x) { return sqrtf(x); }
static float op_gelu(float x) {
    const float sqrt_two_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(sqrt_two_over_pi *
                         (x + 0.044715f * x * x * x)));
}

tensor* t_exp(tensor* t) { return apply_unary(t, op_exp); }
tensor* t_log(tensor* t) { return apply_unary(t, op_log); }
tensor* t_relu(tensor* t) { return apply_unary(t, op_relu); }
tensor* t_tanh(tensor* t) { return apply_unary(t, op_tanh); }
tensor* t_sigmoid(tensor* t) { return apply_unary(t, op_sigmoid); }
tensor* t_neg(tensor* t) { return apply_unary(t, op_neg); }
tensor* t_sqrt(tensor* t) { return apply_unary(t, op_sqrt); }
tensor* t_gelu(tensor* t) { return apply_unary(t, op_gelu); }

tensor* t_pow(tensor* t, float exponent) {
    if (t == NULL) return NULL;

    tensor* contiguous = t_contiguous(t);
    if (contiguous == NULL) return NULL;

    tensor* out = t_alloc(contiguous->ndim, contiguous->dims);
    if (out == NULL) {
        t_free(contiguous);
        return NULL;
    }

    for (int i = 0; i < out->storage->size; ++i) {
        out->storage->data[i] =
            powf(contiguous->storage->data[contiguous->offset + i], exponent);
    }

    t_free(contiguous);
    return out;
}
