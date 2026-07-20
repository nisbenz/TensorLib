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

static int broadcast_output_shape(const tensor* a,
                                  const tensor* b,
                                  int* output_ndim,
                                  int** output_dims) {
    if (a == NULL || b == NULL || output_ndim == NULL || output_dims == NULL) {
        return 0;
    }

    *output_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    *output_dims = NULL;

    if (*output_ndim == 0) return 1;

    int* dims = (int*)malloc((size_t)*output_ndim * sizeof(int));
    if (dims == NULL) return 0;

    for (int output_axis = 0; output_axis < *output_ndim; ++output_axis) {
        int a_axis = output_axis - (*output_ndim - a->ndim);
        int b_axis = output_axis - (*output_ndim - b->ndim);
        int a_dim = (a_axis < 0) ? 1 : a->dims[a_axis];
        int b_dim = (b_axis < 0) ? 1 : b->dims[b_axis];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            free(dims);
            return 0;
        }

        dims[output_axis] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    *output_dims = dims;
    return 1;
}

static int input_index_for_broadcast(const tensor* input,
                                     const int* output_coords,
                                     int output_ndim) {
    int index = input->offset;
    int rank_offset = output_ndim - input->ndim;

    for (int output_axis = 0; output_axis < output_ndim; ++output_axis) {
        int input_axis = output_axis - rank_offset;
        if (input_axis < 0) continue;

        int coordinate = (input->dims[input_axis] == 1)
                       ? 0
                       : output_coords[output_axis];
        index += coordinate * input->strides[input_axis];
    }

    return index;
}

static tensor* apply_binary(tensor* a,
                            tensor* b,
                            binary_fn scalar_op,
                            contiguous_binary_fn contiguous_op,
                            const char* op_name) {
    if (!tensor_has_valid_metadata(a) || !tensor_has_valid_metadata(b)) return NULL;

    int output_ndim = 0;
    int* output_dims = NULL;
    if (!broadcast_output_shape(a, b, &output_ndim, &output_dims)) {
        fprintf(stderr, "ERROR: Tensors have incompatible shapes to %s.\n", op_name);
        return NULL;
    }

    tensor* c = t_alloc(output_ndim, output_dims);
    free(output_dims);
    if (c == NULL) {
        return NULL;
    }

    int total_elements = tensor_numel(c);
    if (same_shape(a, b) == 1 &&
        same_shape(a, c) == 1 &&
        same_stride(a, b) == 1 &&
        same_stride(a, c) == 1) {
        contiguous_op(c->storage->data + c->offset,
                      a->storage->data + a->offset,
                      b->storage->data + b->offset,
                      total_elements);
    } else {
        int* coords = NULL;
        if (output_ndim > 0) {
            coords = (int*)calloc((size_t)output_ndim, sizeof(int));
        }
        if (output_ndim > 0 && coords == NULL) {
            t_free(c);
            return NULL;
        }

        for (int i = 0; i < total_elements; i++) {
            int idx_a = input_index_for_broadcast(a, coords, output_ndim);
            int idx_b = input_index_for_broadcast(b, coords, output_ndim);
            int idx_c = get_flat_index_nd(c, coords);
            c->storage->data[idx_c] = scalar_op(a->storage->data[idx_a], b->storage->data[idx_b]);
            advance_coords(coords, c->dims, c->ndim);
        }
        free(coords);
    }

    return c;
}

tensor* t_add(tensor* a, tensor* b) { return apply_binary(a, b, op_add, add_contiguous, "add"); }
tensor* t_sub(tensor* a, tensor* b) { return apply_binary(a, b, op_sub, sub_contiguous, "subtract"); }
tensor* t_mul(tensor* a, tensor* b) { return apply_binary(a, b, op_mul, mul_contiguous, "multiply"); }
tensor* t_div(tensor* a, tensor* b) { return apply_binary(a, b, op_div, div_contiguous, "divide"); }

static tensor* apply_scalar_binary(tensor* a,
                                   float scalar,
                                   binary_fn scalar_op) {
    if (!tensor_has_valid_metadata(a) || scalar_op == NULL) return NULL;

    tensor* out = t_alloc(a->ndim, a->dims);
    if (out == NULL) return NULL;

    int total_elements = tensor_numel(a);
    if (is_contiguous(a) && is_contiguous(out)) {
        for (int i = 0; i < total_elements; ++i) {
            out->storage->data[out->offset + i] =
                scalar_op(a->storage->data[a->offset + i], scalar);
        }
        return out;
    }

    int* coords = NULL;
    if (a->ndim > 0) {
        coords = (int*)calloc((size_t)a->ndim, sizeof(int));
        if (coords == NULL) {
            t_free(out);
            return NULL;
        }
    }

    for (int i = 0; i < total_elements; ++i) {
        int input_index = get_flat_index_nd(a, coords);
        int output_index = get_flat_index_nd(out, coords);
        out->storage->data[output_index] =
            scalar_op(a->storage->data[input_index], scalar);
        advance_coords(coords, a->dims, a->ndim);
    }

    free(coords);
    return out;
}

tensor* t_add_scalar(tensor* a, float scalar) {
    return apply_scalar_binary(a, scalar, op_add);
}

tensor* t_sub_scalar(tensor* a, float scalar) {
    return apply_scalar_binary(a, scalar, op_sub);
}

tensor* t_mul_scalar(tensor* a, float scalar) {
    return apply_scalar_binary(a, scalar, op_mul);
}

tensor* t_div_scalar(tensor* a, float scalar) {
    return apply_scalar_binary(a, scalar, op_div);
}

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
