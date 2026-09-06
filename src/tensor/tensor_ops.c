#include <stdio.h>
#include <stdlib.h>
#include "../../include/tensorlib/tensor.h"
#include <math.h>
#include "parallel.h"

/* Element-wise loops only benefit from OpenMP once the working set exceeds a
 * single core's share of memory bandwidth (i.e. beyond the CPU cache). Below
 * this element count a single thread is already at peak and forking only adds
 * overhead, so the op stays serial for small tensors. */
#ifndef TENSORLIB_OP_MIN_PARALLEL_ELEMENTS
#define TENSORLIB_OP_MIN_PARALLEL_ELEMENTS (1 << 22)
#endif

typedef float (*binary_fn)(float, float);
typedef void (*contiguous_binary_fn)(float* restrict, const float* restrict, const float* restrict, int);

/* Run a contiguous binary op in equal, disjoint slices across N OpenMP threads
 * (or in one slice when there is nothing to parallelize). The slice offsets
 * require a contiguous-linear layout, which callers must already guarantee. */
static void contiguous_binary_chunked(contiguous_binary_fn op,
                                      float* restrict c,
                                      const float* restrict a,
                                      const float* restrict b,
                                      int n,
                                      int threads) {
    if (threads <= 1) {
        op(c, a, b, n);
        return;
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int tn = omp_get_num_threads();
        int start = (int)(((long long)tid * n) / tn);
        int end = (int)(((long long)(tid + 1) * n) / tn);
        op(c + start, a + start, b + start, end - start);
    }
#endif
}

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

static int broadcast_step(const tensor* input, int output_axis, int output_ndim)
{
    int input_axis = output_axis - (output_ndim - input->ndim);
    if (input_axis < 0 || input->dims[input_axis] == 1) return 0;
    return input->strides[input_axis];
}

/* Fast path for element-wise binary ops where the output is contiguous and each
 * operand is a contiguous right-aligned prefix of the output shape with every
 * dimension either equal to the output's or equal to 1 (broadcast). This covers
 * the LayerNorm/linear/attention patterns (keepdim reductions, per-channel
 * weight/bias) and replaces per-element coordinate index math with a streaming
 * row loop. Returns 1 when it handled the operation. */
static int try_contiguous_broadcast(tensor* a, tensor* b,
                                    tensor* c, binary_fn scalar_op) {
    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(c)) return 0;

    int ro_a = c->ndim - a->ndim;
    int ro_b = c->ndim - b->ndim;
    if (ro_a < 0 || ro_b < 0) return 0;

    /* Each operand must be a right-aligned prefix with dims that are either
     * equal to the output's or 1; track the last broadcast axis across both.
     * A full-shaped operand has no broadcast axis (-1) and is allowed. */
    int la = -1;
    int lb = -1;
    for (int i = 0; i < c->ndim; ++i) {
        int bi_a = i - ro_a;
        int bi_b = i - ro_b;
        int da = (bi_a < 0) ? 1 : a->dims[bi_a];
        int db = (bi_b < 0) ? 1 : b->dims[bi_b];
        if (da != 1 && da != c->dims[i]) return 0;
        if (db != 1 && db != c->dims[i]) return 0;
        if (da == 1 && c->dims[i] > 1) la = i;
        if (db == 1 && c->dims[i] > 1) lb = i;
    }
    int s = la > lb ? la : lb;
    if (s < 0) return 0; /* both full shaped: handled by the same-shape path */

    /* No singleton broadcast may appear inside the contiguous inner region. */
    for (int i = s + 1; i < c->ndim; ++i) {
        int bi_a = i - ro_a;
        int bi_b = i - ro_b;
        int da = (bi_a < 0) ? 1 : a->dims[bi_a];
        int db = (bi_b < 0) ? 1 : b->dims[bi_b];
        if ((da == 1 && c->dims[i] > 1) || (db == 1 && c->dims[i] > 1)) return 0;
    }

    int outer = 1;
    int block = 1;
    for (int i = 0; i < c->ndim; ++i) {
        if (i <= s) outer *= c->dims[i];
        else block *= c->dims[i];
    }
    if (outer <= 0 || block <= 0) return 0;

    const float* a_data = a->storage->data + a->offset;
    const float* b_data = b->storage->data + b->offset;
    float* c_data = c->storage->data + c->offset;
    int threads = tensorlib_parallel_threads(
        (long long)outer * block, TENSORLIB_OP_MIN_PARALLEL_ELEMENTS, outer);

#ifdef _OPENMP
#pragma omp parallel for if(threads > 1) schedule(static) num_threads(threads)
#endif
    for (int r = 0; r < outer; ++r) {
        /* Row-major decomposition of r over the outer axes 0..s. */
        int remaining = r;
        int base_c = 0;
        int prefix_a = 0;
        int prefix_b = 0;
        for (int i = s; i >= 0; --i) {
            int bi_a = i - ro_a;
            int bi_b = i - ro_b;
            int coord = remaining % c->dims[i];
            remaining /= c->dims[i];
            base_c += coord * c->strides[i];
            if (bi_a >= 0 && a->dims[bi_a] != 1) prefix_a += coord * a->strides[bi_a];
            if (bi_b >= 0 && b->dims[bi_b] != 1) prefix_b += coord * b->strides[bi_b];
        }
        for (int k = 0; k < block; ++k) {
            c_data[base_c + k] = scalar_op(a_data[prefix_a + k],
                                           b_data[prefix_b + k]);
        }
    }
    return 1;
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
        contiguous_binary_chunked(contiguous_op,
                                  c->storage->data + c->offset,
                                  a->storage->data + a->offset,
                                  b->storage->data + b->offset,
                                  total_elements,
                                  tensorlib_parallel_threads(total_elements, TENSORLIB_OP_MIN_PARALLEL_ELEMENTS, 0));
    } else if (try_contiguous_broadcast(a, b, c, scalar_op)) {
        /* Handled by the streaming broadcast fast path. */
    } else {
        int* coords = NULL;
        int* steps_a = NULL;
        int* steps_b = NULL;
        int idx_a = a->offset;
        int idx_b = b->offset;
        int idx_c = c->offset;
        if (output_ndim > 0) {
            coords = (int*)calloc((size_t)output_ndim, sizeof(int));
            steps_a = (int*)malloc((size_t)output_ndim * sizeof(int));
            steps_b = (int*)malloc((size_t)output_ndim * sizeof(int));
        }
        if (output_ndim > 0 && (coords == NULL || steps_a == NULL ||
                                steps_b == NULL)) {
            free(steps_b);
            free(steps_a);
            free(coords);
            t_free(c);
            return NULL;
        }
        for (int axis = 0; axis < output_ndim; ++axis) {
            steps_a[axis] = broadcast_step(a, axis, output_ndim);
            steps_b[axis] = broadcast_step(b, axis, output_ndim);
        }

        for (int i = 0; i < total_elements; i++) {
            c->storage->data[idx_c] = scalar_op(a->storage->data[idx_a], b->storage->data[idx_b]);
            for (int axis = output_ndim - 1; axis >= 0; --axis) {
                coords[axis]++;
                idx_a += steps_a[axis];
                idx_b += steps_b[axis];
                idx_c += c->strides[axis];
                if (coords[axis] < c->dims[axis]) break;
                coords[axis] = 0;
                idx_a -= c->dims[axis] * steps_a[axis];
                idx_b -= c->dims[axis] * steps_b[axis];
                idx_c -= c->dims[axis] * c->strides[axis];
            }
        }
        free(steps_b);
        free(steps_a);
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
        float* out_data = out->storage->data + out->offset;
        const float* in_data = a->storage->data + a->offset;
        int threads = tensorlib_parallel_threads(total_elements, TENSORLIB_OP_MIN_PARALLEL_ELEMENTS, 0);
        if (threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
            {
                int tid = omp_get_thread_num();
                int tn = omp_get_num_threads();
                int start = (int)(((long long)tid * total_elements) / tn);
                int end = (int)(((long long)(tid + 1) * total_elements) / tn);
                for (int i = start; i < end; ++i) {
                    out_data[i] = scalar_op(in_data[i], scalar);
                }
            }
#endif
        } else {
            for (int i = 0; i < total_elements; ++i) {
                out_data[i] = scalar_op(in_data[i], scalar);
            }
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

    int total_elements = tensor_numel(contiguous);
    float* out_data = out->storage->data;
    const float* in_data = contiguous->storage->data + contiguous->offset;
    int threads = tensorlib_parallel_threads(total_elements, TENSORLIB_OP_MIN_PARALLEL_ELEMENTS, 0);
    if (threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            int tn = omp_get_num_threads();
            int start = (int)(((long long)tid * total_elements) / tn);
            int end = (int)(((long long)(tid + 1) * total_elements) / tn);
            for (int i = start; i < end; ++i) {
                out_data[i] = fn(in_data[i]);
            }
        }
#endif
    } else {
        for (int i = 0; i < total_elements; ++i) {
            out_data[i] = fn(in_data[i]);
        }
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

    int total_elements = tensor_numel(t);
    float* out_data = out->storage->data;
    const float* in_data = contiguous->storage->data + contiguous->offset;
    int threads = tensorlib_parallel_threads(total_elements, TENSORLIB_OP_MIN_PARALLEL_ELEMENTS, 0);
    if (threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            int tn = omp_get_num_threads();
            int start = (int)(((long long)tid * total_elements) / tn);
            int end = (int)(((long long)(tid + 1) * total_elements) / tn);
            for (int i = start; i < end; ++i) {
                out_data[i] = powf(in_data[i], exponent);
            }
        }
#endif
    } else {
        for (int i = 0; i < total_elements; ++i) {
            out_data[i] = powf(in_data[i], exponent);
        }
    }

    t_free(contiguous);
    return out;
}
