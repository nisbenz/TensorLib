#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include <math.h>
// In your tensor/storage struct definition, or just at the point of use:
static inline void add_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
static inline void sub_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] - b[i];
    }
}
static inline void mul_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}
static inline void div_contiguous(float* restrict c,
                                  const float* restrict a,
                                  const float* restrict b,
                                  int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] / b[i];
    }
}
tensor* t_add(tensor* a, tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (same_shape(a, b) == 0) {
        fprintf(stderr, "ERROR: Tensors must have identical shapes to add.\n");
        return NULL;
    }

    tensor* c = (tensor*)malloc(sizeof(tensor));
    if (c == NULL) return NULL;
    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = 0;

    if (init_t(c, a) != 0) {
        t_free(c);
        return NULL;
    }

    int total_elements = c->storage->size;

    if (same_stride(a, b) == 1 && same_stride(a, c) == 1) {
        add_contiguous(c->storage->data, a->storage->data, b->storage->data, total_elements);
    } else {
        int* coords = (int*)calloc(a->ndim, sizeof(int));
        if (coords == NULL) {
            t_free(c);
            return NULL;
        }

        for (int i = 0; i < total_elements; i++) {
            int idx_a = get_flat_index_nd(a, coords);
            int idx_b = get_flat_index_nd(b, coords);
            int idx_c = get_flat_index_nd(c, coords);

            c->storage->data[idx_c] = a->storage->data[idx_a] + b->storage->data[idx_b];

            advance_coords(coords, a->dims, a->ndim);
        }
        free(coords);
    }

    return c;
}
tensor* t_sub(tensor* a, tensor* b){
    if (a == NULL || b == NULL) return NULL;
    if (same_shape(a, b) == 0) {
        fprintf(stderr, "ERROR: Tensors must have identical shapes to subtract.\n");
        return NULL;
    }
    tensor* c = (tensor*)malloc(sizeof(tensor));
    if (c == NULL) return NULL;
    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = 0;

    if (init_t(c, a) != 0) {
        t_free(c);
        return NULL;
    }

    int total_elements = c->storage->size;
    if (same_stride(a, b) == 1 && same_stride(a, c) == 1) {
        sub_contiguous(c->storage->data, a->storage->data, b->storage->data, total_elements);
    }else {  int* coords = (int*)calloc(a->ndim, sizeof(int));

        if (coords == NULL) {
            t_free(c);
            return NULL;
        }

        for (int i = 0; i < total_elements; i++) {
            int idx_a = get_flat_index_nd(a, coords);
            int idx_b = get_flat_index_nd(b, coords);
            int idx_c = get_flat_index_nd(c, coords);

            c->storage->data[idx_c] = a->storage->data[idx_a] - b->storage->data[idx_b];

            advance_coords(coords, a->dims, a->ndim);
        }
        free(coords);

    }
    return c;
}


tensor* t_mul(tensor* a, tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (same_shape(a, b) == 0) {
        fprintf(stderr, "ERROR: Tensors must have identical shapes to multiply.\n");
        return NULL;
    }

    tensor* c = (tensor*)malloc(sizeof(tensor));
    if (c == NULL) return NULL;
    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = 0;

    if (init_t(c, a) != 0) {
        t_free(c);
        return NULL;
    }

    int total_elements = c->storage->size;

    if (same_stride(a, b) == 1 && same_stride(a, c) == 1) {
        mul_contiguous(c->storage->data, a->storage->data, b->storage->data, total_elements);
    } else {
        int* coords = (int*)calloc(a->ndim, sizeof(int));
        if (coords == NULL) {
            t_free(c);
            return NULL;
        }

        for (int i = 0; i < total_elements; i++) {
            int idx_a = get_flat_index_nd(a, coords);
            int idx_b = get_flat_index_nd(b, coords);
            int idx_c = get_flat_index_nd(c, coords);

            c->storage->data[idx_c] = a->storage->data[idx_a] * b->storage->data[idx_b];

            advance_coords(coords, a->dims, a->ndim);
        }
        free(coords);
    }

    return c;
}


tensor* t_div(tensor* a, tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (same_shape(a, b) == 0) {
        fprintf(stderr, "ERROR: Tensors must have identical shapes to divide.\n");
        return NULL;
    }

    tensor* c = (tensor*)malloc(sizeof(tensor));
    if (c == NULL) return NULL;
    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = 0;

    if (init_t(c, a) != 0) {
        t_free(c);
        return NULL;
    }

    int total_elements = c->storage->size;

    if (same_stride(a, b) == 1 && same_stride(a, c) == 1) {
        div_contiguous(c->storage->data, a->storage->data, b->storage->data, total_elements);
    } else {
        int* coords = (int*)calloc(a->ndim, sizeof(int));
        if (coords == NULL) {
            t_free(c);
            return NULL;
        }

        for (int i = 0; i < total_elements; i++) {
            int idx_a = get_flat_index_nd(a, coords);
            int idx_b = get_flat_index_nd(b, coords);
            int idx_c = get_flat_index_nd(c, coords);

            c->storage->data[idx_c] = a->storage->data[idx_a] / b->storage->data[idx_b];

            advance_coords(coords, a->dims, a->ndim);
        }
        free(coords);
    }

    return c;
}
tensor* t_exp(tensor* t) {
    tensor* contig_t = t;
    int is_temp_view = 0;

    if (!is_contiguous(t)) {
        contig_t = t_contiguous(t);
        is_temp_view = 1;
    }

    tensor* out = t_alloc(contig_t->ndim, contig_t->dims);

    for (int i = 0; i < out->storage->size; ++i) {
        float x = contig_t->storage->data[i];
        out->storage->data[i] = expf(x);
    }

    if (is_temp_view) {
        t_free(contig_t);
    }

    return out;
}
tensor* t_log(tensor* t) {
    tensor* contig_t = t;
    int is_temp_view = 0;

    if (!is_contiguous(t)) {
        contig_t = t_contiguous(t);
        is_temp_view = 1;
    }

    tensor* out = t_alloc(contig_t->ndim, contig_t->dims);

    for (int i = 0; i < out->storage->size; ++i) {
        float x = contig_t->storage->data[i];
        out->storage->data[i] = logf(x);
    }

    if (is_temp_view) {
        t_free(contig_t);
    }

    return out;
}
tensor* t_relu(tensor* t) {
    tensor* contig_t = t;
    int is_temp_view = 0;

    if (!is_contiguous(t)) {
        contig_t = t_contiguous(t);
        is_temp_view = 1;
    }

    tensor* out = t_alloc(contig_t->ndim, contig_t->dims);

    for (int i = 0; i < out->storage->size; ++i) {
        float x = contig_t->storage->data[i];
        if (x < 0) {
            out->storage->data[i] = 0;
        }
    }

    if (is_temp_view) {
        t_free(contig_t);
    }

    return out;
}
tensor* t_tanh(tensor* t) {
    tensor* contig_t = t;
    int is_temp_view = 0;

    if (!is_contiguous(t)) {
        contig_t = t_contiguous(t);
        is_temp_view = 1;
    }

    tensor* out = t_alloc(contig_t->ndim, contig_t->dims);

    for (int i = 0; i < out->storage->size; ++i) {
        float x = contig_t->storage->data[i];
        out->storage->data[i] = tanhf(x);
    }

    if (is_temp_view) {
        t_free(contig_t);
    }

    return out;
}
tensor* t_sigmoid(tensor* t) {
    tensor* contig_t = t;
    int is_temp_view = 0;

    if (!is_contiguous(t)) {
        contig_t = t_contiguous(t);
        is_temp_view = 1;
    }

    tensor* out = t_alloc(contig_t->ndim, contig_t->dims);

    for (int i = 0; i < out->storage->size; ++i) {
        float x = contig_t->storage->data[i];
        out->storage->data[i] = 1/(1+ expf(-x));
    }

    if (is_temp_view) {
        t_free(contig_t);
    }

    return out;
}
tensor* t_pow(tensor* t, float exponent) {
    tensor* contig_t = t;
    int is_temp_view = 0;

    if (!is_contiguous(t)) {
        contig_t = t_contiguous(t);
        is_temp_view = 1;
    }

    tensor* out = t_alloc(contig_t->ndim, contig_t->dims);

    for (int i = 0; i < out->storage->size; ++i) {
        float x = contig_t->storage->data[i];
        out->storage->data[i] = powf(x,exponent);
    }

    if (is_temp_view) {
        t_free(contig_t);
    }

    return out;
}