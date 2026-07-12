#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include "../include/tensor.h"

void calc_strides(int ndim, int* dims, int* strides) {
    if (ndim <= 0) return;
    if (dims == NULL || strides == NULL) return;

    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (dims[i] <= 0 || stride > (size_t)INT_MAX) {
            for (int j = 0; j < ndim; ++j) strides[j] = 0;
            return;
        }
        strides[i] = (int)stride;
        if (i > 0 && stride > (size_t)INT_MAX / (size_t)dims[i]) {
            for (int j = 0; j < ndim; ++j) strides[j] = 0;
            return;
        }
        stride *= (size_t)dims[i];
    }
}

int get_flat_index_nd(tensor* t, int* coords) {
    if (t == NULL || (t->ndim > 0 && coords == NULL)) return 0;
    int flat_idx = t->offset;
    for (int i = 0; i < t->ndim; i++) flat_idx += coords[i] * t->strides[i];
    return flat_idx;
}

int same_shape(tensor* a, tensor* b) {
    if (a == NULL || b == NULL || a->ndim < 0 || b->ndim < 0) return 0;
    if (a->ndim != b->ndim) return 0;
    if (a->ndim > 0 && (a->dims == NULL || b->dims == NULL)) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i]) return 0;
    }
    return 1;
}

int same_stride(tensor* a, tensor* b) {
    if (a == NULL || b == NULL || a->ndim < 0 || b->ndim < 0) return 0;
    if (a->ndim != b->ndim) return 0;
    if (a->ndim > 0 && (a->strides == NULL || b->strides == NULL)) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->strides[i] != b->strides[i]) return 0;
    }
    return 1;
}

void advance_coords(int* coords, const int* dims, int ndim) {
    if (coords == NULL || dims == NULL || ndim <= 0) return;
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i]++;
        if (coords[i] < dims[i]) break;
        coords[i] = 0;
    }
}

int is_contiguous(tensor* t) {
    if (t == NULL || tensor_numel(t) == 0) return 0;
    size_t expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (expected_stride > (size_t)INT_MAX ||
            t->strides[i] != (int)expected_stride) return 0;
        expected_stride *= (size_t)t->dims[i];
    }
    return 1;
}

int tensor_numel(tensor* t) {
    if (t == NULL || t->ndim < 0 || (t->ndim > 0 && t->dims == NULL)) return 0;
    size_t total = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->dims[i] <= 0 || total > (size_t)INT_MAX / (size_t)t->dims[i]) return 0;
        total *= (size_t)t->dims[i];
    }
    return (int)total;
}