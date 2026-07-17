#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "../include/tensor.h"

int tensor_checked_numel(int ndim, const int* dims, size_t* result) {
    if (result == NULL || ndim < 0 || (ndim > 0 && dims == NULL)) return 0;

    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        /* Zero-sized dimensions are intentionally unsupported. */
        if (dims[i] <= 0) return 0;
        if (total > (size_t)INT_MAX / (size_t)dims[i]) return 0;
        total *= (size_t)dims[i];
    }

    *result = total;
    return 1;
}

int tensor_has_valid_shape(const tensor* t) {
    size_t unused;
    return t != NULL && tensor_checked_numel(t->ndim, t->dims, &unused);
}

int tensor_has_valid_layout(const tensor* t) {
    size_t unused;
    if (t == NULL || !tensor_checked_numel(t->ndim, t->dims, &unused)) return 0;
    if (t->ndim > 0 && t->strides == NULL) return 0;
    if (t->offset < 0) return 0;
    for (int i = 0; i < t->ndim; ++i) {
        if (t->strides[i] <= 0) return 0;
    }
    return 1;
}

int tensor_has_valid_metadata(const tensor* t) {
    if (!tensor_has_valid_layout(t)) return 0;
    if (t->storage == NULL || t->storage->data == NULL) return 0;
    if (t->storage->ref_count <= 0 || t->storage->size <= 0) return 0;
    if ((size_t)t->offset >= (size_t)t->storage->size) return 0;

    size_t max_index = (size_t)t->offset;
    for (int i = 0; i < t->ndim; ++i) {
        size_t step_count = (size_t)(t->dims[i] - 1);
        size_t stride = (size_t)t->strides[i];
        if (step_count != 0 && stride > SIZE_MAX / step_count) return 0;
        size_t axis_span = step_count * stride;
        if (max_index > SIZE_MAX - axis_span) return 0;
        max_index += axis_span;
    }

    return max_index < (size_t)t->storage->size;
}

int tensor_copy_metadata(int ndim, const int* dims, const int* strides, int** out_dims, int** out_strides) {
    if (out_dims == NULL || out_strides == NULL) return 1;
    *out_dims = NULL;
    *out_strides = NULL;

    if (ndim < 0) return 1;
    if (ndim == 0) return 0;
    if (dims == NULL || strides == NULL) return 1;

    size_t metadata_bytes = (size_t)ndim * sizeof(int);
    *out_dims = (int*)malloc(metadata_bytes);
    *out_strides = (int*)malloc(metadata_bytes);
    if (*out_dims == NULL || *out_strides == NULL) {
        free(*out_dims);
        free(*out_strides);
        *out_dims = NULL;
        *out_strides = NULL;
        return 1;
    }

    for (int i = 0; i < ndim; ++i) {
        (*out_dims)[i] = dims[i];
        (*out_strides)[i] = strides[i];
    }
    return 0;
}

void calc_strides(int ndim, const int* dims, int* strides) {
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
    if (!tensor_has_valid_layout(t) || (t->ndim > 0 && coords == NULL)) return 0;
    int flat_idx = t->offset;
    for (int i = 0; i < t->ndim; i++) flat_idx += coords[i] * t->strides[i];
    return flat_idx;
}

int same_shape(tensor* a, tensor* b) {
    if (a == NULL || b == NULL) return 0;
    if (!tensor_has_valid_shape(a) || !tensor_has_valid_shape(b)) return 0;
    if (a->ndim != b->ndim) return 0;
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
    size_t unused;
    if (coords == NULL || !tensor_checked_numel(ndim, dims, &unused) || ndim <= 0) return;
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i]++;
        if (coords[i] < dims[i]) break;
        coords[i] = 0;
    }
}

int is_contiguous(tensor* t) {
    if (!tensor_has_valid_layout(t)) return 0;
    size_t expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (expected_stride > (size_t)INT_MAX ||
            t->strides[i] != (int)expected_stride) return 0;
        expected_stride *= (size_t)t->dims[i];
    }
    return 1;
}

int tensor_numel(tensor* t) {
    size_t total;
    if (t == NULL || !tensor_checked_numel(t->ndim, t->dims, &total)) return 0;
    return (int)total;
}
