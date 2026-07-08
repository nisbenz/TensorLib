#include "../include/tensor.h"
#include <stdlib.h>
void calc_strides(int ndim, int* dims, int* strides) {
    if (ndim > 0) {
        strides[ndim - 1] = 1;

        for (int i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }
}

int get_flat_index_nd(tensor* t, int* coords) {
    if (t == NULL || (t->ndim > 0 && coords == NULL)) return 0;
    int flat_idx = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        flat_idx += coords[i] * t->strides[i];
    }
    return flat_idx;
}

int same_shape(tensor* a, tensor* b) {
    if (a == NULL || b == NULL) return 0;
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (int i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i]) {
            return 0;
        }
    }
    return 1;
}

int same_stride(tensor* a, tensor* b) {
    if (a == NULL || b == NULL) return 0;
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (int i = 0; i < a->ndim; i++) {
        if (a->strides[i] != b->strides[i]) {
            return 0;
        }
    }
    return 1;
}

void advance_coords(int* coords, const int* dims, int ndim) {
    if (coords == NULL || dims == NULL) return;
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i]++;
        if (coords[i] < dims[i]) {
            break;
        } else {
            coords[i] = 0;
        }
    }
}

int is_contiguous(tensor* t) {
    if (t == NULL) return 0;
    int expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected_stride) {
            return 0;
        }
        expected_stride *= t->dims[i];
    }
    return 1;
}

int tensor_numel(tensor* t) {
    if (t == NULL) return 0;
    int total = 1;
    for (int i = 0; i < t->ndim; i++) {
        total *= t->dims[i];
    }
    return total;
}
