#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include "../include/tensor.h"

static int checked_numel(int ndim, const int* dims, size_t* result) {
    if (result == NULL || ndim < 0 || (ndim > 0 && dims == NULL)) return 0;

    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        /* Zero-sized dimensions are intentionally unsupported. */
        if (dims[i] <= 0) return 0;
        if (total > (size_t)INT_MAX / (size_t)dims[i]) return 0;
        total *= (size_t)dims[i];
    }
    if (total > SIZE_MAX / sizeof(float)) return 0;
    *result = total;
    return 1;
}

void add_ref_count(Storage* a, tensor* b) {
    if (a != NULL && b != NULL) {
        a->ref_count++;
        b->storage = a;
    }
}

Storage* s_alloc(int ndim, const int* dims) {
    size_t count;
    if (!checked_numel(ndim, dims, &count)) return NULL;

    Storage* s = (Storage*)malloc(sizeof(Storage));
    if (s == NULL) return NULL;
    s->ref_count = 1;
    s->size = (int)count;
    s->data = (float*)malloc(count * sizeof(float));
    if (s->data == NULL) {
        free(s);
        return NULL;
    }
    return s;
}

tensor* t_alloc(int ndim, const int* dims) {
    size_t count;
    if (!checked_numel(ndim, dims, &count)) return NULL;
    (void)count;

    tensor* a = (tensor*)calloc(1, sizeof(tensor));
    if (a == NULL) return NULL;
    a->ndim = ndim;

    if (ndim > 0) {
        size_t metadata_bytes = (size_t)ndim * sizeof(int);
        a->dims = (int*)malloc(metadata_bytes);
        a->strides = (int*)malloc(metadata_bytes);
        if (a->dims == NULL || a->strides == NULL) {
            t_free(a);
            return NULL;
        }
        for (int i = 0; i < ndim; ++i) a->dims[i] = dims[i];
    }

    a->storage = s_alloc(ndim, dims);
    if (a->storage == NULL) {
        t_free(a);
        return NULL;
    }
    calc_strides(ndim, a->dims, a->strides);
    return a;
}

void t_free(tensor* t) {
    if (t == NULL) return;
    if (t->storage != NULL) {
        if (t->storage->ref_count > 1) {
            t->storage->ref_count--;
        } else {
            free(t->storage->data);
            free(t->storage);
        }
    }
    free(t->dims);
    free(t->strides);
    free(t);
}

int init_t(tensor* c, tensor* ref) {
    if (c == NULL || ref == NULL) return 1;
    int total_elements = tensor_numel(ref);
    if (total_elements == 0) return 1;

    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = ref->ndim;
    c->offset = 0;

    c->storage = (Storage*)malloc(sizeof(Storage));
    if (c->storage == NULL) return 1;
    c->storage->ref_count = 1;
    c->storage->size = total_elements;
    c->storage->data = (float*)calloc((size_t)total_elements, sizeof(float));
    if (c->storage->data == NULL) {
        free(c->storage);
        c->storage = NULL;
        return 1;
    }

    if (ref->ndim > 0) {
        size_t metadata_bytes = (size_t)ref->ndim * sizeof(int);
        c->dims = (int*)malloc(metadata_bytes);
        c->strides = (int*)malloc(metadata_bytes);
        if (c->dims == NULL || c->strides == NULL) {
            free(c->dims);
            free(c->strides);
            free(c->storage->data);
            free(c->storage);
            c->dims = NULL;
            c->strides = NULL;
            c->storage = NULL;
            return 1;
        }
        for (int i = 0; i < ref->ndim; i++) c->dims[i] = ref->dims[i];
        calc_strides(c->ndim, c->dims, c->strides);
    }
    return 0;
}

tensor* t_clone(tensor* t) {
    if (t == NULL || tensor_numel(t) == 0) return NULL;
    tensor* a = t_alloc(t->ndim, t->dims);
    if (a == NULL) return NULL;

    int total_elements = tensor_numel(t);
    int* coords = NULL;
    if (t->ndim > 0) {
        coords = (int*)calloc((size_t)t->ndim, sizeof(int));
        if (coords == NULL) {
            t_free(a);
            return NULL;
        }
    }

    for (int i = 0; i < total_elements; i++) {
        int src_idx = get_flat_index_nd(t, coords);
        a->storage->data[i] = t->storage->data[src_idx];
        advance_coords(coords, t->dims, t->ndim);
    }
    free(coords);
    return a;
}