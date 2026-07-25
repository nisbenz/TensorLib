#include <stdlib.h>
#include "../../include/tensorlib/tensor.h"

void add_ref_count(Storage* a, tensor* b) {
    if (a != NULL && b != NULL) {
        a->ref_count++;
        b->storage = a;
    }
}

Storage* s_alloc(int ndim, const int* dims) {
    size_t count;
    if (!tensor_checked_numel(ndim, dims, &count)) return NULL;

    Storage* s = (Storage*)malloc(sizeof(Storage));
    if (s == NULL) return NULL;
    s->ref_count = 1;
    s->size = (int)count;
    s->version = 0;
    s->data = (float*)malloc(count * sizeof(float));
    if (s->data == NULL) {
        free(s);
        return NULL;
    }
    return s;
}

tensor* t_alloc(int ndim, const int* dims) {
    size_t count;
    if (!tensor_checked_numel(ndim, dims, &count)) return NULL;
    (void)count;

    tensor* a = (tensor*)calloc(1, sizeof(tensor));
    if (a == NULL) return NULL;
    a->ndim = ndim;

    if (ndim > 0) {
        int* strides = (int*)malloc((size_t)ndim * sizeof(int));
        if (strides == NULL) {
            t_free(a);
            return NULL;
        }
        calc_strides(ndim, dims, strides);
        if (tensor_copy_metadata(ndim, dims, strides, &a->dims, &a->strides) != 0) {
            free(strides);
            t_free(a);
            return NULL;
        }
        free(strides);
    }

    a->storage = s_alloc(ndim, dims);
    if (a->storage == NULL) {
        t_free(a);
        return NULL;
    }
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
    if (c == NULL || !tensor_has_valid_shape(ref)) return 1;
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
    c->storage->version = 0;
    c->storage->data = (float*)calloc((size_t)total_elements, sizeof(float));
    if (c->storage->data == NULL) {
        free(c->storage);
        c->storage = NULL;
        return 1;
    }

    if (ref->ndim > 0) {
        int* strides = (int*)malloc((size_t)ref->ndim * sizeof(int));
        if (strides == NULL) {
            free(c->storage->data);
            free(c->storage);
            c->storage = NULL;
            return 1;
        }
        calc_strides(ref->ndim, ref->dims, strides);
        if (tensor_copy_metadata(ref->ndim, ref->dims, strides, &c->dims, &c->strides) != 0) {
            free(strides);
            free(c->storage->data);
            free(c->storage);
            c->storage = NULL;
            return 1;
        }
        free(strides);
    }
    return 0;
}

tensor* t_clone(tensor* t) {
    if (!tensor_has_valid_metadata(t)) return NULL;
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

void tensor_mark_modified(tensor* value) {
    if (value == NULL || value->storage == NULL) return;
    value->storage->version++;
}
