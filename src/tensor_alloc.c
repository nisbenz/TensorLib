#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tensor.h"

// Bumps a storage's ref count and attaches it to a (freshly created) view.
void add_ref_count(Storage* a, tensor* b) {
    if (a != NULL && b != NULL) {
        a->ref_count++;
        b->storage = a;
    }
}

Storage* s_alloc(int ndim, const int* dims) {
    if (ndim > 0 && dims == NULL) return NULL;
    Storage* s = (Storage*)malloc(sizeof(Storage));
    if (s == NULL) return NULL;
    s->ref_count = 1;
    s->data = NULL;

    int size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= dims[i];
    }
    s->size = size;

    s->data = (float*)malloc(size * sizeof(float));
    if (s->data == NULL) {
        free(s);
        return NULL;
    }

    return s;
}

tensor* t_alloc(int ndim, const int* dims) {
    if (ndim < 0 || (ndim > 0 && dims == NULL)) return NULL;

    tensor* a = (tensor*)malloc(sizeof(tensor));
    if (a == NULL) return NULL;

    a->storage = NULL;
    a->dims = NULL;
    a->strides = NULL;
    a->ndim = ndim;

    if (ndim > 0) {
        a->dims = (int*)malloc(ndim * sizeof(int));
        a->strides = (int*)malloc(ndim * sizeof(int));
        if (a->dims == NULL || a->strides == NULL) {
            t_free(a);
            return NULL;
        }

        for (int i = 0; i < ndim; ++i) {
            a->dims[i] = dims[i];
        }
    }

    a->storage = s_alloc(ndim, dims);
    if (a->storage == NULL) {
        t_free(a);
        return NULL;
    }

    if (ndim > 0) {
        calc_strides(ndim, a->dims, a->strides);
    }
    return a;
}

void t_free(tensor* t) {
    if (t == NULL) return;

    if (t->storage != NULL) {
        if (t->storage->ref_count > 1) {
            t->storage->ref_count--;
        } else {
            if (t->storage->data != NULL) {
                free(t->storage->data);
            }
            free(t->storage);
        }
    }
    if (t->dims != NULL) {
        free(t->dims);
    }
    if (t->strides != NULL) {
        free(t->strides);
    }
    free(t);
}

// Initializes tensor c to be a fresh, contiguous, zero-filled tensor
// with the same shape as ref. Used as the "allocate the output" step
// for elementwise ops like t_add.
int init_t(tensor* c, tensor* ref) {
    if (c == NULL || ref == NULL) return 1;

    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = ref->ndim;

    int total_elements = 1;
    for (int i = 0; i < ref->ndim; i++) {
        total_elements *= ref->dims[i];
    }

    c->storage = (Storage*)malloc(sizeof(Storage));
    if (c->storage == NULL) return 1;
    c->storage->ref_count = 1;
    c->storage->size = total_elements;
    c->storage->data = (float*)calloc(total_elements, sizeof(float));
    if (c->storage->data == NULL) {
        free(c->storage);
        c->storage = NULL;
        return 1;
    }

    if (ref->ndim > 0) {
        c->dims = (int*)malloc(ref->ndim * sizeof(int));
        c->strides = (int*)malloc(ref->ndim * sizeof(int));
        if (c->dims == NULL || c->strides == NULL) {
            return 1; // Caller is expected to call t_free(c) to clean up
        }
        for (int i = 0; i < ref->ndim; i++) {
            c->dims[i] = ref->dims[i];
            c->strides[i] = ref->strides[i];
        }
    }
    return 0;
}

tensor* t_clone(tensor* t) {
    if (t == NULL) return NULL;
    tensor* a = t_alloc(t->ndim, t->dims);
    if (a == NULL) return NULL;

    if (a->storage != NULL && a->storage->data != NULL && t->storage != NULL && t->storage->data != NULL) {
        memcpy(a->storage->data, t->storage->data, a->storage->size * sizeof(float));
    }

    for (int i = 0; i < a->ndim; ++i) {
        a->strides[i] = t->strides[i];
    }
    return a;
}