#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include <string.h>

void add_ref_count(Storage* a, tensor* b) {
    if (a != NULL && b != NULL) {
        a->ref_count++;
        b->storage = a;
    }
}

tensor* t_transpose(tensor* a, int dim0, int dim1) {
    if (a == NULL) return NULL;
    if (dim0 < 0 || dim0 >= a->ndim || dim1 < 0 || dim1 >= a->ndim) {
        fprintf(stderr, "ERROR: Transpose dimensions out of bounds.\n");
        return NULL;
    }

    tensor* b = (tensor*)malloc(sizeof(tensor));
    if (b == NULL) return NULL;

    b->storage = NULL;
    b->dims = NULL;
    b->strides = NULL;
    b->ndim = a->ndim;

    if (a->ndim > 0) {
        b->dims = (int*)malloc(a->ndim * sizeof(int));
        b->strides = (int*)malloc(a->ndim * sizeof(int));
        if (b->dims == NULL || b->strides == NULL) {
            t_free(b);
            return NULL;
        }

        for (int i = 0; i < a->ndim; i++) {
            b->dims[i] = a->dims[i];
            b->strides[i] = a->strides[i];
        }

        b->dims[dim0] = a->dims[dim1];
        b->dims[dim1] = a->dims[dim0];

        b->strides[dim0] = a->strides[dim1];
        b->strides[dim1] = a->strides[dim0];
    }

    add_ref_count(a->storage, b);
    return b;
}

int get_flat_index_nd(tensor* t, int* coords) {
    if (t == NULL || coords == NULL) return 0;
    int flat_idx = 0;
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
        for (int i = 0; i < total_elements; i++) {
            c->storage->data[i] = a->storage->data[i] + b->storage->data[i];
        }
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
        a->strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            a->strides[i] = a->strides[i + 1] * a->dims[i + 1];
        }
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

tensor* t_contiguous(tensor* t) {
    if (t == NULL) return NULL;
    if (is_contiguous(t)) {
        tensor* view = (tensor*)malloc(sizeof(tensor));
        if (view == NULL) return NULL;

        view->storage = NULL;
        view->dims = NULL;
        view->strides = NULL;
        view->ndim = t->ndim;

        if (t->ndim > 0) {
            view->dims = (int*)malloc(t->ndim * sizeof(int));
            view->strides = (int*)malloc(t->ndim * sizeof(int));
            if (view->dims == NULL || view->strides == NULL) {
                t_free(view);
                return NULL;
            }
            for(int i = 0; i < t->ndim; i++) {
                view->dims[i] = t->dims[i];
                view->strides[i] = t->strides[i];
            }
        }

        add_ref_count(t->storage, view);
        return view;
    }

    tensor* flat_tensor = t_alloc(t->ndim, t->dims);
    if (flat_tensor == NULL) return NULL;

    int* coords = (int*)calloc(t->ndim, sizeof(int));
    if (coords == NULL) {
        t_free(flat_tensor);
        return NULL;
    }

    int total_elements = flat_tensor->storage->size;
    for (int i = 0; i < total_elements; i++) {
        int old_flat_idx = get_flat_index_nd(t, coords);
        flat_tensor->storage->data[i] = t->storage->data[old_flat_idx];
        advance_coords(coords, t->dims, t->ndim);
    }

    free(coords);
    return flat_tensor;
}