#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
void add_ref_count(Storage* a , tensor* b){
 a->ref_count++;
 b->storage =a;
}
tensor* t_transpose(tensor* a, int dim0, int dim1) {
    tensor* b = (tensor*)malloc(sizeof(tensor));

    add_ref_count(a->storage, b);
    b->ndim = a->ndim;

    for(int i = 0; i < a->ndim; i++) {
        b->dims[i] = a->dims[i];
        b->strides[i] = a->strides[i];
    }

    b->dims[dim0] = a->dims[dim1];
    b->dims[dim1] = a->dims[dim0];

    b->strides[dim0] = a->strides[dim1];
    b->strides[dim1] = a->strides[dim0];
    return b;
}
int get_flat_index_nd(tensor* t, int* coords) {
    int flat_idx = 0;

    for (int i = 0; i < t->ndim; i++) {
        flat_idx += coords[i] * t->strides[i];
    }

    return flat_idx;
}
int same_shape(tensor* a, tensor* b) {
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
void init_t(tensor* c, tensor* ref) {
    int total_elements = 1;
    for (int i = 0; i < ref->ndim; i++) {
        total_elements *= ref->dims[i];
    }

    c->storage = (Storage*)malloc(sizeof(Storage));
    c->storage->data = (float*)calloc(total_elements, sizeof(float)); // calloc initializes to 0.0
    c->storage->ref_count = 1;

    c->ndim = ref->ndim;
    for (int i = 0; i < ref->ndim; i++) {
        c->dims[i] = ref->dims[i];
        c->strides[i] = ref->strides[i];
    }
}
void advance_coords(int* coords, int* dims, int ndim) {
    for (int i = ndim - 1; i >= 0; i--) {

        if (coords[i] < dims[i]) {
            break;
        } else {
            coords[i] = 0;
        }
    }
}

tensor* t_add(tensor* a, tensor* b) {
    if (same_shape(a, b) == 0) {
        fprintf(stderr, "ERROR: Tensors must have identical shapes to add.\n");
        return NULL;
    }

    tensor* c = (tensor*)malloc(sizeof(tensor));
    init_t(c, a);

    int total_elements = c->storage->size;

    if (same_stride(a, b) == 1 && same_stride(a, c) == 1) {
        for (int i = 0; i < total_elements; i++) {
            c->storage->data[i] = a->storage->data[i] + b->storage->data[i];
        }
    } else {

        int coords[8] = {0};

        for (int i = 0; i < total_elements; i++) {
            int idx_a = get_flat_index_nd(a, coords);
            int idx_b = get_flat_index_nd(b, coords);

            c->storage->data[i] = a->storage->data[idx_a] + b->storage->data[idx_b];

            advance_coords(coords, a->dims, a->ndim);
        }
    }

    return c;
}
