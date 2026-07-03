#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"

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