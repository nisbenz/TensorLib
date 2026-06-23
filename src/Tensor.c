#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
void add_ref_count(Storage* a , tensor* b){
 a->ref_count++;
 b->storage =a;
}
tensor* tensor_transpose(tensor* a, int dim0, int dim1) {
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

