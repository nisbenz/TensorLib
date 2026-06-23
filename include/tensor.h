//
// Created by profil on 6/23/2026.
//

#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H
typedef struct {
    float* data;
    int ref_count;
} Storage;

typedef struct {
    Storage* storage;
    int ndim;
    int dims[8];
    int strides[8];
} tensor;
void add_ref_count(Storage* a , tensor* b);
tensor* tensor_transpose(tensor* a, int dim0, int dim1);


#endif //TENSORLIB_TENSOR_H
