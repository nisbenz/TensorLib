#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H
typedef struct {
    float* data;
    int ref_count;
    int size;
} Storage;

typedef struct {
    Storage* storage;
    int ndim;
    int dims[8];
    int strides[8];
} tensor;
void add_ref_count(Storage* a , tensor* b);
tensor* t_transpose(tensor* a, int dim0, int dim1);
int same_shape(tensor* a, tensor* b);
int same_stride(tensor* a, tensor* b);
void init_t(tensor* c, tensor* ref);
tensor* t_add(tensor* a, tensor* b);
void advance_coords(int* coords, int* dims, int ndim);
int get_flat_index_nd(tensor* t, int* coords);
#endif //TENSORLIB_TENSOR_H
