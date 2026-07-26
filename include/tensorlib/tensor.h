#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float* data;
    int ref_count;
    int size;
    uint64_t version;
} Storage;

typedef struct {
    Storage* storage;
    int ndim;
    int* dims;
    int* strides;
    int offset;
} tensor;

typedef struct tensor_matmul_packed_rhs tensor_matmul_packed_rhs;

void add_ref_count(Storage* a, tensor* b);
tensor* t_transpose(tensor* a, int dim0, int dim1);
int same_shape(tensor* a, tensor* b);
int same_stride(tensor* a, tensor* b);
int init_t(tensor* c, tensor* ref);
/* Elementwise binary operations use right-aligned NumPy-style broadcasting. */
tensor* t_add(tensor* a, tensor* b);
tensor* t_sub(tensor* a, tensor* b);
tensor* t_mul(tensor* a, tensor* b);
tensor* t_div(tensor* a, tensor* b);
/* Scalar helpers implement tensor op float without allocating a scalar tensor. */
tensor* t_add_scalar(tensor* a, float scalar);
tensor* t_sub_scalar(tensor* a, float scalar);
tensor* t_mul_scalar(tensor* a, float scalar);
tensor* t_div_scalar(tensor* a, float scalar);
void advance_coords(int* coords, const int* dims, int ndim);
int get_flat_index_nd(tensor* t, int* coords);
int tensor_numel(tensor* t);
int tensor_checked_numel(int ndim, const int* dims, size_t* result);
int tensor_has_valid_shape(const tensor* t);
int tensor_has_valid_layout(const tensor* t);
int tensor_has_valid_metadata(const tensor* t);
int tensor_copy_metadata(int ndim, const int* dims, const int* strides, int** out_dims, int** out_strides);
/* Scalars use ndim == 0. Dimensions must be positive; zero-sized tensors are unsupported. */
Storage* s_alloc(int ndim, const int* dims);
tensor* t_alloc(int ndim, const int* dims);
void t_free(tensor* t);
tensor* t_clone(tensor* t);
/*
 * Marks a completed in-place value update. Views share this counter. Direct
 * storage writes are supported for initialization before graph capture; later
 * writes must call this function for autograd mutation detection.
 */
void tensor_mark_modified(tensor* value);
int is_contiguous(tensor* t);
tensor* t_contiguous(tensor* t);
void calc_strides(int ndim, const int* dims, int* strides);
tensor* t_reshape(tensor* a, int new_ndim, int* new_dims);
tensor* t_squeeze(tensor* a, int dim);
tensor* t_unsqueeze(tensor* a, int dim);
/* Expand creates a zero-copy view; broadcast dimensions use zero strides. */
tensor* t_expand(tensor* a, int new_ndim, const int* new_dims);
tensor* t_slice(tensor* a, int dim, int start, int end);
/*
 * Select rows from a rank-2 table. Indices may have any rank and are stored as
 * finite, integral float values. The result shape is indices.shape followed by
 * the table's column count.
 */
tensor* t_gather_rows(tensor* table, tensor* indices);
tensor* t_exp(tensor* t);
tensor* t_log(tensor* t);
tensor* t_relu(tensor* t);
tensor* t_tanh(tensor* t);
tensor* t_sigmoid(tensor* t);
tensor* t_pow(tensor* t, float exponent);
tensor* t_neg(tensor* t);
tensor* t_sqrt(tensor* t);
tensor* t_gelu(tensor* t);
/* Legacy reductions remove the reduced dimension. */
tensor* t_sum(tensor* a, int dim);
tensor* t_mean(tensor* a, int dim);
tensor* t_max(tensor* a, int dim);
/* Keepdim reductions retain the reduced dimension with size one. */
tensor* t_sum_keepdim(tensor* a, int dim);
tensor* t_mean_keepdim(tensor* a, int dim);
tensor* t_max_keepdim(tensor* a, int dim);
tensor* t_matmul(tensor* a, tensor* b);
tensor_matmul_packed_rhs* t_pack_matmul_rhs(const tensor* rhs);
tensor* t_matmul_packed_rhs(const tensor* lhs,
                            const tensor_matmul_packed_rhs* rhs);
void t_free_matmul_packed_rhs(tensor_matmul_packed_rhs* rhs);
#endif //TENSORLIB_TENSOR_H
