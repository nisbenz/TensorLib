#ifndef TENSORLIB_TENSOR_MATMUL_H
#define TENSORLIB_TENSOR_MATMUL_H

#include "tensor.h"

typedef struct {
    int is_vector;
    int batch_rank;
    int rows;
    int inner;
    int columns;
} matmul_operand_info;

#define TENSORLIB_MATMUL_BLOCK_SIZE 32

int matmul_avx2_available(void);

void matmul_2d_blocked_contiguous(const float* a,
                                  const float* b,
                                  float* output,
                                  int rows,
                                  int inner,
                                  int columns);

void matmul_2d_avx2_contiguous(const float* a,
                               const float* b,
                               float* output,
                               int rows,
                               int inner,
                               int columns);

int operand_batch_dim(const tensor* operand,
                      const matmul_operand_info* info,
                      int output_axis,
                      int output_batch_ndim);

int operand_batch_offset(const tensor* operand,
                         const matmul_operand_info* info,
                         const int* batch_coords,
                         int batch_ndim);

int output_batch_offset(const tensor* output,
                        const int* batch_coords,
                        int batch_ndim);

int broadcast_batch_shape(const tensor* a,
                          const matmul_operand_info* a_info,
                          const tensor* b,
                          const matmul_operand_info* b_info,
                          int batch_ndim,
                          int* batch_dims);

void matmul_2d_strided(const tensor* a,
                       const matmul_operand_info* a_info,
                       int a_base,
                       const tensor* b,
                       const matmul_operand_info* b_info,
                       int b_base,
                       tensor* output,
                       int output_base,
                       int output_batch_ndim);

#endif /* TENSORLIB_TENSOR_MATMUL_H */
