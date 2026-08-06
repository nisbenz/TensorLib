#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define TENSORLIB_HAS_AVX2_KERNEL 1
#define TENSORLIB_AVX2_TARGET __attribute__((target("avx2,fma")))
#elif defined(_M_AVX2)
#include <intrin.h>
#include <immintrin.h>
#define TENSORLIB_HAS_AVX2_KERNEL 1
#define TENSORLIB_AVX2_TARGET
#else
#define TENSORLIB_HAS_AVX2_KERNEL 0
#endif

#include "../../include/tensorlib/tensor_matmul.h"
#include "parallel.h"

enum {
    TENSORLIB_MATMUL_MC = 64,
    TENSORLIB_MATMUL_NC = 64,
    TENSORLIB_MATMUL_KC = 128,
    TENSORLIB_MATMUL_NR = 16,
    TENSORLIB_MATMUL_MR = 4,
    TENSORLIB_MATMUL_MAX_NDIM = 32
};

/* Below this many FLOPs, OpenMP fork/join overhead outweighs the speedup, so
 * the matmul stays single-threaded (small tensors are best on one core). */
#ifndef TENSORLIB_MATMUL_MIN_PARALLEL_FLOPS
#define TENSORLIB_MATMUL_MIN_PARALLEL_FLOPS 8388608LL
#endif

struct tensor_matmul_packed_rhs {
    int batch_rank;
    int* batch_dims;
    int inner;
    int columns;
    int panel_count;
    size_t batch_count;
    size_t values_per_batch;
    float* data;
};

static int checked_size_multiply(size_t left, size_t right, size_t* result) {
    if (result == NULL || (right != 0 && left > SIZE_MAX / right)) return 0;
    *result = left * right;
    return 1;
}

#if TENSORLIB_HAS_AVX2_KERNEL
static int matmul_2d_packed_rhs_avx2(const tensor* lhs, int lhs_base,
                                      const float* packed_rhs, int rows,
                                      int inner, int columns, tensor* output,
                                      int output_base, int output_batch_ndim);
#endif

static void* matmul_aligned_malloc(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 32);
#else
    return malloc(bytes);
#endif
}

static void matmul_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int packed_rhs_batch_offset(const tensor* rhs, size_t batch_index,
                                   int batch_rank, int* offset) {
    if (offset == NULL) return 0;

    int result = rhs->offset;
    for (int axis = batch_rank - 1; axis >= 0; --axis) {
        int coordinate = (int)(batch_index % (size_t)rhs->dims[axis]);
        batch_index /= (size_t)rhs->dims[axis];
        result += coordinate * rhs->strides[axis];
    }

    *offset = result;
    return 1;
}

static void pack_rhs_batch(const tensor* rhs, int rhs_base,
                           float* packed_data, int inner, int columns,
                           int panel_count) {
    int inner_stride = rhs->strides[rhs->ndim - 2];
    int column_stride = rhs->strides[rhs->ndim - 1];

    for (int panel = 0; panel < panel_count; ++panel) {
        int column_start = panel * TENSORLIB_MATMUL_NR;
        float* panel_data = packed_data +
            (size_t)panel * (size_t)inner * TENSORLIB_MATMUL_NR;

        for (int k = 0; k < inner; ++k) {
            for (int column = 0; column < TENSORLIB_MATMUL_NR; ++column) {
                int source_column = column_start + column;
                panel_data[k * TENSORLIB_MATMUL_NR + column] =
                    (source_column < columns)
                        ? rhs->storage->data[rhs_base + k * inner_stride +
                                             source_column * column_stride]
                        : 0.0f;
            }
        }
    }
}

tensor_matmul_packed_rhs* t_pack_matmul_rhs(const tensor* rhs) {
    if (!tensor_has_valid_metadata(rhs) || rhs->ndim < 2) return NULL;

    int batch_rank = rhs->ndim - 2;
    int inner_stride = rhs->strides[rhs->ndim - 2];
    int column_stride = rhs->strides[rhs->ndim - 1];
    if (inner_stride <= 0 || column_stride <= 0) return NULL;

    tensor_matmul_packed_rhs* packed =
        (tensor_matmul_packed_rhs*)calloc(1, sizeof(*packed));
    if (packed == NULL) return NULL;

    packed->batch_rank = batch_rank;
    packed->inner = rhs->dims[rhs->ndim - 2];
    packed->columns = rhs->dims[rhs->ndim - 1];
    packed->panel_count = (packed->columns + TENSORLIB_MATMUL_NR - 1) /
                          TENSORLIB_MATMUL_NR;

    if (!tensor_checked_numel(batch_rank, rhs->dims, &packed->batch_count)) {
        t_free_matmul_packed_rhs(packed);
        return NULL;
    }

    if (batch_rank > 0) {
        packed->batch_dims = (int*)malloc((size_t)batch_rank * sizeof(int));
        if (packed->batch_dims == NULL) {
            t_free_matmul_packed_rhs(packed);
            return NULL;
        }

        for (int axis = 0; axis < batch_rank; ++axis) {
            if (rhs->strides[axis] <= 0) {
                t_free_matmul_packed_rhs(packed);
                return NULL;
            }
            packed->batch_dims[axis] = rhs->dims[axis];
        }
    }

    size_t panel_values;
    if (!checked_size_multiply((size_t)packed->panel_count,
                               (size_t)packed->inner,
                               &panel_values) ||
        !checked_size_multiply(panel_values, TENSORLIB_MATMUL_NR,
                               &packed->values_per_batch)) {
        t_free_matmul_packed_rhs(packed);
        return NULL;
    }

    size_t total_values;
    if (!checked_size_multiply(packed->batch_count, packed->values_per_batch,
                               &total_values) ||
        !checked_size_multiply(total_values, sizeof(float), &total_values)) {
        t_free_matmul_packed_rhs(packed);
        return NULL;
    }

    packed->data = (float*)matmul_aligned_malloc(total_values);
    if (packed->data == NULL) {
        t_free_matmul_packed_rhs(packed);
        return NULL;
    }

    for (size_t batch = 0; batch < packed->batch_count; ++batch) {
        int rhs_base;
        packed_rhs_batch_offset(rhs, batch, batch_rank, &rhs_base);
        pack_rhs_batch(rhs, rhs_base,
                       packed->data + batch * packed->values_per_batch,
                       packed->inner, packed->columns, packed->panel_count);
    }

    return packed;
}

void t_free_matmul_packed_rhs(tensor_matmul_packed_rhs* rhs) {
    if (rhs == NULL) return;
    matmul_aligned_free(rhs->data);
    free(rhs->batch_dims);
    free(rhs);
}

static void batch_index_to_coords(size_t index, const int* dims, int ndim,
                                  int* coords) {
    for (int axis = ndim - 1; axis >= 0; --axis) {
        coords[axis] = (int)(index % (size_t)dims[axis]);
        index /= (size_t)dims[axis];
    }
}

static int packed_rhs_batch_dim(const tensor_matmul_packed_rhs* rhs,
                                int output_axis, int output_batch_ndim) {
    int rank_offset = output_batch_ndim - rhs->batch_rank;
    if (output_axis < rank_offset) return 1;
    return rhs->batch_dims[output_axis - rank_offset];
}

static size_t packed_rhs_batch_index(const tensor_matmul_packed_rhs* rhs,
                                     const int* batch_coords,
                                     int output_batch_ndim) {
    size_t result = 0;
    int rank_offset = output_batch_ndim - rhs->batch_rank;

    for (int axis = 0; axis < rhs->batch_rank; ++axis) {
        int coordinate = batch_coords[rank_offset + axis];
        if (rhs->batch_dims[axis] == 1) coordinate = 0;
        result = result * (size_t)rhs->batch_dims[axis] + (size_t)coordinate;
    }

    return result;
}

static void matmul_2d_packed_rhs_scalar(const tensor* lhs,
                                         const matmul_operand_info* lhs_info,
                                         int lhs_base,
                                         const float* packed_rhs,
                                         int columns,
                                         tensor* output,
                                         int output_base,
                                         int output_batch_ndim) {
    int lhs_row_stride = lhs->strides[lhs->ndim - 2];
    int lhs_inner_stride = lhs->strides[lhs->ndim - 1];
    int output_row_stride = output->strides[output_batch_ndim];
    int output_column_stride = output->strides[output_batch_ndim + 1];

    for (int row = 0; row < lhs_info->rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const float* rhs_panel = packed_rhs +
                (size_t)(column / TENSORLIB_MATMUL_NR) *
                (size_t)lhs_info->inner * TENSORLIB_MATMUL_NR;
            float sum = 0.0f;

            for (int inner = 0; inner < lhs_info->inner; ++inner) {
                sum += lhs->storage->data[lhs_base + row * lhs_row_stride +
                                          inner * lhs_inner_stride] *
                       rhs_panel[inner * TENSORLIB_MATMUL_NR +
                                 column % TENSORLIB_MATMUL_NR];
            }

            output->storage->data[output_base + row * output_row_stride +
                                  column * output_column_stride] = sum;
        }
    }
}

static void matmul_packed_rhs_batch(const tensor* lhs,
                                    const matmul_operand_info* lhs_info,
                                    const tensor_matmul_packed_rhs* rhs,
                                    tensor* output,
                                    const int* batch_dims,
                                    int batch_ndim,
                                    size_t batch) {
    int coords[TENSORLIB_MATMUL_MAX_NDIM];
    batch_index_to_coords(batch, batch_dims, batch_ndim, coords);

    int lhs_base = operand_batch_offset(lhs, lhs_info, coords, batch_ndim);
    int output_base = output_batch_offset(output, coords, batch_ndim);
    size_t rhs_batch = packed_rhs_batch_index(rhs, coords, batch_ndim);

    const float* packed_rhs = rhs->data + rhs_batch * rhs->values_per_batch;
#if TENSORLIB_HAS_AVX2_KERNEL
    int lhs_row_stride = lhs->strides[lhs->ndim - 2];
    int lhs_inner_stride = lhs->strides[lhs->ndim - 1];
    if (lhs_row_stride > 0 && lhs_inner_stride > 0 &&
        matmul_avx2_available() &&
        matmul_2d_packed_rhs_avx2(lhs, lhs_base, packed_rhs,
                                   lhs_info->rows, lhs_info->inner,
                                   rhs->columns, output, output_base,
                                   batch_ndim)) {
        /* The AVX2 kernel completed this batch. */
    } else {
        matmul_2d_packed_rhs_scalar(lhs, lhs_info, lhs_base,
                                    packed_rhs, rhs->columns,
                                    output, output_base, batch_ndim);
    }
#else
    matmul_2d_packed_rhs_scalar(lhs, lhs_info, lhs_base,
                                packed_rhs, rhs->columns,
                                output, output_base, batch_ndim);
#endif
}

tensor* t_matmul_packed_rhs(const tensor* lhs,
                            const tensor_matmul_packed_rhs* rhs) {
    if (!tensor_has_valid_metadata(lhs) || rhs == NULL || lhs->ndim < 2) {
        return NULL;
    }

    matmul_operand_info lhs_info = {
        .is_vector = 0,
        .batch_rank = lhs->ndim - 2,
        .rows = lhs->dims[lhs->ndim - 2],
        .inner = lhs->dims[lhs->ndim - 1],
        .columns = 0
    };
    if (lhs_info.inner != rhs->inner) return NULL;

    int batch_ndim = (lhs_info.batch_rank > rhs->batch_rank)
                   ? lhs_info.batch_rank : rhs->batch_rank;
    int* batch_dims = NULL;
    if (batch_ndim > 0) {
        batch_dims = (int*)malloc((size_t)batch_ndim * sizeof(int));
        if (batch_dims == NULL) return NULL;
    }

    for (int axis = 0; axis < batch_ndim; ++axis) {
        int lhs_dim = operand_batch_dim(lhs, &lhs_info, axis, batch_ndim);
        int rhs_dim = packed_rhs_batch_dim(rhs, axis, batch_ndim);
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            free(batch_dims);
            return NULL;
        }
        batch_dims[axis] = (lhs_dim > rhs_dim) ? lhs_dim : rhs_dim;
    }

    int output_dims_count = batch_ndim + 2;
    int* output_dims = (int*)malloc((size_t)output_dims_count * sizeof(int));
    if (output_dims == NULL) {
        free(batch_dims);
        return NULL;
    }
    for (int axis = 0; axis < batch_ndim; ++axis) output_dims[axis] = batch_dims[axis];
    output_dims[batch_ndim] = lhs_info.rows;
    output_dims[batch_ndim + 1] = rhs->columns;

    tensor* output = t_alloc(output_dims_count, output_dims);
    free(output_dims);
    if (output == NULL) {
        free(batch_dims);
        return NULL;
    }

    size_t batch_count;
    if (!tensor_checked_numel(batch_ndim, batch_dims, &batch_count)) {
        free(batch_dims);
        t_free(output);
        return NULL;
    }

    int batch_parallel = (batch_ndim <= TENSORLIB_MATMUL_MAX_NDIM);
    long long batch_flops = 2LL * lhs_info.rows * lhs_info.inner *
                            rhs->columns * (long long)batch_count;
    int threads = tensorlib_parallel_threads(
        batch_flops, TENSORLIB_MATMUL_MIN_PARALLEL_FLOPS,
        batch_parallel ? (int)batch_count : 1);

    if (threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (long long batch = 0; batch < (long long)batch_count; ++batch) {
            matmul_packed_rhs_batch(lhs, &lhs_info, rhs, output, batch_dims,
                                    batch_ndim, (size_t)batch);
        }
    } else {
        for (size_t batch = 0; batch < batch_count; ++batch) {
            matmul_packed_rhs_batch(lhs, &lhs_info, rhs, output, batch_dims,
                                    batch_ndim, batch);
        }
    }

    free(batch_dims);
    return output;
}

void matmul_2d_blocked_contiguous(const float* restrict a,
                                  const float* restrict b,
                                  float* restrict output,
                                  int rows,
                                  int inner,
                                  int columns) {
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            output[row * columns + column] = 0.0f;
        }
    }

    for (int row_block = 0; row_block < rows; row_block += TENSORLIB_MATMUL_BLOCK_SIZE) {
        int row_end = row_block + TENSORLIB_MATMUL_BLOCK_SIZE;
        if (row_end > rows) row_end = rows;

        for (int inner_block = 0; inner_block < inner; inner_block += TENSORLIB_MATMUL_BLOCK_SIZE) {
            int inner_end = inner_block + TENSORLIB_MATMUL_BLOCK_SIZE;
            if (inner_end > inner) inner_end = inner;

            for (int column_block = 0;
                 column_block < columns;
                 column_block += TENSORLIB_MATMUL_BLOCK_SIZE) {
                int column_end = column_block + TENSORLIB_MATMUL_BLOCK_SIZE;
                if (column_end > columns) column_end = columns;

                for (int row = row_block; row < row_end; ++row) {
                    for (int k = inner_block; k < inner_end; ++k) {
                        float a_value = a[row * inner + k];
                        for (int column = column_block; column < column_end; ++column) {
                            output[row * columns + column] +=
                                a_value * b[k * columns + column];
                        }
                    }
                }
            }
        }
    }
}

#if TENSORLIB_HAS_AVX2_KERNEL
int matmul_avx2_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached;

#if defined(__GNUC__)
    __builtin_cpu_init();
    cached = __builtin_cpu_supports("avx2") != 0 &&
             __builtin_cpu_supports("fma") != 0;
#else
    int registers[4];
    __cpuid(registers, 0);
    unsigned int highest_leaf = (unsigned int)registers[0];
    __cpuidex(registers, 1, 0);
    if ((registers[2] & (1 << 12)) == 0 || highest_leaf < 7) {
        cached = 0;
    } else {
        __cpuidex(registers, 7, 0);
        cached = (registers[1] & (1 << 5)) != 0;
    }
#endif
    return cached;
}

TENSORLIB_AVX2_TARGET
static void matmul_4x16_kernel(const float* restrict a,
                               const float* restrict packed_b,
                               float* restrict output,
                               int inner,
                               int output_stride,
                               int k_start,
                               int k_count,
                               int accumulate) {
    __m256 c00;
    __m256 c01;
    __m256 c10;
    __m256 c11;
    __m256 c20;
    __m256 c21;
    __m256 c30;
    __m256 c31;

    if (accumulate) {
        c00 = _mm256_loadu_ps(output + 0 * output_stride + 0);
        c01 = _mm256_loadu_ps(output + 0 * output_stride + 8);
        c10 = _mm256_loadu_ps(output + 1 * output_stride + 0);
        c11 = _mm256_loadu_ps(output + 1 * output_stride + 8);
        c20 = _mm256_loadu_ps(output + 2 * output_stride + 0);
        c21 = _mm256_loadu_ps(output + 2 * output_stride + 8);
        c30 = _mm256_loadu_ps(output + 3 * output_stride + 0);
        c31 = _mm256_loadu_ps(output + 3 * output_stride + 8);
    } else {
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
    }

    const float* a0 = a + 0 * inner + k_start;
    const float* a1 = a + 1 * inner + k_start;
    const float* a2 = a + 2 * inner + k_start;
    const float* a3 = a + 3 * inner + k_start;
    const float* b = packed_b + (size_t)k_start * TENSORLIB_MATMUL_NR;

    for (int k = 0; k < k_count; ++k) {
        __m256 b0 = _mm256_loadu_ps(b + 0);
        __m256 b1 = _mm256_loadu_ps(b + 8);

        __m256 a0_value = _mm256_broadcast_ss(a0 + k);
        __m256 a1_value = _mm256_broadcast_ss(a1 + k);
        __m256 a2_value = _mm256_broadcast_ss(a2 + k);
        __m256 a3_value = _mm256_broadcast_ss(a3 + k);

        c00 = _mm256_fmadd_ps(a0_value, b0, c00);
        c01 = _mm256_fmadd_ps(a0_value, b1, c01);
        c10 = _mm256_fmadd_ps(a1_value, b0, c10);
        c11 = _mm256_fmadd_ps(a1_value, b1, c11);
        c20 = _mm256_fmadd_ps(a2_value, b0, c20);
        c21 = _mm256_fmadd_ps(a2_value, b1, c21);
        c30 = _mm256_fmadd_ps(a3_value, b0, c30);
        c31 = _mm256_fmadd_ps(a3_value, b1, c31);
        b += TENSORLIB_MATMUL_NR;
    }

    _mm256_storeu_ps(output + 0 * output_stride + 0, c00);
    _mm256_storeu_ps(output + 0 * output_stride + 8, c01);
    _mm256_storeu_ps(output + 1 * output_stride + 0, c10);
    _mm256_storeu_ps(output + 1 * output_stride + 8, c11);
    _mm256_storeu_ps(output + 2 * output_stride + 0, c20);
    _mm256_storeu_ps(output + 2 * output_stride + 8, c21);
    _mm256_storeu_ps(output + 3 * output_stride + 0, c30);
    _mm256_storeu_ps(output + 3 * output_stride + 8, c31);
}

static void pack_b_panels(const float* restrict b,
                          float* restrict packed_b,
                          int inner,
                          int columns,
                          int panel_count) {
    for (int panel = 0; panel < panel_count; ++panel) {
        int column_start = panel * TENSORLIB_MATMUL_NR;
        float* panel_data = packed_b +
            (size_t)panel * (size_t)inner * TENSORLIB_MATMUL_NR;

        for (int k = 0; k < inner; ++k) {
            for (int column = 0; column < TENSORLIB_MATMUL_NR; ++column) {
                int source_column = column_start + column;
                panel_data[k * TENSORLIB_MATMUL_NR + column] =
                    (source_column < columns)
                        ? b[k * columns + source_column]
                        : 0.0f;
            }
        }
    }
}

TENSORLIB_AVX2_TARGET
static void matmul_2d_contiguous_row_block(const float* restrict a,
                                           const float* restrict packed_b,
                                           float* restrict output,
                                           int row_block, int row_end,
                                           int full_columns,
                                           int inner, int columns) {
    for (int column_block = 0; column_block < full_columns;
         column_block += TENSORLIB_MATMUL_NC) {
        int column_end = column_block + TENSORLIB_MATMUL_NC;
        if (column_end > full_columns) column_end = full_columns;

        for (int inner_block = 0; inner_block < inner;
             inner_block += TENSORLIB_MATMUL_KC) {
            int inner_end = inner_block + TENSORLIB_MATMUL_KC;
            if (inner_end > inner) inner_end = inner;
            int accumulate = (inner_block != 0);

            for (int column_panel = column_block;
                 column_panel < column_end;
                 column_panel += TENSORLIB_MATMUL_NR) {
                const float* packed_panel = packed_b +
                    (size_t)(column_panel / TENSORLIB_MATMUL_NR) *
                    (size_t)inner * TENSORLIB_MATMUL_NR;

                for (int row = row_block; row < row_end;
                     row += TENSORLIB_MATMUL_MR) {
                    matmul_4x16_kernel(
                        a + row * inner,
                        packed_panel,
                        output + row * columns + column_panel,
                        inner,
                        columns,
                        inner_block,
                        inner_end - inner_block,
                        accumulate);
                }
            }
        }
    }
}

TENSORLIB_AVX2_TARGET
void matmul_2d_avx2_contiguous(const float* restrict a,
                               const float* restrict b,
                               float* restrict output,
                               int rows,
                               int inner,
                               int columns) {
    int full_rows = rows - (rows % TENSORLIB_MATMUL_MR);
    int full_columns = columns - (columns % TENSORLIB_MATMUL_NR);
    int panel_count = (columns + TENSORLIB_MATMUL_NR - 1) /
                      TENSORLIB_MATMUL_NR;

    if (full_rows > 0 && full_columns > 0) {
        size_t packed_count = (size_t)panel_count * (size_t)inner *
                              TENSORLIB_MATMUL_NR;
        float* packed_b = (float*)matmul_aligned_malloc(
            packed_count * sizeof(float));

        if (packed_b == NULL) {
            matmul_2d_blocked_contiguous(a, b, output, rows, inner, columns);
            return;
        }

        pack_b_panels(b, packed_b, inner, columns, panel_count);

        int row_block_count = (full_rows + TENSORLIB_MATMUL_MC - 1) /
                              TENSORLIB_MATMUL_MC;
        long long flops = 2LL * rows * inner * columns;
        int threads = tensorlib_parallel_threads(
            flops, TENSORLIB_MATMUL_MIN_PARALLEL_FLOPS, row_block_count);

        if (threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int rb = 0; rb < row_block_count; ++rb) {
                int row_block = rb * TENSORLIB_MATMUL_MC;
                int row_end = row_block + TENSORLIB_MATMUL_MC;
                if (row_end > full_rows) row_end = full_rows;
                matmul_2d_contiguous_row_block(
                    a, packed_b, output, row_block, row_end,
                    full_columns, inner, columns);
            }
        } else {
            for (int rb = 0; rb < row_block_count; ++rb) {
                int row_block = rb * TENSORLIB_MATMUL_MC;
                int row_end = row_block + TENSORLIB_MATMUL_MC;
                if (row_end > full_rows) row_end = full_rows;
                matmul_2d_contiguous_row_block(
                    a, packed_b, output, row_block, row_end,
                    full_columns, inner, columns);
            }
        }

        matmul_aligned_free(packed_b);
    }

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            if (row < full_rows && column < full_columns) continue;

            float sum = 0.0f;
            for (int k = 0; k < inner; ++k) {
                sum += a[row * inner + k] * b[k * columns + column];
            }
            output[row * columns + column] = sum;
        }
    }
}

TENSORLIB_AVX2_TARGET
static void matmul_4x16_packed_a_kernel(const float* restrict packed_a,
                                         const float* restrict packed_b,
                                         float* restrict output,
                                         int output_stride,
                                         int k_count,
                                         int accumulate) {
    __m256 c00;
    __m256 c01;
    __m256 c10;
    __m256 c11;
    __m256 c20;
    __m256 c21;
    __m256 c30;
    __m256 c31;

    if (accumulate) {
        c00 = _mm256_loadu_ps(output + 0 * output_stride + 0);
        c01 = _mm256_loadu_ps(output + 0 * output_stride + 8);
        c10 = _mm256_loadu_ps(output + 1 * output_stride + 0);
        c11 = _mm256_loadu_ps(output + 1 * output_stride + 8);
        c20 = _mm256_loadu_ps(output + 2 * output_stride + 0);
        c21 = _mm256_loadu_ps(output + 2 * output_stride + 8);
        c30 = _mm256_loadu_ps(output + 3 * output_stride + 0);
        c31 = _mm256_loadu_ps(output + 3 * output_stride + 8);
    } else {
        c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
    }

    const float* a0 = packed_a;
    const float* a1 = packed_a + k_count;
    const float* a2 = packed_a + 2 * k_count;
    const float* a3 = packed_a + 3 * k_count;

    for (int k = 0; k < k_count; ++k) {
        __m256 b0 = _mm256_loadu_ps(packed_b + k * TENSORLIB_MATMUL_NR);
        __m256 b1 = _mm256_loadu_ps(packed_b + k * TENSORLIB_MATMUL_NR + 8);
        __m256 a0_value = _mm256_broadcast_ss(a0 + k);
        __m256 a1_value = _mm256_broadcast_ss(a1 + k);
        __m256 a2_value = _mm256_broadcast_ss(a2 + k);
        __m256 a3_value = _mm256_broadcast_ss(a3 + k);

        c00 = _mm256_fmadd_ps(a0_value, b0, c00);
        c01 = _mm256_fmadd_ps(a0_value, b1, c01);
        c10 = _mm256_fmadd_ps(a1_value, b0, c10);
        c11 = _mm256_fmadd_ps(a1_value, b1, c11);
        c20 = _mm256_fmadd_ps(a2_value, b0, c20);
        c21 = _mm256_fmadd_ps(a2_value, b1, c21);
        c30 = _mm256_fmadd_ps(a3_value, b0, c30);
        c31 = _mm256_fmadd_ps(a3_value, b1, c31);
    }

    _mm256_storeu_ps(output + 0 * output_stride + 0, c00);
    _mm256_storeu_ps(output + 0 * output_stride + 8, c01);
    _mm256_storeu_ps(output + 1 * output_stride + 0, c10);
    _mm256_storeu_ps(output + 1 * output_stride + 8, c11);
    _mm256_storeu_ps(output + 2 * output_stride + 0, c20);
    _mm256_storeu_ps(output + 2 * output_stride + 8, c21);
    _mm256_storeu_ps(output + 3 * output_stride + 0, c30);
    _mm256_storeu_ps(output + 3 * output_stride + 8, c31);
}

static void pack_lhs_block(const tensor* lhs, int lhs_base, int row_start,
                           int row_count, int k_start, int k_count,
                           float* packed_lhs) {
    int row_stride = lhs->strides[lhs->ndim - 2];
    int inner_stride = lhs->strides[lhs->ndim - 1];

    for (int row = 0; row < row_count; ++row) {
        float* packed_row = packed_lhs + (size_t)row * k_count;
        int source_base = lhs_base + (row_start + row) * row_stride +
                          k_start * inner_stride;
        for (int k = 0; k < k_count; ++k) {
            packed_row[k] = lhs->storage->data[source_base + k * inner_stride];
        }
    }
}

static void matmul_2d_packed_rhs_scalar_tails(const tensor* lhs, int lhs_base,
                                               const float* packed_rhs,
                                               int rows, int inner, int columns,
                                               int full_rows, int full_columns,
                                               tensor* output, int output_base,
                                               int output_batch_ndim) {
    int lhs_row_stride = lhs->strides[lhs->ndim - 2];
    int lhs_inner_stride = lhs->strides[lhs->ndim - 1];
    int output_row_stride = output->strides[output_batch_ndim];
    int output_column_stride = output->strides[output_batch_ndim + 1];

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            if (row < full_rows && column < full_columns) continue;
            const float* rhs_panel = packed_rhs +
                (size_t)(column / TENSORLIB_MATMUL_NR) * inner *
                TENSORLIB_MATMUL_NR;
            float sum = 0.0f;
            for (int k = 0; k < inner; ++k) {
                sum += lhs->storage->data[lhs_base + row * lhs_row_stride +
                                          k * lhs_inner_stride] *
                       rhs_panel[k * TENSORLIB_MATMUL_NR +
                                 column % TENSORLIB_MATMUL_NR];
            }
            output->storage->data[output_base + row * output_row_stride +
                                  column * output_column_stride] = sum;
        }
    }
}

TENSORLIB_AVX2_TARGET
static int matmul_2d_packed_rhs_avx2(const tensor* lhs, int lhs_base,
                                      const float* packed_rhs, int rows,
                                      int inner, int columns, tensor* output,
                                      int output_base, int output_batch_ndim) {
    size_t workspace_values;
    if (!checked_size_multiply(TENSORLIB_MATMUL_MC, TENSORLIB_MATMUL_KC,
                               &workspace_values)) {
        return 0;
    }
    float* packed_lhs = (float*)matmul_aligned_malloc(workspace_values * sizeof(float));
    if (packed_lhs == NULL) return 0;

    int full_rows = rows - rows % TENSORLIB_MATMUL_MR;
    int full_columns = columns - columns % TENSORLIB_MATMUL_NR;
    int output_stride = output->strides[output_batch_ndim];

    for (int row_block = 0; row_block < full_rows;
         row_block += TENSORLIB_MATMUL_MC) {
        int row_end = row_block + TENSORLIB_MATMUL_MC;
        if (row_end > full_rows) row_end = full_rows;
        int row_count = row_end - row_block;

        for (int k_start = 0; k_start < inner; k_start += TENSORLIB_MATMUL_KC) {
            int k_end = k_start + TENSORLIB_MATMUL_KC;
            if (k_end > inner) k_end = inner;
            int k_count = k_end - k_start;
            int accumulate = (k_start != 0);

            pack_lhs_block(lhs, lhs_base, row_block, row_count,
                           k_start, k_count, packed_lhs);

            for (int column = 0; column < full_columns;
                 column += TENSORLIB_MATMUL_NR) {
                const float* rhs_panel = packed_rhs +
                    (size_t)(column / TENSORLIB_MATMUL_NR) * inner *
                    TENSORLIB_MATMUL_NR +
                    (size_t)k_start * TENSORLIB_MATMUL_NR;

                for (int row = row_block; row < row_end;
                     row += TENSORLIB_MATMUL_MR) {
                    matmul_4x16_packed_a_kernel(
                        packed_lhs + (size_t)(row - row_block) * k_count,
                        rhs_panel,
                        output->storage->data + output_base +
                            row * output_stride + column,
                        output_stride, k_count, accumulate);
                }
            }
        }
    }

    matmul_2d_packed_rhs_scalar_tails(lhs, lhs_base, packed_rhs,
                                      rows, inner, columns,
                                      full_rows, full_columns,
                                      output, output_base, output_batch_ndim);
    matmul_aligned_free(packed_lhs);
    return 1;
}
#else
int matmul_avx2_available(void) {
    return 0;
}
#endif

int operand_batch_dim(const tensor* operand,
                      const matmul_operand_info* info,
                      int output_axis,
                      int output_batch_ndim) {
    int rank_offset = output_batch_ndim - info->batch_rank;
    if (output_axis < rank_offset) return 1;
    return operand->dims[output_axis - rank_offset];
}

int operand_batch_offset(const tensor* operand,
                         const matmul_operand_info* info,
                         const int* batch_coords,
                         int batch_ndim) {
    int offset = operand->offset;

    for (int operand_axis = 0; operand_axis < info->batch_rank; ++operand_axis) {
        int output_axis = batch_ndim - info->batch_rank + operand_axis;
        int coordinate = batch_coords[output_axis];

        /* A singleton batch axis is broadcast by reusing coordinate zero. */
        if (operand->dims[operand_axis] == 1) coordinate = 0;
        offset += coordinate * operand->strides[operand_axis];
    }

    return offset;
}

int output_batch_offset(const tensor* output,
                        const int* batch_coords,
                        int batch_ndim) {
    int offset = output->offset;
    for (int axis = 0; axis < batch_ndim; ++axis) {
        offset += batch_coords[axis] * output->strides[axis];
    }
    return offset;
}

int broadcast_batch_shape(const tensor* a,
                          const matmul_operand_info* a_info,
                          const tensor* b,
                          const matmul_operand_info* b_info,
                          int batch_ndim,
                          int* batch_dims) {
    for (int axis = 0; axis < batch_ndim; ++axis) {
        int a_dim = operand_batch_dim(a, a_info, axis, batch_ndim);
        int b_dim = operand_batch_dim(b, b_info, axis, batch_ndim);

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) return 0;
        batch_dims[axis] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    return 1;
}

void matmul_2d_strided(const tensor* a,
                       const matmul_operand_info* a_info,
                       int a_base,
                       const tensor* b,
                       const matmul_operand_info* b_info,
                       int b_base,
                       tensor* output,
                       int output_base,
                       int output_batch_ndim) {
    const int a_row_stride = a_info->is_vector ? 0 : a->strides[a->ndim - 2];
    const int a_inner_stride = a->strides[a->ndim - 1];
    const int b_inner_stride = b->strides[b_info->is_vector ? 0 : b->ndim - 2];
    const int b_column_stride = b_info->is_vector ? 0 : b->strides[b->ndim - 1];

    int output_row_stride = 0;
    int output_column_stride = 0;
    if (!a_info->is_vector && !b_info->is_vector) {
        output_row_stride = output->strides[output_batch_ndim];
        output_column_stride = output->strides[output_batch_ndim + 1];
    } else if (a_info->is_vector) {
        /* The promoted leading row dimension is removed from the result. */
        if (!b_info->is_vector) {
            output_column_stride = output->strides[output_batch_ndim];
        }
    } else {
        /* The promoted trailing column dimension is removed from the result. */
        output_row_stride = output->strides[output_batch_ndim];
    }

    if (!a_info->is_vector && !b_info->is_vector &&
        a_row_stride == a_info->inner && a_inner_stride == 1 &&
        b_inner_stride == b_info->columns && b_column_stride == 1 &&
        output_row_stride == b_info->columns && output_column_stride == 1) {
        const float* a_data = a->storage->data + a_base;
        const float* b_data = b->storage->data + b_base;
        float* output_data = output->storage->data + output_base;

#if TENSORLIB_HAS_AVX2_KERNEL
        if (matmul_avx2_available()) {
            matmul_2d_avx2_contiguous(a_data, b_data, output_data,
                                      a_info->rows, a_info->inner, b_info->columns);
        } else {
            matmul_2d_blocked_contiguous(a_data, b_data, output_data,
                                         a_info->rows, a_info->inner, b_info->columns);
        }
#else
        matmul_2d_blocked_contiguous(a_data, b_data, output_data,
                                     a_info->rows, a_info->inner, b_info->columns);
#endif
        return;
    }

    for (int row = 0; row < a_info->rows; ++row) {
        for (int column = 0; column < b_info->columns; ++column) {
            float sum = 0.0f;

            for (int inner = 0; inner < a_info->inner; ++inner) {
                int a_index = a_base + row * a_row_stride + inner * a_inner_stride;
                int b_index = b_base + inner * b_inner_stride + column * b_column_stride;
                sum += a->storage->data[a_index] * b->storage->data[b_index];
            }

            int output_index = output_base + row * output_row_stride + column * output_column_stride;
            output->storage->data[output_index] = sum;
        }
    }
}

static int matmul_matrix_operands_are_contiguous(const tensor* a,
                                                 const matmul_operand_info* a_info,
                                                 const tensor* b,
                                                 const matmul_operand_info* b_info) {
    if (a == NULL || a_info == NULL || b == NULL || b_info == NULL ||
        a_info->is_vector || b_info->is_vector) {
        return 0;
    }

    return a->strides[a->ndim - 2] == a_info->inner &&
           a->strides[a->ndim - 1] == 1 &&
           b->strides[b->ndim - 2] == b_info->columns &&
           b->strides[b->ndim - 1] == 1;
}

static tensor* try_packed_matrix_matmul(const tensor* a,
                                        const matmul_operand_info* a_info,
                                        const tensor* b,
                                        const matmul_operand_info* b_info) {
    if (a == NULL || a_info == NULL || b == NULL || b_info == NULL ||
        a_info->is_vector || b_info->is_vector ||
        matmul_matrix_operands_are_contiguous(a, a_info, b, b_info) ||
        !matmul_avx2_available()) {
        return NULL;
    }
    /*
     * Keep packing below t_matmul's public dispatch boundary. This lets
     * autograd and ordinary callers get the same view-aware kernel choice.
     * The reusable packed-RHS API rejects unsupported zero-stride RHS views;
     * the caller falls back to the general strided implementation below.
     */
    tensor_matmul_packed_rhs* packed_rhs = t_pack_matmul_rhs(b);
    if (packed_rhs == NULL) return NULL;

    tensor* output = t_matmul_packed_rhs(a, packed_rhs);
    t_free_matmul_packed_rhs(packed_rhs);
    return output;
}

static void matmul_batch(const tensor* a,
                         const matmul_operand_info* a_info,
                         const tensor* b,
                         const matmul_operand_info* b_info,
                         tensor* output,
                         const int* batch_dims,
                         int batch_ndim,
                         size_t batch) {
    int coords[TENSORLIB_MATMUL_MAX_NDIM];
    batch_index_to_coords(batch, batch_dims, batch_ndim, coords);

    int a_base = operand_batch_offset(a, a_info, coords, batch_ndim);
    int b_base = operand_batch_offset(b, b_info, coords, batch_ndim);
    int output_base = output_batch_offset(output, coords, batch_ndim);

    matmul_2d_strided(a, a_info, a_base,
                      b, b_info, b_base,
                      output, output_base, batch_ndim);
}

tensor* t_matmul(tensor* a, tensor* b) {
    if (!tensor_has_valid_metadata(a) || !tensor_has_valid_metadata(b)) {
        return NULL;
    }

    /* Scalars are not matrix operands. Vectors are supported through promotion. */
    if (a->ndim == 0 || b->ndim == 0) {
        fprintf(stderr, "ERROR: Matmul operands must have at least one dimension.\n");
        return NULL;
    }

    matmul_operand_info a_info = {
        .is_vector = (a->ndim == 1),
        .batch_rank = (a->ndim > 1) ? a->ndim - 2 : 0,
        .rows = (a->ndim == 1) ? 1 : a->dims[a->ndim - 2],
        .inner = a->dims[a->ndim - 1],
        .columns = 0
    };
    matmul_operand_info b_info = {
        .is_vector = (b->ndim == 1),
        .batch_rank = (b->ndim > 1) ? b->ndim - 2 : 0,
        .rows = 0,
        .inner = (b->ndim == 1) ? b->dims[0] : b->dims[b->ndim - 2],
        .columns = (b->ndim == 1) ? 1 : b->dims[b->ndim - 1]
    };

    if (a_info.inner != b_info.inner) {
        fprintf(stderr, "ERROR: Matmul inner dimensions do not match.\n");
        return NULL;
    }

    if (a_info.batch_rank > INT_MAX - 2 || b_info.batch_rank > INT_MAX - 2) {
        return NULL;
    }

    int batch_ndim = (a_info.batch_rank > b_info.batch_rank)
                   ? a_info.batch_rank
                   : b_info.batch_rank;
    int* batch_dims = NULL;
    if (batch_ndim > 0) {
        batch_dims = (int*)malloc((size_t)batch_ndim * sizeof(int));
        if (batch_dims == NULL) return NULL;
    }

    if (!broadcast_batch_shape(a, &a_info, b, &b_info, batch_ndim, batch_dims)) {
        fprintf(stderr, "ERROR: Matmul batch dimensions are not broadcast-compatible.\n");
        free(batch_dims);
        return NULL;
    }

    tensor* packed_output = try_packed_matrix_matmul(a, &a_info, b, &b_info);
    if (packed_output != NULL) {
        free(batch_dims);
        return packed_output;
    }

    int output_ndim = batch_ndim + 2 - a_info.is_vector - b_info.is_vector;
    int* output_dims = NULL;
    if (output_ndim > 0) {
        output_dims = (int*)malloc((size_t)output_ndim * sizeof(int));
        if (output_dims == NULL) {
            free(batch_dims);
            return NULL;
        }

        for (int axis = 0; axis < batch_ndim; ++axis) {
            output_dims[axis] = batch_dims[axis];
        }

        int output_axis = batch_ndim;
        if (!a_info.is_vector) output_dims[output_axis++] = a_info.rows;
        if (!b_info.is_vector) output_dims[output_axis] = b_info.columns;
    }

    tensor* output = t_alloc(output_ndim, output_dims);
    free(output_dims);
    if (output == NULL) {
        free(batch_dims);
        return NULL;
    }

    size_t batch_count;
    if (!tensor_checked_numel(batch_ndim, batch_dims, &batch_count)) {
        free(batch_dims);
        t_free(output);
        return NULL;
    }

    int batch_parallel = (batch_ndim <= TENSORLIB_MATMUL_MAX_NDIM);
    long long batch_flops = 2LL * a_info.rows * a_info.inner *
                            b_info.columns * (long long)batch_count;
    int threads = tensorlib_parallel_threads(
        batch_flops, TENSORLIB_MATMUL_MIN_PARALLEL_FLOPS,
        batch_parallel ? (int)batch_count : 1);

    if (threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (long long batch_index = 0;
             batch_index < (long long)batch_count; ++batch_index) {
            matmul_batch(a, &a_info, b, &b_info, output, batch_dims,
                         batch_ndim, (size_t)batch_index);
        }
    } else {
        for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            matmul_batch(a, &a_info, b, &b_info, output, batch_dims,
                         batch_ndim, batch_index);
        }
    }

    free(batch_dims);
    return output;
}
