#ifndef TENSORLIB_AUTOGRAD_INTERNAL_H
#define TENSORLIB_AUTOGRAD_INTERNAL_H

#include "../../include/tensorlib/autograd.h"

#define AG_BACKWARD_OP_COUNT ((int)AG_OP_CROSS_ENTROPY + 1)

typedef struct {
    double operation_seconds[AG_BACKWARD_OP_COUNT];
    unsigned long operation_calls[AG_BACKWARD_OP_COUNT];
    double traversal_seconds;
    double reduction_seconds;
    double accumulation_seconds;
    double merge_seconds;
} ag_backward_stats;

void ag_backward_stats_enable(int enabled);
void ag_backward_stats_reset(void);
void ag_backward_stats_read(ag_backward_stats* output);

ag_tensor* ag_make_result(tensor* output,
                          ag_op operation,
                          int input_count,
                          ag_tensor* const* inputs,
                          ag_backward_fn backward,
                          void* context,
                          void (*free_context)(void*));

tensor* ag_full_like(const tensor* reference, float value);
tensor* ag_sum_to_shape(const tensor* source, const tensor* target, float scale);

ag_tensor* ag_matmul_packed_rhs(
    const ag_tensor* a,
    const ag_tensor* b,
    const tensor_matmul_packed_rhs* packed_rhs
);
ag_tensor* ag_matmul_packed_rhs_with_backward_pack(
    const ag_tensor* a,
    const ag_tensor* b,
    const tensor_matmul_packed_rhs* packed_rhs,
    const tensor_matmul_packed_rhs* backward_rhs
);

ag_tensor* ag_layer_norm(const ag_tensor* input,
                         const ag_tensor* weight,
                         const ag_tensor* bias,
                         float epsilon);

ag_tensor* ag_softmax_last_dim(const ag_tensor* input,
                               int log_softmax,
                               int causal);

ag_tensor* ag_cross_entropy(const ag_tensor* logits, const tensor* targets);

#endif
