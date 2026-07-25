#ifndef TENSORLIB_AUTOGRAD_H
#define TENSORLIB_AUTOGRAD_H

#include <stddef.h>

#include "tensor.h"

/*
 * TensorLib records a dynamic reverse-mode graph during eager forward
 * execution. Graph nodes retain their inputs; outputs own their creator nodes.
 * Backward retains the graph and repeated calls accumulate gradients.
 *
 * Backward is transactional: validation or allocation failure leaves all
 * previously accumulated gradients unchanged. Broadcast contributions are
 * reduced back to each input's original shape before accumulation.
 *
 * Values participating in a graph must not be changed before backward.
 * Library-controlled in-place writes must use tensor_mark_modified(); direct
 * storage writes are supported only for initialization before graph capture.
 */

/* Forward declarations allow tensors and graph nodes to refer to each other. */
typedef struct ag_tensor ag_tensor;
typedef struct ag_node ag_node;

/* Operations that can create autograd graph nodes. */
typedef enum {
    AG_OP_ADD,
    AG_OP_SUB,
    AG_OP_MUL,
    AG_OP_DIV,
    AG_OP_NEG,
    AG_OP_EXP,
    AG_OP_LOG,
    AG_OP_POW,
    AG_OP_SQRT,
    AG_OP_RELU,
    AG_OP_SIGMOID,
    AG_OP_TANH,
    AG_OP_GELU,
    AG_OP_MATMUL,
    AG_OP_SUM,
    AG_OP_MEAN,
    AG_OP_MAX,
    AG_OP_RESHAPE,
    AG_OP_TRANSPOSE,
    AG_OP_SLICE,
    AG_OP_EXPAND
} ag_op;


/*
 * Local backward rules allocate one owned contribution per differentiable
 * input. The caller supplies an input_count-sized, NULL-initialized array and
 * takes ownership of every tensor written to it.
 */
typedef int (*ag_backward_fn)(
    const ag_node* node,
    const tensor* output_gradient,
    tensor** input_gradients
);


struct ag_tensor {
    tensor* value;
    tensor* grad;

    int requires_grad;

    ag_node* creator;

    int ref_count;
};


struct ag_node {
    ag_op operation;

    int input_count;
    ag_tensor** inputs;

    /* Non-owning; the output owns this node through ag_tensor.creator. */
    ag_tensor* output;

    ag_backward_fn backward;

    /* Operation-specific saved values or metadata. */
    void* context;
    void (*free_context)(void* context);

    /* Storage versions captured after forward for stale-graph detection. */
    uint64_t* input_versions;
    uint64_t output_version;

    int ref_count;
};

/* Takes ownership of value, including when construction fails. */
ag_tensor* ag_from_owned_tensor(tensor* value, int requires_grad);
/*
 * Returns a zero-copy leaf alias with requires_grad == 0, grad == NULL, and
 * creator == NULL. Shape, strides, offset, storage, and storage version are
 * shared with value.
 */
ag_tensor* ag_detach(const ag_tensor* value);
void ag_tensor_retain(ag_tensor* value);
void ag_tensor_release(ag_tensor* value);
void ag_node_retain(ag_node* node);
void ag_node_release(ag_node* node);

/*
 * Differentiable forward operations mirror the corresponding raw t_*
 * operation's shape, broadcasting, view, and numerical behavior. Invalid
 * arguments return NULL. A result records a creator only when at least one
 * input requires gradients; otherwise it is an untracked leaf.
 */
ag_tensor* ag_add(const ag_tensor* a, const ag_tensor* b);
ag_tensor* ag_sub(const ag_tensor* a, const ag_tensor* b);
ag_tensor* ag_mul(const ag_tensor* a, const ag_tensor* b);
ag_tensor* ag_div(const ag_tensor* a, const ag_tensor* b);
ag_tensor* ag_neg(const ag_tensor* value);
ag_tensor* ag_exp(const ag_tensor* value);
ag_tensor* ag_log(const ag_tensor* value);
/*
 * Raises each element to a scalar exponent. Backward uses
 * exponent * x^(exponent - 1), with exact zero/one exponent special cases.
 */
ag_tensor* ag_pow(const ag_tensor* value, float exponent);
/* Backward computes upstream / (2 * sqrt(value)); IEEE domains are preserved. */
ag_tensor* ag_sqrt(const ag_tensor* value);
/* ReLU uses derivative zero at zero and a NaN gradient for NaN input. */
ag_tensor* ag_relu(const ag_tensor* value);
/* Sigmoid backward reuses the forward output y as y * (1 - y). */
ag_tensor* ag_sigmoid(const ag_tensor* value);
/* Tanh backward reuses the forward output y as 1 - y^2. */
ag_tensor* ag_tanh(const ag_tensor* value);
/* Backward differentiates the same tanh approximation used by t_gelu. */
ag_tensor* ag_gelu(const ag_tensor* value);
ag_tensor* ag_reshape(const ag_tensor* value, int new_ndim, const int* new_dims);
ag_tensor* ag_transpose(const ag_tensor* value, int dim0, int dim1);
ag_tensor* ag_slice(const ag_tensor* value, int dim, int start, int end);
ag_tensor* ag_expand(const ag_tensor* value, int new_ndim, const int* new_dims);
ag_tensor* ag_sum(const ag_tensor* value, int dim, int keepdim);
ag_tensor* ag_mean(const ag_tensor* value, int dim, int keepdim);
ag_tensor* ag_max(const ag_tensor* value, int dim, int keepdim);
ag_tensor* ag_matmul(const ag_tensor* a, const ag_tensor* b);
/* Scalar outputs are seeded with one; non-scalars require an explicit seed. */
int ag_backward(ag_tensor* loss);
int ag_backward_with_grad(ag_tensor* output, const tensor* output_gradient);
/* Gradients are owned tensors; zeroing releases them and stores NULL. */
void ag_zero_grad(ag_tensor* value);
void ag_zero_grad_all(ag_tensor* root);

#endif /* TENSORLIB_AUTOGRAD_H */
