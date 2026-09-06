#include <stdlib.h>

#include "./../../include/tensorlib/autograd_internal.h"
#include "../tensor/tensor_matmul_internal.h"

typedef struct {
    tensor_matmul_packed_rhs* backward_rhs;
} matmul_context;

static void free_matmul_context(void* opaque)
{
    matmul_context* context = (matmul_context*)opaque;
    if (context == NULL) return;
    t_free_matmul_packed_rhs(context->backward_rhs);
    free(context);
}

static tensor* promote_operand(tensor* value, int left_operand) {
    if (value->ndim != 1) return t_contiguous(value);
    return t_unsqueeze(value, left_operand ? 0 : 1);
}

static tensor* promote_output_gradient(const tensor* gradient,
                                       int left_vector,
                                       int right_vector) {
    if (left_vector && right_vector) {
        int dims[2] = {1, 1};
        return t_reshape((tensor*)gradient, 2, dims);
    }
    if (left_vector) return t_unsqueeze((tensor*)gradient, gradient->ndim - 1);
    if (right_vector) return t_unsqueeze((tensor*)gradient, gradient->ndim);
    return t_contiguous((tensor*)gradient);
}

static tensor* remove_vector_dimension(tensor* gradient, int left_operand) {
    int dim = left_operand ? gradient->ndim - 2 : gradient->ndim - 1;
    tensor* squeezed = t_squeeze(gradient, dim);
    t_free(gradient);
    return squeezed;
}

static int backward_matmul(const ag_node* node,
                           const tensor* output_gradient,
                           tensor** input_gradients) {
    if (node == NULL || node->input_count != 2 || input_gradients == NULL ||
        !tensor_has_valid_metadata(output_gradient)) return 1;

    tensor* a = node->inputs[0]->value;
    tensor* b = node->inputs[1]->value;
    int a_vector = a->ndim == 1;
    int b_vector = b->ndim == 1;
    tensor* promoted_a = promote_operand(a, 1);
    tensor* promoted_b = promote_operand(b, 0);
    tensor* promoted_gradient = promote_output_gradient(output_gradient, a_vector, b_vector);
    tensor* transposed_a = promoted_a != NULL
                         ? t_transpose(promoted_a, promoted_a->ndim - 2, promoted_a->ndim - 1)
                         : NULL;
    tensor* transposed_b = promoted_b != NULL
                         ? t_transpose(promoted_b, promoted_b->ndim - 2, promoted_b->ndim - 1)
                         : NULL;
    if (promoted_a == NULL || promoted_b == NULL || promoted_gradient == NULL ||
        transposed_a == NULL || transposed_b == NULL) goto fail;

    matmul_context* context = (matmul_context*)node->context;
    if (node->inputs[0]->requires_grad) {
        input_gradients[0] = context != NULL && context->backward_rhs != NULL
                            ? t_matmul_packed_rhs(promoted_gradient,
                                                  context->backward_rhs)
                            : t_matmul(promoted_gradient, transposed_b);
        if (input_gradients[0] == NULL) goto fail;
        if (a_vector) {
            input_gradients[0] = remove_vector_dimension(input_gradients[0], 1);
            if (input_gradients[0] == NULL) goto fail;
        }
    }
    if (node->inputs[1]->requires_grad) {
        input_gradients[1] = b->ndim == 2
                           ? tensor_matmul_backward_rhs(a, output_gradient, b)
                           : t_matmul(transposed_a, promoted_gradient);
        if (input_gradients[1] == NULL) goto fail;
        if (b_vector) {
            input_gradients[1] = remove_vector_dimension(input_gradients[1], 0);
            if (input_gradients[1] == NULL) goto fail;
        }
    }

    t_free(transposed_b); t_free(transposed_a); t_free(promoted_gradient);
    t_free(promoted_b); t_free(promoted_a);
    return 0;

fail:
    t_free(input_gradients[1]); input_gradients[1] = NULL;
    t_free(input_gradients[0]); input_gradients[0] = NULL;
    t_free(transposed_b); t_free(transposed_a); t_free(promoted_gradient);
    t_free(promoted_b); t_free(promoted_a);
    return 1;
}

static ag_tensor* make_matmul_result(const ag_tensor* a,
                                     const ag_tensor* b,
                                     tensor* output,
                                     matmul_context* context) {
    ag_tensor* inputs[2] = {(ag_tensor*)a, (ag_tensor*)b};
    return ag_make_result(output, AG_OP_MATMUL, 2, inputs,
                          backward_matmul, context,
                          context == NULL ? NULL : free_matmul_context);
}

ag_tensor* ag_matmul(const ag_tensor* a, const ag_tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    return make_matmul_result(a, b, t_matmul(a->value, b->value), NULL);
}

ag_tensor* ag_matmul_packed_rhs(const ag_tensor* a,
                                const ag_tensor* b,
                                const tensor_matmul_packed_rhs* packed_rhs) {
    return ag_matmul_packed_rhs_with_backward_pack(a, b, packed_rhs, NULL);
}

ag_tensor* ag_matmul_packed_rhs_with_backward_pack(
    const ag_tensor* a,
    const ag_tensor* b,
    const tensor_matmul_packed_rhs* packed_rhs,
    const tensor_matmul_packed_rhs* backward_rhs)
{
    matmul_context* context = NULL;
    if (a == NULL || b == NULL || packed_rhs == NULL) return NULL;
    if (backward_rhs != NULL) {
        context = (matmul_context*)calloc(1, sizeof(*context));
        if (context == NULL) return NULL;
        context->backward_rhs = (tensor_matmul_packed_rhs*)backward_rhs;
        t_retain_matmul_packed_rhs(context->backward_rhs);
    }
    return make_matmul_result(a, b,
                              t_matmul_packed_rhs(a->value, packed_rhs),
                              context);
}
