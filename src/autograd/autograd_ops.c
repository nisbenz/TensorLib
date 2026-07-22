#include "autograd_internal.h"

static void free_gradients(tensor** gradients, int count) {
    for (int i = 0; i < count; ++i) {
        t_free(gradients[i]);
        gradients[i] = NULL;
    }
}

static int backward_add(const ag_node* node,
                        const tensor* output_gradient,
                        tensor** input_gradients) {
    if (node == NULL || node->input_count != 2 ||
        !tensor_has_valid_metadata(output_gradient) || input_gradients == NULL) return 1;
    for (int i = 0; i < 2; ++i) {
        if (node->inputs[i]->requires_grad) {
            input_gradients[i] = t_clone((tensor*)output_gradient);
            if (input_gradients[i] == NULL) {
                free_gradients(input_gradients, 2);
                return 1;
            }
        }
    }
    return 0;
}

static int backward_sub(const ag_node* node,
                        const tensor* output_gradient,
                        tensor** input_gradients) {
    if (node == NULL || node->input_count != 2 ||
        !tensor_has_valid_metadata(output_gradient) || input_gradients == NULL) return 1;
    if (node->inputs[0]->requires_grad) {
        input_gradients[0] = t_clone((tensor*)output_gradient);
        if (input_gradients[0] == NULL) return 1;
    }
    if (node->inputs[1]->requires_grad) {
        input_gradients[1] = t_neg((tensor*)output_gradient);
        if (input_gradients[1] == NULL) {
            free_gradients(input_gradients, 2);
            return 1;
        }
    }
    return 0;
}

static int backward_mul(const ag_node* node,
                        const tensor* output_gradient,
                        tensor** input_gradients) {
    if (node == NULL || node->input_count != 2 ||
        !tensor_has_valid_metadata(output_gradient) || input_gradients == NULL) return 1;
    if (node->inputs[0]->requires_grad) {
        input_gradients[0] = t_mul((tensor*)output_gradient, node->inputs[1]->value);
        if (input_gradients[0] == NULL) return 1;
    }
    if (node->inputs[1]->requires_grad) {
        input_gradients[1] = t_mul((tensor*)output_gradient, node->inputs[0]->value);
        if (input_gradients[1] == NULL) {
            free_gradients(input_gradients, 2);
            return 1;
        }
    }
    return 0;
}

static int backward_div(const ag_node* node,
                        const tensor* output_gradient,
                        tensor** input_gradients) {
    if (node == NULL || node->input_count != 2 ||
        !tensor_has_valid_metadata(output_gradient) || input_gradients == NULL) return 1;
    if (node->inputs[0]->requires_grad) {
        input_gradients[0] = t_div((tensor*)output_gradient, node->inputs[1]->value);
        if (input_gradients[0] == NULL) return 1;
    }
    if (node->inputs[1]->requires_grad) {
        tensor* denominator = t_mul(node->inputs[1]->value, node->inputs[1]->value);
        tensor* numerator = t_mul((tensor*)output_gradient, node->inputs[0]->value);
        tensor* quotient = (denominator != NULL && numerator != NULL)
                         ? t_div(numerator, denominator) : NULL;
        input_gradients[1] = quotient != NULL ? t_neg(quotient) : NULL;
        t_free(quotient);
        t_free(numerator);
        t_free(denominator);
        if (input_gradients[1] == NULL) {
            free_gradients(input_gradients, 2);
            return 1;
        }
    }
    return 0;
}

typedef tensor* (*binary_forward_fn)(tensor*, tensor*);

static ag_tensor* apply_binary(const ag_tensor* a,
                               const ag_tensor* b,
                               ag_op operation,
                               binary_forward_fn forward,
                               ag_backward_fn backward) {
    if (a == NULL || b == NULL || forward == NULL) return NULL;
    tensor* output = forward(a->value, b->value);
    ag_tensor* inputs[2] = {(ag_tensor*)a, (ag_tensor*)b};
    return ag_make_result(output, operation, 2, inputs, backward, NULL, NULL);
}

ag_tensor* ag_add(const ag_tensor* a, const ag_tensor* b) {
    return apply_binary(a, b, AG_OP_ADD, t_add, backward_add);
}

ag_tensor* ag_sub(const ag_tensor* a, const ag_tensor* b) {
    return apply_binary(a, b, AG_OP_SUB, t_sub, backward_sub);
}

ag_tensor* ag_mul(const ag_tensor* a, const ag_tensor* b) {
    return apply_binary(a, b, AG_OP_MUL, t_mul, backward_mul);
}

ag_tensor* ag_div(const ag_tensor* a, const ag_tensor* b) {
    return apply_binary(a, b, AG_OP_DIV, t_div, backward_div);
}
