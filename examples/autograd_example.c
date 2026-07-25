#include <stdio.h>
#include <stdlib.h>

#include "./../include/tensorlib/autograd.h"

typedef struct {
    const ag_tensor* value;
    const char* name;
} named_tensor;

static const char* operation_name(ag_op operation) {
    switch (operation) {
        case AG_OP_ADD: return "ADD";
        case AG_OP_SUB: return "SUB";
        case AG_OP_MUL: return "MUL";
        case AG_OP_DIV: return "DIV";
        case AG_OP_NEG: return "NEG";
        case AG_OP_EXP: return "EXP";
        case AG_OP_LOG: return "LOG";
        case AG_OP_POW: return "POW";
        case AG_OP_SQRT: return "SQRT";
        case AG_OP_RELU: return "RELU";
        case AG_OP_MATMUL: return "MATMUL";
        case AG_OP_SUM: return "SUM";
        case AG_OP_MEAN: return "MEAN";
        case AG_OP_MAX: return "MAX";
        case AG_OP_RESHAPE: return "RESHAPE";
        case AG_OP_TRANSPOSE: return "TRANSPOSE";
        case AG_OP_SLICE: return "SLICE";
        case AG_OP_EXPAND: return "EXPAND";
        default: return "UNKNOWN";
    }
}

static const char* tensor_name(const ag_tensor* value,
                               const named_tensor* names,
                               int name_count) {
    for (int i = 0; i < name_count; ++i) {
        if (names[i].value == value) return names[i].name;
    }
    return "temporary";
}

static void print_indent(int depth) {
    for (int i = 0; i < depth; ++i) printf("  ");
}

static void print_shape(const tensor* value) {
    if (value->ndim == 0) {
        printf("scalar");
        return;
    }
    printf("[");
    for (int axis = 0; axis < value->ndim; ++axis) {
        printf("%d%s", value->dims[axis], axis + 1 == value->ndim ? "" : ", ");
    }
    printf("]");
}

static void print_values(const tensor* value) {
    int* coords = value->ndim > 0
                ? (int*)calloc((size_t)value->ndim, sizeof(int)) : NULL;
    if (value->ndim > 0 && coords == NULL) {
        printf("<allocation failed>");
        return;
    }
    printf("[");
    for (int i = 0; i < tensor_numel((tensor*)value); ++i) {
        int index = get_flat_index_nd((tensor*)value, coords);
        printf("%.6f%s", value->storage->data[index],
               i + 1 == tensor_numel((tensor*)value) ? "" : ", ");
        advance_coords(coords, value->dims, value->ndim);
    }
    printf("]");
    free(coords);
}

static void print_tracked_tensor(const ag_tensor* value,
                                 const char* name) {
    printf("%-12s shape=", name);
    print_shape(value->value);
    printf(" requires_grad=%s creator=%s\n",
           value->requires_grad ? "true" : "false",
           value->creator == NULL ? "leaf" : operation_name(value->creator->operation));
    printf("  value: ");
    print_values(value->value);
    printf("\n  grad:  ");
    if (value->grad == NULL) printf("<not computed>");
    else print_values(value->grad);
    printf("\n");
}

static void print_graph(const ag_tensor* value,
                        const named_tensor* names,
                        int name_count,
                        int depth) {
    print_indent(depth);
    printf("tensor %s ", tensor_name(value, names, name_count));
    print_shape(value->value);
    printf(" requires_grad=%s\n", value->requires_grad ? "true" : "false");
    if (value->creator == NULL) {
        print_indent(depth + 1);
        printf("leaf tensor\n");
        return;
    }

    print_indent(depth + 1);
    printf("created by %s (%d input%s)\n",
           operation_name(value->creator->operation),
           value->creator->input_count,
           value->creator->input_count == 1 ? "" : "s");
    for (int i = 0; i < value->creator->input_count; ++i) {
        print_indent(depth + 1);
        printf("input[%d]:\n", i);
        print_graph(value->creator->inputs[i], names, name_count, depth + 2);
    }
}

static ag_tensor* make_tensor(int ndim,
                              const int* dims,
                              const float* values,
                              int requires_grad) {
    tensor* raw = t_alloc(ndim, dims);
    if (raw == NULL) return NULL;
    for (int i = 0; i < tensor_numel(raw); ++i) raw->storage->data[i] = values[i];
    return ag_from_owned_tensor(raw, requires_grad);
}

int main(void) {
    int input_dims[2] = {2, 3};
    int weight_dims[2] = {3, 2};
    int bias_dims[1] = {2};
    float input_values[6] = {1.0f, 0.5f, -1.0f, 2.0f, -0.5f, 1.0f};
    float weight_values[6] = {0.10f, -0.20f, 0.30f, 0.25f, -0.15f, 0.40f};
    float bias_values[2] = {0.05f, -0.10f};

    ag_tensor* input = make_tensor(2, input_dims, input_values, 0);
    ag_tensor* weights = make_tensor(2, weight_dims, weight_values, 1);
    ag_tensor* bias = make_tensor(1, bias_dims, bias_values, 1);
    ag_tensor* linear = NULL;
    ag_tensor* shifted = NULL;
    ag_tensor* prediction = NULL;
    ag_tensor* sample_mean = NULL;
    ag_tensor* loss = NULL;
    int status = 1;

    if (input == NULL || weights == NULL || bias == NULL) goto cleanup;
    linear = ag_matmul(input, weights);
    shifted = ag_add(linear, bias);
    prediction = ag_exp(shifted);
    sample_mean = ag_mean(prediction, 1, 0);
    loss = ag_mean(sample_mean, 0, 0);
    if (linear == NULL || shifted == NULL || prediction == NULL ||
        sample_mean == NULL || loss == NULL) goto cleanup;

    named_tensor names[] = {
        {input, "input"}, {weights, "weights"}, {bias, "bias"},
        {linear, "linear"}, {shifted, "shifted"},
        {prediction, "prediction"}, {sample_mean, "sample_mean"},
        {loss, "loss"}
    };
    int name_count = (int)(sizeof(names) / sizeof(names[0]));

    printf("=== Tracked computation graph ===\n");
    print_graph(loss, names, name_count, 0);

    printf("\n=== Tensors before backward ===\n");
    for (int i = 0; i < name_count; ++i) {
        print_tracked_tensor(names[i].value, names[i].name);
    }

    if (ag_backward(loss) != 0) goto cleanup;

    printf("\n=== Tensors after backward ===\n");
    for (int i = 0; i < name_count; ++i) {
        print_tracked_tensor(names[i].value, names[i].name);
    }

    printf("\nThe input participates in the DAG but has no gradient because "
           "requires_grad=false.\n");
    printf("Weights and bias are leaves, so their gradients are ready for an optimizer.\n");
    status = 0;

cleanup:
    ag_tensor_release(loss);
    ag_tensor_release(sample_mean);
    ag_tensor_release(prediction);
    ag_tensor_release(shifted);
    ag_tensor_release(linear);
    ag_tensor_release(bias);
    ag_tensor_release(weights);
    ag_tensor_release(input);
    if (status != 0) fprintf(stderr, "ERROR: failed to build or backpropagate the example graph.\n");
    return status;
}
