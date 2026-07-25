#include <math.h>

#include "../../include/tensorlib/nn.h"

static float nn_tensor_at_flat(const tensor* value, int flat)
{
    int index = value->offset;
    int remaining = flat;

    for (int dim = value->ndim - 1; dim >= 0; --dim) {
        int coordinate = remaining % value->dims[dim];
        remaining /= value->dims[dim];
        index += coordinate * value->strides[dim];
    }
    return value->storage->data[index];
}

ag_tensor* nn_log_softmax(const ag_tensor* logits)
{
    ag_tensor* maximum;
    ag_tensor* shifted;
    ag_tensor* exponentials;
    ag_tensor* sum;
    ag_tensor* log_sum;
    ag_tensor* result;
    int axis;

    if (logits == NULL || !tensor_has_valid_metadata(logits->value) ||
        logits->value->ndim < 1) {
        return NULL;
    }
    axis = logits->value->ndim - 1;
    maximum = ag_max(logits, axis, 1);
    if (maximum == NULL) return NULL;
    shifted = ag_sub(logits, maximum);
    ag_tensor_release(maximum);
    if (shifted == NULL) return NULL;
    exponentials = ag_exp(shifted);
    if (exponentials == NULL) {
        ag_tensor_release(shifted);
        return NULL;
    }
    sum = ag_sum(exponentials, axis, 1);
    ag_tensor_release(exponentials);
    if (sum == NULL) {
        ag_tensor_release(shifted);
        return NULL;
    }
    log_sum = ag_log(sum);
    ag_tensor_release(sum);
    if (log_sum == NULL) {
        ag_tensor_release(shifted);
        return NULL;
    }
    result = ag_sub(shifted, log_sum);
    ag_tensor_release(log_sum);
    ag_tensor_release(shifted);
    return result;
}

ag_tensor* nn_softmax(const ag_tensor* logits)
{
    ag_tensor* log_probabilities = nn_log_softmax(logits);
    ag_tensor* probabilities;

    if (log_probabilities == NULL) return NULL;
    probabilities = ag_exp(log_probabilities);
    ag_tensor_release(log_probabilities);
    return probabilities;
}

static int nn_targets_match(const tensor* logits, const tensor* targets)
{
    if (!tensor_has_valid_metadata(targets) ||
        targets->ndim != logits->ndim - 1) {
        return 0;
    }
    for (int i = 0; i < targets->ndim; ++i) {
        if (targets->dims[i] != logits->dims[i]) return 0;
    }
    return 1;
}

ag_tensor* nn_cross_entropy(const ag_tensor* logits, const tensor* targets)
{
    ag_tensor* log_probabilities;
    ag_tensor* selector;
    ag_tensor* selected;
    ag_tensor* class_sum;
    ag_tensor* loss;
    tensor* selector_value;
    int classes;
    int rows;

    if (logits == NULL || !tensor_has_valid_metadata(logits->value) ||
        logits->value->ndim < 1 || !nn_targets_match(logits->value, targets)) {
        return NULL;
    }
    classes = logits->value->dims[logits->value->ndim - 1];
    rows = tensor_numel((tensor*)targets);
    selector_value = t_alloc(logits->value->ndim, logits->value->dims);
    if (selector_value == NULL) return NULL;
    for (int i = 0; i < tensor_numel(selector_value); ++i) {
        selector_value->storage->data[i] = 0.0f;
    }
    for (int row = 0; row < rows; ++row) {
        float target = nn_tensor_at_flat(targets, row);
        int class_index;
        if (!isfinite(target) || floorf(target) != target ||
            target < 0.0f || target >= (float)classes) {
            t_free(selector_value);
            return NULL;
        }
        class_index = (int)target;
        selector_value->storage->data[row * classes + class_index] = 1.0f;
    }
    selector = ag_from_owned_tensor(selector_value, 0);
    if (selector == NULL) return NULL;
    log_probabilities = nn_log_softmax(logits);
    if (log_probabilities == NULL) {
        ag_tensor_release(selector);
        return NULL;
    }
    selected = ag_mul(log_probabilities, selector);
    ag_tensor_release(log_probabilities);
    ag_tensor_release(selector);
    if (selected == NULL) return NULL;
    class_sum = ag_sum(selected, selected->value->ndim - 1, 0);
    ag_tensor_release(selected);
    if (class_sum == NULL) return NULL;
    loss = ag_neg(class_sum);
    ag_tensor_release(class_sum);
    if (loss == NULL) return NULL;
    while (loss->value->ndim > 0) {
        ag_tensor* reduced = ag_mean(loss, 0, 0);
        ag_tensor_release(loss);
        if (reduced == NULL) return NULL;
        loss = reduced;
    }
    return loss;
}
