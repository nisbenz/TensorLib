#include <math.h>

#include "../../include/tensorlib/nn.h"

ag_tensor* nn_apply_causal_mask(const ag_tensor* scores)
{
    tensor* mask_value;
    ag_tensor* mask;
    ag_tensor* result;
    int dims[2];
    int sequence;

    if (scores == NULL || !tensor_has_valid_metadata(scores->value) ||
        scores->value->ndim < 2) {
        return NULL;
    }
    sequence = scores->value->dims[scores->value->ndim - 1];
    if (scores->value->dims[scores->value->ndim - 2] != sequence) {
        return NULL;
    }
    dims[0] = sequence;
    dims[1] = sequence;
    mask_value = t_alloc(2, dims);
    if (mask_value == NULL) return NULL;
    for (int row = 0; row < sequence; ++row) {
        for (int column = 0; column < sequence; ++column) {
            mask_value->storage->data[row * sequence + column] =
                column <= row ? 0.0f : -INFINITY;
        }
    }
    mask = ag_from_owned_tensor(mask_value, 0);
    if (mask == NULL) return NULL;
    result = ag_add(scores, mask);
    ag_tensor_release(mask);
    return result;
}
