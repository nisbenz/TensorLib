#include "../../include/tensorlib/nn.h"
#include "../../include/tensorlib/autograd_internal.h"

ag_tensor* nn_log_softmax(const ag_tensor* logits)
{
    return ag_softmax_last_dim(logits, 1, 0);
}

ag_tensor* nn_softmax(const ag_tensor* logits)
{
    return ag_softmax_last_dim(logits, 0, 0);
}

ag_tensor* nn_cross_entropy(const ag_tensor* logits, const tensor* targets)
{
    return ag_cross_entropy(logits, targets);
}
