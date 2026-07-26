#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_internal.h"

static ag_tensor* attention_module_forward(const nn_module* module,
                                           const ag_tensor* input)
{
    return nn_multihead_attention_forward(
        (const nn_multihead_attention*)module, input);
}

static void attention_module_destroy(nn_module* module)
{
    nn_multihead_attention_destroy((nn_multihead_attention*)module);
}

static char* attention_child_name(const char* name, const char* suffix)
{
    int required;
    char* result;

    if (name == NULL || suffix == NULL) return NULL;
    required = snprintf(NULL, 0, "%s.%s", name, suffix);
    if (required < 0 || (size_t)required == SIZE_MAX) return NULL;
    result = (char*)malloc((size_t)required + 1);
    if (result != NULL) {
        snprintf(result, (size_t)required + 1, "%s.%s", name, suffix);
    }
    return result;
}

static nn_linear* create_projection(const char* name,
                                    const char* suffix,
                                    int channels,
                                    nn_rng* rng)
{
    char* child_name = attention_child_name(name, suffix);
    nn_linear* result;

    if (child_name == NULL) return NULL;
    result = nn_linear_create(child_name,
                              channels,
                              channels,
                              1,
                              NN_INIT_XAVIER_UNIFORM,
                              NN_INIT_ZERO,
                              rng);
    free(child_name);
    return result;
}

static int register_child(nn_module* parent, nn_module* child)
{
    if (child == NULL) return -1;
    if (nn_module_register_child(parent, child) != 0) {
        child->destroy(child);
        return -1;
    }
    return 0;
}

nn_multihead_attention* nn_multihead_attention_create(
    const char* name,
    int channels,
    int head_count,
    float dropout_probability,
    nn_rng* rng)
{
    nn_multihead_attention* attention;
    char* dropout_name;

    if (channels <= 0 || head_count <= 0 || channels % head_count != 0 ||
        rng == NULL) {
        return NULL;
    }
    attention = (nn_multihead_attention*)calloc(1, sizeof(*attention));
    if (attention == NULL) return NULL;
    if (nn_module_init_base(&attention->base,
                            "MultiheadAttention",
                            name,
                            attention_module_forward,
                            attention_module_destroy) != 0) {
        free(attention);
        return NULL;
    }
    attention->channels = channels;
    attention->head_count = head_count;
    attention->head_width = channels / head_count;

    attention->query = create_projection(name, "query", channels, rng);
    if (register_child(&attention->base,
                       attention->query == NULL ? NULL :
                       &attention->query->base) != 0) {
        attention->query = NULL;
        goto fail;
    }
    attention->key = create_projection(name, "key", channels, rng);
    if (register_child(&attention->base,
                       attention->key == NULL ? NULL :
                       &attention->key->base) != 0) {
        attention->key = NULL;
        goto fail;
    }
    attention->value = create_projection(name, "value", channels, rng);
    if (register_child(&attention->base,
                       attention->value == NULL ? NULL :
                       &attention->value->base) != 0) {
        attention->value = NULL;
        goto fail;
    }
    attention->output = create_projection(name, "output", channels, rng);
    if (register_child(&attention->base,
                       attention->output == NULL ? NULL :
                       &attention->output->base) != 0) {
        attention->output = NULL;
        goto fail;
    }
    dropout_name = attention_child_name(name, "output_dropout");
    if (dropout_name == NULL) goto fail;
    attention->output_dropout = nn_dropout_create(
        dropout_name, dropout_probability, rng);
    free(dropout_name);
    if (register_child(&attention->base,
                       attention->output_dropout == NULL ? NULL :
                       &attention->output_dropout->base) != 0) {
        attention->output_dropout = NULL;
        goto fail;
    }
    return attention;

fail:
    nn_module_destroy_base(&attention->base);
    free(attention);
    return NULL;
}

void nn_multihead_attention_destroy(nn_multihead_attention* attention)
{
    if (attention == NULL) return;
    nn_module_destroy_base(&attention->base);
    free(attention);
}

static ag_tensor* project_heads(const nn_linear* projection,
                                const ag_tensor* input,
                                int batch,
                                int time,
                                int heads,
                                int width)
{
    int shape[4] = {batch, time, heads, width};
    ag_tensor* projected = nn_linear_forward(projection, input);
    ag_tensor* reshaped;
    ag_tensor* transposed;

    if (projected == NULL) return NULL;
    reshaped = ag_reshape(projected, 4, shape);
    ag_tensor_release(projected);
    if (reshaped == NULL) return NULL;
    transposed = ag_transpose(reshaped, 1, 2);
    ag_tensor_release(reshaped);
    return transposed;
}

ag_tensor* nn_multihead_attention_forward(
    const nn_multihead_attention* attention,
    const ag_tensor* input)
{
    ag_tensor* query = NULL;
    ag_tensor* key = NULL;
    ag_tensor* value = NULL;
    ag_tensor* transposed_key = NULL;
    ag_tensor* scores = NULL;
    ag_tensor* scaled = NULL;
    ag_tensor* masked = NULL;
    ag_tensor* probabilities = NULL;
    ag_tensor* context = NULL;
    ag_tensor* transposed_context = NULL;
    ag_tensor* merged = NULL;
    ag_tensor* projected = NULL;
    ag_tensor* result = NULL;
    int merged_shape[3];
    int batch;
    int time;

    if (attention == NULL || input == NULL ||
        !tensor_has_valid_metadata(input->value) ||
        input->value->ndim != 3 ||
        input->value->dims[2] != attention->channels ||
        attention->query == NULL || attention->key == NULL ||
        attention->value == NULL || attention->output == NULL ||
        attention->output_dropout == NULL) {
        return NULL;
    }
    batch = input->value->dims[0];
    time = input->value->dims[1];
    query = project_heads(attention->query, input, batch, time,
                          attention->head_count, attention->head_width);
    if (query == NULL) goto cleanup;
    key = project_heads(attention->key, input, batch, time,
                        attention->head_count, attention->head_width);
    if (key == NULL) goto cleanup;
    value = project_heads(attention->value, input, batch, time,
                          attention->head_count, attention->head_width);
    if (value == NULL) goto cleanup;
    transposed_key = ag_transpose(key, 2, 3);
    if (transposed_key == NULL) goto cleanup;
    scores = ag_matmul(query, transposed_key);
    if (scores == NULL) goto cleanup;
    scaled = ag_div_scalar(scores, sqrtf((float)attention->head_width));
    if (scaled == NULL) goto cleanup;
    masked = nn_apply_causal_mask(scaled);
    if (masked == NULL) goto cleanup;
    probabilities = nn_softmax(masked);
    if (probabilities == NULL) goto cleanup;
    context = ag_matmul(probabilities, value);
    if (context == NULL) goto cleanup;
    transposed_context = ag_transpose(context, 1, 2);
    if (transposed_context == NULL) goto cleanup;
    merged_shape[0] = batch;
    merged_shape[1] = time;
    merged_shape[2] = attention->channels;
    merged = ag_reshape(transposed_context, 3, merged_shape);
    if (merged == NULL) goto cleanup;
    projected = nn_linear_forward(attention->output, merged);
    if (projected == NULL) goto cleanup;
    result = nn_dropout_forward(attention->output_dropout, projected);

cleanup:
    ag_tensor_release(projected);
    ag_tensor_release(merged);
    ag_tensor_release(transposed_context);
    ag_tensor_release(context);
    ag_tensor_release(probabilities);
    ag_tensor_release(masked);
    ag_tensor_release(scaled);
    ag_tensor_release(scores);
    ag_tensor_release(transposed_key);
    ag_tensor_release(value);
    ag_tensor_release(key);
    ag_tensor_release(query);
    return result;
}
