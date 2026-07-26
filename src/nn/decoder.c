#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_internal.h"

static ag_tensor* decoder_module_forward(const nn_module* module,
                                         const ag_tensor* input)
{
    return nn_decoder_forward((const nn_decoder*)module, input);
}

static void decoder_module_destroy(nn_module* module)
{
    nn_decoder_destroy((nn_decoder*)module);
}

static char* decoder_child_name(const char* name, const char* suffix)
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

static int decoder_config_valid(const nn_decoder_config* config)
{
    return config != NULL &&
           config->vocabulary_size > 0 &&
           config->context_length > 0 &&
           config->channels > 0 &&
           config->head_count > 0 &&
           config->channels % config->head_count == 0 &&
           isfinite(config->dropout_probability) &&
           config->dropout_probability >= 0.0f &&
           config->dropout_probability < 1.0f &&
           isfinite(config->layer_norm_epsilon) &&
           config->layer_norm_epsilon > 0.0f;
}

static int register_decoder_child(nn_module* parent, nn_module* child)
{
    if (child == NULL) return -1;
    if (nn_module_register_child(parent, child) != 0) {
        child->destroy(child);
        return -1;
    }
    return 0;
}

nn_decoder* nn_decoder_create(const char* name,
                              const nn_decoder_config* config,
                              nn_rng* rng)
{
    nn_decoder* decoder;
    char* child_name;

    if (!decoder_config_valid(config) || rng == NULL) return NULL;
    decoder = (nn_decoder*)calloc(1, sizeof(*decoder));
    if (decoder == NULL) return NULL;
    if (nn_module_init_base(&decoder->base,
                            "Decoder",
                            name,
                            decoder_module_forward,
                            decoder_module_destroy) != 0) {
        free(decoder);
        return NULL;
    }
    decoder->config = *config;

    child_name = decoder_child_name(name, "token_embedding");
    if (child_name == NULL) goto fail;
    decoder->token_embedding = nn_embedding_create(
        child_name,
        config->vocabulary_size,
        config->channels,
        NN_INIT_XAVIER_UNIFORM,
        rng);
    free(child_name);
    if (register_decoder_child(
        &decoder->base,
        decoder->token_embedding == NULL ? NULL :
        &decoder->token_embedding->base) != 0) {
        decoder->token_embedding = NULL;
        goto fail;
    }

    child_name = decoder_child_name(name, "position_embedding");
    if (child_name == NULL) goto fail;
    decoder->positional_embedding = nn_positional_embedding_create(
        child_name,
        config->context_length,
        config->channels,
        NN_INIT_XAVIER_UNIFORM,
        rng);
    free(child_name);
    if (register_decoder_child(
        &decoder->base,
        decoder->positional_embedding == NULL ? NULL :
        &decoder->positional_embedding->base) != 0) {
        decoder->positional_embedding = NULL;
        goto fail;
    }

    for (int index = 0; index < 2; ++index) {
        const char* suffix = index == 0 ? "blocks.0" : "blocks.1";
        child_name = decoder_child_name(name, suffix);
        if (child_name == NULL) goto fail;
        decoder->blocks[index] = nn_decoder_block_create(
            child_name,
            config->channels,
            config->head_count,
            config->dropout_probability,
            config->layer_norm_epsilon,
            rng);
        free(child_name);
        if (register_decoder_child(
            &decoder->base,
            decoder->blocks[index] == NULL ? NULL :
            &decoder->blocks[index]->base) != 0) {
            decoder->blocks[index] = NULL;
            goto fail;
        }
    }

    child_name = decoder_child_name(name, "final_norm");
    if (child_name == NULL) goto fail;
    decoder->final_norm = nn_layer_norm_create(
        child_name, config->channels, config->layer_norm_epsilon, 1);
    free(child_name);
    if (register_decoder_child(
        &decoder->base,
        decoder->final_norm == NULL ? NULL :
        &decoder->final_norm->base) != 0) {
        decoder->final_norm = NULL;
        goto fail;
    }

    child_name = decoder_child_name(name, "language_model_head");
    if (child_name == NULL) goto fail;
    decoder->language_model_head = nn_linear_create(
        child_name,
        config->channels,
        config->vocabulary_size,
        1,
        NN_INIT_XAVIER_UNIFORM,
        NN_INIT_ZERO,
        rng);
    free(child_name);
    if (register_decoder_child(
        &decoder->base,
        decoder->language_model_head == NULL ? NULL :
        &decoder->language_model_head->base) != 0) {
        decoder->language_model_head = NULL;
        goto fail;
    }
    return decoder;

fail:
    nn_module_destroy_base(&decoder->base);
    free(decoder);
    return NULL;
}

void nn_decoder_destroy(nn_decoder* decoder)
{
    if (decoder == NULL) return;
    nn_module_destroy_base(&decoder->base);
    free(decoder);
}

static int token_input_valid(const nn_decoder* decoder,
                             const ag_tensor* token_ids)
{
    return decoder != NULL && token_ids != NULL &&
           !token_ids->requires_grad &&
           tensor_has_valid_metadata(token_ids->value) &&
           token_ids->value->ndim == 2 &&
           token_ids->value->dims[1] <= decoder->config.context_length;
}

ag_tensor* nn_decoder_forward(const nn_decoder* decoder,
                              const ag_tensor* token_ids)
{
    ag_tensor* tokens = NULL;
    ag_tensor* current = NULL;
    ag_tensor* next = NULL;
    ag_tensor* normalized = NULL;
    ag_tensor* result = NULL;

    if (!token_input_valid(decoder, token_ids) ||
        decoder->token_embedding == NULL ||
        decoder->positional_embedding == NULL ||
        decoder->blocks[0] == NULL || decoder->blocks[1] == NULL ||
        decoder->final_norm == NULL ||
        decoder->language_model_head == NULL) {
        return NULL;
    }
    tokens = nn_embedding_forward(decoder->token_embedding, token_ids);
    if (tokens == NULL) goto cleanup;
    current = nn_positional_embedding_forward(
        decoder->positional_embedding, tokens);
    if (current == NULL) goto cleanup;
    for (int index = 0; index < 2; ++index) {
        next = nn_decoder_block_forward(decoder->blocks[index], current);
        if (next == NULL) goto cleanup;
        ag_tensor_release(current);
        current = next;
        next = NULL;
    }
    normalized = nn_layer_norm_forward(decoder->final_norm, current);
    if (normalized == NULL) goto cleanup;
    result = nn_linear_forward(decoder->language_model_head, normalized);

cleanup:
    ag_tensor_release(normalized);
    ag_tensor_release(next);
    ag_tensor_release(current);
    ag_tensor_release(tokens);
    return result;
}

ag_tensor* nn_decoder_loss(const nn_decoder* decoder,
                           const ag_tensor* token_ids,
                           const tensor* targets)
{
    ag_tensor* logits;
    ag_tensor* flattened_logits;
    ag_tensor* loss;
    tensor* flattened_targets;
    int rows;
    int logit_dims[2];
    int target_dims[1];

    if (!token_input_valid(decoder, token_ids) ||
        !tensor_has_valid_metadata(targets) ||
        targets->ndim != 2 ||
        targets->dims[0] != token_ids->value->dims[0] ||
        targets->dims[1] != token_ids->value->dims[1]) {
        return NULL;
    }
    logits = nn_decoder_forward(decoder, token_ids);
    if (logits == NULL) return NULL;
    rows = tensor_numel((tensor*)targets);
    logit_dims[0] = rows;
    logit_dims[1] = decoder->config.vocabulary_size;
    target_dims[0] = rows;
    flattened_logits = ag_reshape(logits, 2, logit_dims);
    ag_tensor_release(logits);
    if (flattened_logits == NULL) return NULL;
    flattened_targets = t_reshape((tensor*)targets, 1, target_dims);
    if (flattened_targets == NULL) {
        ag_tensor_release(flattened_logits);
        return NULL;
    }
    loss = nn_cross_entropy(flattened_logits, flattened_targets);
    t_free(flattened_targets);
    ag_tensor_release(flattened_logits);
    return loss;
}
