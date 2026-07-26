#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_internal.h"

static ag_tensor* decoder_block_module_forward(const nn_module* module,
                                               const ag_tensor* input)
{
    return nn_decoder_block_forward((const nn_decoder_block*)module, input);
}

static void decoder_block_module_destroy(nn_module* module)
{
    nn_decoder_block_destroy((nn_decoder_block*)module);
}

static char* block_child_name(const char* name, const char* suffix)
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

static int register_block_child(nn_module* parent, nn_module* child)
{
    if (child == NULL) return -1;
    if (nn_module_register_child(parent, child) != 0) {
        child->destroy(child);
        return -1;
    }
    return 0;
}

nn_decoder_block* nn_decoder_block_create(
    const char* name,
    int channels,
    int head_count,
    float dropout_probability,
    float layer_norm_epsilon,
    nn_rng* rng)
{
    nn_decoder_block* block;
    char* child_name;
    int hidden_width;

    if (channels <= 0 || channels > INT_MAX / 4 ||
        head_count <= 0 || channels % head_count != 0 || rng == NULL) {
        return NULL;
    }
    hidden_width = channels * 4;
    block = (nn_decoder_block*)calloc(1, sizeof(*block));
    if (block == NULL) return NULL;
    if (nn_module_init_base(&block->base,
                            "DecoderBlock",
                            name,
                            decoder_block_module_forward,
                            decoder_block_module_destroy) != 0) {
        free(block);
        return NULL;
    }
    block->channels = channels;

    child_name = block_child_name(name, "attention_norm");
    if (child_name == NULL) goto fail;
    block->attention_norm = nn_layer_norm_create(
        child_name, channels, layer_norm_epsilon, 1);
    free(child_name);
    if (register_block_child(
        &block->base,
        block->attention_norm == NULL ? NULL :
        &block->attention_norm->base) != 0) {
        block->attention_norm = NULL;
        goto fail;
    }

    child_name = block_child_name(name, "attention");
    if (child_name == NULL) goto fail;
    block->attention = nn_multihead_attention_create(
        child_name, channels, head_count, dropout_probability, rng);
    free(child_name);
    if (register_block_child(
        &block->base,
        block->attention == NULL ? NULL : &block->attention->base) != 0) {
        block->attention = NULL;
        goto fail;
    }

    child_name = block_child_name(name, "mlp_norm");
    if (child_name == NULL) goto fail;
    block->mlp_norm = nn_layer_norm_create(
        child_name, channels, layer_norm_epsilon, 1);
    free(child_name);
    if (register_block_child(
        &block->base,
        block->mlp_norm == NULL ? NULL : &block->mlp_norm->base) != 0) {
        block->mlp_norm = NULL;
        goto fail;
    }

    child_name = block_child_name(name, "mlp_input");
    if (child_name == NULL) goto fail;
    block->mlp_input = nn_linear_create(
        child_name, channels, hidden_width, 1,
        NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO, rng);
    free(child_name);
    if (register_block_child(
        &block->base,
        block->mlp_input == NULL ? NULL : &block->mlp_input->base) != 0) {
        block->mlp_input = NULL;
        goto fail;
    }

    child_name = block_child_name(name, "mlp_output");
    if (child_name == NULL) goto fail;
    block->mlp_output = nn_linear_create(
        child_name, hidden_width, channels, 1,
        NN_INIT_XAVIER_UNIFORM, NN_INIT_ZERO, rng);
    free(child_name);
    if (register_block_child(
        &block->base,
        block->mlp_output == NULL ? NULL : &block->mlp_output->base) != 0) {
        block->mlp_output = NULL;
        goto fail;
    }

    child_name = block_child_name(name, "mlp_dropout");
    if (child_name == NULL) goto fail;
    block->mlp_dropout = nn_dropout_create(
        child_name, dropout_probability, rng);
    free(child_name);
    if (register_block_child(
        &block->base,
        block->mlp_dropout == NULL ? NULL : &block->mlp_dropout->base) != 0) {
        block->mlp_dropout = NULL;
        goto fail;
    }
    return block;

fail:
    nn_module_destroy_base(&block->base);
    free(block);
    return NULL;
}

void nn_decoder_block_destroy(nn_decoder_block* block)
{
    if (block == NULL) return;
    nn_module_destroy_base(&block->base);
    free(block);
}

ag_tensor* nn_decoder_block_forward(
    const nn_decoder_block* block,
    const ag_tensor* input)
{
    ag_tensor* normalized_attention = NULL;
    ag_tensor* attention_output = NULL;
    ag_tensor* attention_residual = NULL;
    ag_tensor* normalized_mlp = NULL;
    ag_tensor* hidden = NULL;
    ag_tensor* activated = NULL;
    ag_tensor* projected = NULL;
    ag_tensor* dropped = NULL;
    ag_tensor* result = NULL;

    if (block == NULL || input == NULL ||
        !tensor_has_valid_metadata(input->value) ||
        input->value->ndim != 3 ||
        input->value->dims[2] != block->channels ||
        block->attention_norm == NULL || block->attention == NULL ||
        block->mlp_norm == NULL || block->mlp_input == NULL ||
        block->mlp_output == NULL || block->mlp_dropout == NULL) {
        return NULL;
    }
    normalized_attention = nn_layer_norm_forward(
        block->attention_norm, input);
    if (normalized_attention == NULL) goto cleanup;
    attention_output = nn_multihead_attention_forward(
        block->attention, normalized_attention);
    if (attention_output == NULL) goto cleanup;
    attention_residual = ag_add(input, attention_output);
    if (attention_residual == NULL) goto cleanup;
    normalized_mlp = nn_layer_norm_forward(
        block->mlp_norm, attention_residual);
    if (normalized_mlp == NULL) goto cleanup;
    hidden = nn_linear_forward(block->mlp_input, normalized_mlp);
    if (hidden == NULL) goto cleanup;
    activated = ag_gelu(hidden);
    if (activated == NULL) goto cleanup;
    projected = nn_linear_forward(block->mlp_output, activated);
    if (projected == NULL) goto cleanup;
    dropped = nn_dropout_forward(block->mlp_dropout, projected);
    if (dropped == NULL) goto cleanup;
    result = ag_add(attention_residual, dropped);

cleanup:
    ag_tensor_release(dropped);
    ag_tensor_release(projected);
    ag_tensor_release(activated);
    ag_tensor_release(hidden);
    ag_tensor_release(normalized_mlp);
    ag_tensor_release(attention_residual);
    ag_tensor_release(attention_output);
    ag_tensor_release(normalized_attention);
    return result;
}
