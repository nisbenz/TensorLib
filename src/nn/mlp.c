#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_internal.h"

static ag_tensor* nn_mlp_module_forward(const nn_module* module,
                                        const ag_tensor* input)
{
    return nn_mlp_forward((const nn_mlp*)module, input);
}

static void nn_mlp_module_destroy(nn_module* module)
{
    nn_mlp_destroy((nn_mlp*)module);
}

static char* nn_mlp_layer_name(const char* model_name, size_t index)
{
    int required;
    char* result;

    required = snprintf(NULL, 0, "%s.layers.%zu", model_name, index);
    if (required < 0 || (size_t)required == SIZE_MAX) return NULL;
    result = (char*)malloc((size_t)required + 1);
    if (result == NULL) return NULL;
    snprintf(result, (size_t)required + 1, "%s.layers.%zu", model_name, index);
    return result;
}

static int nn_mlp_config_valid(const nn_mlp_config* config)
{
    if (config == NULL || config->input_features <= 0 ||
        config->output_features <= 0 || config->activations == NULL ||
        config->hidden_count == SIZE_MAX ||
        (config->hidden_count > 0 && config->hidden_sizes == NULL)) {
        return 0;
    }
    for (size_t i = 0; i < config->hidden_count; ++i) {
        if (config->hidden_sizes[i] <= 0) return 0;
    }
    return 1;
}

nn_mlp* nn_mlp_create(const char* name,
                      const nn_mlp_config* config,
                      nn_rng* rng)
{
    nn_mlp* model;
    int in_features;

    if (!nn_mlp_config_valid(config)) return NULL;
    model = (nn_mlp*)calloc(1, sizeof(*model));
    if (model == NULL) return NULL;
    if (nn_module_init_base(&model->base,
                            "MLP",
                            name,
                            nn_mlp_module_forward,
                            nn_mlp_module_destroy) != 0) {
        free(model);
        return NULL;
    }
    model->layer_count = config->hidden_count + 1;
    model->activations = (nn_activation*)malloc(
        model->layer_count * sizeof(*model->activations));
    if (model->activations == NULL) goto fail;
    memcpy(model->activations,
           config->activations,
           model->layer_count * sizeof(*model->activations));

    in_features = config->input_features;
    for (size_t i = 0; i < model->layer_count; ++i) {
        int out_features = i < config->hidden_count
            ? config->hidden_sizes[i]
            : config->output_features;
        char* layer_name = nn_mlp_layer_name(name, i);
        nn_linear* layer;

        if (layer_name == NULL) goto fail;
        layer = nn_linear_create(layer_name,
                                 in_features,
                                 out_features,
                                 config->use_bias,
                                 config->weight_init,
                                 config->bias_init,
                                 rng);
        free(layer_name);
        if (layer == NULL) goto fail;
        if (nn_module_register_child(&model->base, &layer->base) != 0) {
            nn_linear_destroy(layer);
            goto fail;
        }
        in_features = out_features;
    }
    return model;

fail:
    free(model->activations);
    model->activations = NULL;
    nn_module_destroy_base(&model->base);
    free(model);
    return NULL;
}

void nn_mlp_destroy(nn_mlp* model)
{
    if (model == NULL) return;
    free(model->activations);
    nn_module_destroy_base(&model->base);
    free(model);
}

ag_tensor* nn_mlp_forward(const nn_mlp* model, const ag_tensor* input)
{
    ag_tensor* current;

    if (model == NULL || input == NULL ||
        model->layer_count != model->base.child_count ||
        (model->layer_count > 0 &&
         (model->activations == NULL || model->base.children == NULL))) {
        return NULL;
    }
    current = (ag_tensor*)input;
    ag_tensor_retain(current);
    for (size_t i = 0; i < model->layer_count; ++i) {
        ag_tensor* next = nn_module_forward(model->base.children[i], current);
        ag_tensor_release(current);
        if (next == NULL) return NULL;
        current = next;
        if (model->activations[i].forward != NULL) {
            next = model->activations[i].forward(
                &model->activations[i], current);
            ag_tensor_release(current);
            if (next == NULL) return NULL;
            current = next;
        }
    }
    return current;
}
