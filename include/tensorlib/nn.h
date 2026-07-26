#ifndef TENSORLIB_NN_H
#define TENSORLIB_NN_H

#include <stddef.h>
#include <stdint.h>

#include "autograd.h"


/* Forward declarations. */
typedef struct nn_rng nn_rng;
typedef struct nn_parameter nn_parameter;
typedef struct nn_activation nn_activation;
typedef struct nn_module nn_module;
typedef struct nn_linear nn_linear;
typedef struct nn_embedding nn_embedding;
typedef struct nn_layer_norm nn_layer_norm;
typedef struct nn_mlp nn_mlp;
typedef struct nn_mlp_config nn_mlp_config;
typedef struct nn_sgd nn_sgd;


/* Deterministic random-number generator state. */
struct nn_rng {
    uint64_t state;
};


/* Weight and bias initialization policies. */
typedef enum {
    NN_INIT_ZERO,
    NN_INIT_XAVIER_UNIFORM,
    NN_INIT_XAVIER_NORMAL,
    NN_INIT_HE_UNIFORM,
    NN_INIT_HE_NORMAL,
    NN_INIT_ONE
} nn_init_kind;


struct nn_parameter {
    char* name;
    ag_tensor* value;
    int trainable;
};



typedef ag_tensor* (*nn_activation_forward_fn)(
    const nn_activation* activation,
    const ag_tensor* input
);

struct nn_activation {
    const char* name;
    nn_activation_forward_fn forward;
    const void* context;
};


/* Generic module callbacks. */
typedef ag_tensor* (*nn_module_forward_fn)(
    const nn_module* module,
    const ag_tensor* input
);

typedef void (*nn_module_destroy_fn)(nn_module* module);


struct nn_module {
    const char* type_name;
    char* name;

    nn_module_forward_fn forward;
    nn_module_destroy_fn destroy;

    nn_parameter** parameters;
    size_t parameter_count;
    size_t parameter_capacity;

    nn_module** children;
    size_t child_count;
    size_t child_capacity;

    int training;
};


struct nn_linear {
    nn_module base;

    nn_parameter* weight;
    nn_parameter* bias;

    int in_features;
    int out_features;
    int use_bias;
};

struct nn_embedding {
    nn_module base;

    nn_parameter* weight;

    int vocabulary_size;
    int embedding_width;
};

struct nn_layer_norm {
    nn_module base;

    nn_parameter* weight;
    nn_parameter* bias;

    int normalized_width;
    float epsilon;
    int affine;
};


struct nn_mlp_config {
    int input_features;

    const int* hidden_sizes;
    size_t hidden_count;

    int output_features;
    const nn_activation* activations;

    int use_bias;

    nn_init_kind weight_init;
    nn_init_kind bias_init;
};


struct nn_mlp {
    nn_module base;

    nn_activation* activations;
    size_t layer_count;
};

struct nn_sgd {
    nn_module* module;
    float learning_rate;
};


nn_activation nn_activation_relu(void);
nn_activation nn_activation_gelu(void);
nn_activation nn_activation_sigmoid(void);
nn_activation nn_activation_tanh(void);

/* Create a caller-defined activation descriptor. */
nn_activation nn_activation_custom(
    const char* name,
    nn_activation_forward_fn forward,
    const void* context
);

ag_tensor* nn_softmax(const ag_tensor* logits);
ag_tensor* nn_log_softmax(const ag_tensor* logits);
ag_tensor* nn_cross_entropy(
    const ag_tensor* logits,
    const tensor* targets
);


/*
 * RNG API.
 * Invalid random requests return NAN.
 */
void nn_rng_seed(nn_rng* rng, uint64_t seed);
float nn_rng_uniform(nn_rng* rng, float min, float max);
float nn_rng_normal(nn_rng* rng, float mean, float stddev);


/*
 * Parameter API.
 *
 * The constructor allocates the tensor, initializes its storage, wraps it as
 * an Autograd leaf, and transfers ownership of the result to the caller.
 */
nn_parameter* nn_parameter_create(
    const char* name,
    int ndim,
    const int* dims,
    int trainable,
    nn_init_kind initializer,
    nn_rng* rng
);

void nn_parameter_destroy(nn_parameter* parameter);



int nn_module_register_parameter(
    nn_module* module,
    nn_parameter* parameter
);

int nn_module_register_child(
    nn_module* module,
    nn_module* child
);

size_t nn_module_parameter_count(const nn_module* module);
nn_parameter* nn_module_parameter_at(
    const nn_module* module,
    size_t index
);

/* Execute a module's forward callback and return one owned output reference. */
ag_tensor* nn_module_forward(
    const nn_module* module,
    const ag_tensor* input
);


/*
 * Linear-layer API.
 */
nn_linear* nn_linear_create(
    const char* name,
    int in_features,
    int out_features,
    int use_bias,
    nn_init_kind weight_init,
    nn_init_kind bias_init,
    nn_rng* rng
);

void nn_linear_destroy(nn_linear* layer);

ag_tensor* nn_linear_forward(
    const nn_linear* layer,
    const ag_tensor* input
);

/*
 * Embedding inputs are non-differentiable float tensors containing integral
 * token IDs. The generic nn_module_forward API is supported.
 */
nn_embedding* nn_embedding_create(
    const char* name,
    int vocabulary_size,
    int embedding_width,
    nn_init_kind weight_init,
    nn_rng* rng
);

void nn_embedding_destroy(nn_embedding* layer);

ag_tensor* nn_embedding_forward(
    const nn_embedding* layer,
    const ag_tensor* indices
);

/*
 * Normalize the final input dimension using biased variance. When affine is
 * enabled, the learned scale and bias have shape [normalized_width].
 */
nn_layer_norm* nn_layer_norm_create(
    const char* name,
    int normalized_width,
    float epsilon,
    int affine
);

void nn_layer_norm_destroy(nn_layer_norm* layer);

ag_tensor* nn_layer_norm_forward(
    const nn_layer_norm* layer,
    const ag_tensor* input
);


/*
 * MLP API.
 */
nn_mlp* nn_mlp_create(
    const char* name,
    const nn_mlp_config* config,
    nn_rng* rng
);

void nn_mlp_destroy(nn_mlp* model);

ag_tensor* nn_mlp_forward(
    const nn_mlp* model,
    const ag_tensor* input
);


nn_sgd* nn_sgd_create(nn_module* module, float learning_rate);
int nn_sgd_step(nn_sgd* optimizer);
void nn_sgd_zero_grad(nn_sgd* optimizer);
void nn_sgd_destroy(nn_sgd* optimizer);

#endif /* TENSORLIB_NN_H */
