#ifndef TENSORLIB_NN_H
#define TENSORLIB_NN_H

/*
 * Neural-network layer declarations for TensorLib.
 *
 * This header deliberately contains declarations and ownership contracts
 * only. Implementations belong in a future nn/ source directory.
 *
 * The neural-network layer stores the persistent model structure:
 *
 *     model -> modules -> parameters
 *
 * The dynamic Autograd graph is still created by calling ag_* operations
 * during each module forward pass. A module is not itself an Autograd node.
 */

#include <stddef.h>
#include <stdint.h>

#include "autograd.h"


/* Forward declarations. */
typedef struct nn_rng nn_rng;
typedef struct nn_parameter nn_parameter;
typedef struct nn_activation nn_activation;
typedef struct nn_module nn_module;
typedef struct nn_linear nn_linear;
typedef struct nn_mlp nn_mlp;
typedef struct nn_mlp_config nn_mlp_config;


/*
 * Deterministic random-number generator state.
 *
 * The algorithm remains an implementation detail. The seed is public so
 * callers can reproduce parameter initialization and training experiments.
 */
struct nn_rng {
    uint64_t state;
};


/* Weight and bias initialization policies. */
typedef enum {
    NN_INIT_ZERO,
    NN_INIT_XAVIER_UNIFORM,
    NN_INIT_XAVIER_NORMAL,
    NN_INIT_HE_UNIFORM,
    NN_INIT_HE_NORMAL
} nn_init_kind;


/*
 * A persistent trainable tensor.
 *
 * Ownership:
 * - The parameter owns value.
 * - value is an Autograd leaf with requires_grad == 1 when trainable.
 * - value->grad is allocated and accumulated by Autograd during backward.
 * - name is owned by the parameter and is used for diagnostics and state
 *   serialization.
 *
 * A parameter is not an operation node and must have creator == NULL.
 */
struct nn_parameter {
    char* name;
    ag_tensor* value;
    int trainable;
};


/*
 * Activation callback.
 *
 * input is borrowed. The callback returns one owned ag_tensor reference.
 * The activation descriptor is passed to the callback so future
 * parameterized activations can use activation->context.
 */
typedef ag_tensor* (*nn_activation_forward_fn)(
    const nn_activation* activation,
    const ag_tensor* input
);


/*
 * Activation descriptor.
 *
 * Built-in activations are stateless and use context == NULL. A custom or
 * parameterized activation may store borrowed configuration in context; the
 * descriptor does not own that context.
 */
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


/*
 * Common base structure embedded as the first field of every module type.
 *
 * Ownership:
 * - A module owns its registered parameters.
 * - A module owns its registered child modules.
 * - The parameter and child arrays are implementation-managed dynamic
 *   arrays.
 * - type_name is static or borrowed; name is owned by the module.
 */
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


/*
 * Fully connected layer:
 *
 *     output = input @ transpose(weight) + bias
 *
 * PyTorch-style parameter layout:
 *     weight: [out_features, in_features]
 *     bias:   [out_features]
 *
 * weight and bias are convenient aliases to parameters registered in base;
 * base remains the owner of the registered parameter objects.
 */
struct nn_linear {
    nn_module base;

    nn_parameter* weight;
    nn_parameter* bias;

    int in_features;
    int out_features;
    int use_bias;
};


/*
 * MLP construction settings.
 *
 * For hidden_sizes = {100} and hidden_count = 1, this creates:
 *
 *     Linear(input_features, 100)
 *     Linear(100, output_features)
 *
 * activations contains one descriptor per Linear layer, so its length must
 * be hidden_count + 1. An activation with forward == NULL means identity.
 * The config and its arrays are borrowed only during construction.
 */
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


/*
 * Generic multi-layer perceptron.
 *
 * Linear layers are owned by base.children. activations[i] is applied after
 * the i-th Linear layer. layer_count equals base.child_count.
 */
struct nn_mlp {
    nn_module base;

    nn_activation* activations;
    size_t layer_count;
};


/*
 * Built-in activation descriptors.
 *
 * These return small descriptors by value. The returned descriptors do not
 * own resources and are safe to store inside nn_mlp::activations.
 */
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


/*
 * RNG API — declarations only; implementation is future work.
 */
void nn_rng_seed(nn_rng* rng, uint64_t seed);
float nn_rng_uniform(nn_rng* rng, float min, float max);
float nn_rng_normal(nn_rng* rng, float mean, float stddev);


/*
 * Parameter API — declarations only; implementation is future work.
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


/*
 * Module registration API — declarations only; implementation is future
 * work.
 *
 * Registration transfers ownership of parameter or child to module on
 * success. A failed registration leaves ownership with the caller.
 */
int nn_module_register_parameter(
    nn_module* module,
    nn_parameter* parameter
);

int nn_module_register_child(
    nn_module* module,
    nn_module* child
);

/*
 * Recursively count and access parameters for optimizer/model traversal.
 * Traversal visits a module's direct parameters first, then its children
 * depth-first in registration order.
 */
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
 * Linear-layer API — declarations only; implementation is future work.
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
 * MLP API — declarations only; implementation is future work.
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

#endif /* TENSORLIB_NN_H */
