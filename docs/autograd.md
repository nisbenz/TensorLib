# TensorLib autograd contract

TensorLib uses eager reverse-mode automatic differentiation. Each `ag_tensor`
owns a raw tensor value, an optional accumulated gradient, and (for non-leaf
results) the node that produced it.

## Ownership and graph lifetime

- `ag_from_owned_tensor` always consumes the supplied raw tensor, including on
  failure.
- An `ag_tensor` starts with one reference. Graph nodes retain their inputs, so
  callers may release leaves and intermediates after constructing later
  results.
- The output owns its creator node. The node's output pointer is non-owning,
  which prevents a reference cycle.
- Backward does not destroy the graph. The graph remains reusable until its
  reference-counted outputs are released.

## Gradient execution

- `ag_backward` accepts only a scalar output and seeds it with a gradient of
  one.
- `ag_backward_with_grad` accepts a same-shaped explicit upstream gradient.
- A backward pass computes contributions transactionally. Existing persistent
  gradients are updated only after every local backward rule succeeds.
- Repeated successful backward calls accumulate into `ag_tensor.grad`.
- Contributions produced at broadcast shapes are reduced to each input's
  original shape before accumulation.
- `ag_zero_grad` clears one gradient. `ag_zero_grad_all` clears every gradient
  reachable from a graph root.

## Views, detach, and mutation

Tensor views share storage. A detached value will share the exact shape,
strides, offset, storage, and storage version of its source, but it will have no
creator, gradient, or gradient requirement.

Library-controlled writes to an existing tensor value must mark its shared
storage as modified. Backward validates the versions captured during forward
before computing any contribution. A stale graph fails transactionally and
leaves existing gradients unchanged.

Direct writes through `storage->data` are supported for initialization before a
value participates in a graph. Direct writes after graph capture that are not
followed by the public mutation marker are outside the API contract.

## Derivative conventions

- ReLU uses derivative zero at zero and propagates NaN for a NaN input.
- `pow(x, 0)` has derivative zero and `pow(x, 1)` passes through the upstream
  gradient. Other domains follow the platform `powf` behavior.
- Square-root backward divides by twice the forward output. Zero and negative
  inputs therefore follow IEEE-754 infinity and NaN behavior.
- Sigmoid and tanh backward use their saved forward outputs.
- GELU uses the derivative of TensorLib's tanh approximation, not the exact
  error-function GELU.
- Max reduction divides the upstream gradient equally among exact ties and
  propagates NaN gradients for a NaN reduction result.

