# MNIST MLP example

The example reads the standard uncompressed MNIST IDX files, flattens each
28x28 image to 784 normalized floats, and splits the loaded dataset in file
order into 75% training examples and 25% test examples.

The network is `784 input -> 128 ReLU -> 10 Softmax`. It uses TensorLib's
existing `nn_cross_entropy` implementation directly on logits (the stable,
non-redundant equivalent of softmax followed by multiclass cross entropy) and
trains with SGD. Accuracy is computed from the logits because applying softmax
cannot change the predicted class.

Both files must be in this directory:

- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

From the repository root:

```sh
make mnist
```

Or with CMake:

```sh
cmake -S . -B build
cmake --build build --target mnist_mlp
./build/mnist_mlp
```

Paths, epoch count, batch size, and output weights path can be overridden:

```sh
./bin/mnist_mlp IMAGE_FILE LABEL_FILE 10 64 WEIGHTS_FILE
```

The default output is `examples/mnsit/mnist_mlp.weights`. The binary file
starts with the `TLWEIGHT` magic bytes and stores versioned, named parameter
shapes followed by contiguous 32-bit floating-point values.
