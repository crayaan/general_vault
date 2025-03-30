---
aliases:
  - Just-Another-XLA-library
  - JAX ML Framework
---

# JAX: High-Performance Machine Learning Framework

## Overview

JAX is a high-performance numerical computing library developed by Google Research that combines NumPy's familiar API with the power of GPU/TPU acceleration and automatic differentiation. JAX's core philosophy is expressed through its primary transformations: differentiation (`grad`), vectorization (`vmap`), just-in-time compilation (`jit`), and parallelization.

## Core Features

### 1. Functional Programming Paradigm

JAX embraces functional programming principles, emphasizing pure functions and immutable data structures. This design choice enables JAX's powerful program transformations:

- **Pure functions**: Same inputs always produce the same outputs
- **No side effects**: Functions don't modify external state
- **Immutable arrays**: JAX arrays are immutable, unlike NumPy arrays

### 2. Key Transformations

```python
import jax
import jax.numpy as jnp

# Just-in-time compilation
@jax.jit
def fast_function(x):
    return jnp.sin(x) + jnp.cos(x)

# Automatic differentiation
def f(x):
    return jnp.sum(jnp.sin(x) ** 2)

grad_f = jax.grad(f)  # Compute df/dx

# Vectorization
def per_example_function(x):
    return jnp.sin(x) + jnp.cos(x)

vectorized_function = jax.vmap(per_example_function)  # Apply to batches
```

### 3. XLA Compilation

JAX uses XLA (Accelerated Linear Algebra) compiler for hardware acceleration:

- **Transparent compilation**: Seamless integration with NumPy-like code
- **Hardware portability**: CPU, GPU, and TPU support
- **Optimized performance**: Fusion of operations, memory usage optimization
- **SPMD programming**: Single Program Multiple Data support

## Architecture

JAX's architecture consists of multiple layers:

1. **User-facing API layer**: NumPy-compatible API (`jax.numpy`)
2. **Transformation layer**: Core transformations (`grad`, `jit`, `vmap`)
3. **JAX Intermediate Representation (jaxpr)**: Internal representation of computations
4. **XLA Integration layer**: Translation to XLA HLO (High Level Optimizer)
5. **Backend execution**: Hardware-specific implementation (CPU, GPU, TPU)

## Ecosystem

### ML Libraries Built on JAX

JAX serves as a foundation for several higher-level libraries:

- **Flax**: Neural network library (similar to Keras)
- **Optax**: Optimization library with composable gradient transformations
- **Haiku**: Neural network library from DeepMind (similar to PyTorch)
- **Equinox**: PyTorch-like neural network library
- **BlackJAX**: Probabilistic programming and sampling algorithms
- **Oryx**: Probabilistic programming with program transformations

### Example: Neural Network with Flax

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x

# Initialize model
model = MLP(hidden_size=128, output_size=10)
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))

# Setup optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(variables)

# Training step function
@jax.jit
def train_step(variables, opt_state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['images'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['labels']).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables)
    updates, opt_state = optimizer.update(grads, opt_state)
    variables = optax.apply_updates(variables, updates)
    return variables, opt_state, loss
```

## Advantages and Limitations

### Advantages

1. **Performance**: Nearly optimal GPU/TPU utilization
2. **Composable transformations**: Combine `grad`, `jit`, `vmap` in any order
3. **Functional design**: Enables sophisticated program transformations
4. **Automatic parallelism**: Efficient distribution across multiple devices
5. **Compilation cache**: Speeds up repeated operations

### Limitations

1. **Steeper learning curve**: Functional paradigm may be unfamiliar
2. **Debugging challenges**: JIT-compiled code harder to debug
3. **Ecosystem maturity**: Younger than PyTorch and TensorFlow
4. **In-place updates**: Require special care due to immutability

## Performance Comparison

Recent benchmarks show JAX performing competitively against other frameworks:

- **On NVIDIA L4 GPUs**: JAX often outperforms PyTorch and TensorFlow
- **On NVIDIA A100 GPUs**: JAX and PyTorch/XLA show similar performance (due to common XLA backend)
- **For distributed training**: JAX's SPMD programming model excels

The performance advantage varies by:
- Model architecture
- Hardware platform
- Specific optimizations used

## Common Usage Patterns

### 1. Stateful Computations

JAX handles state using functional carry patterns:

```python
def update_state(state, input_data):
    new_state = state + input_data
    output = jnp.sin(new_state)
    return new_state, output

# Initialize state
state = jnp.array(0.0)

# Loop over inputs
for x in inputs:
    state, output = update_state(state, x)
```

### 2. RNG Management

JAX uses splittable PRNGs to maintain reproducibility:

```python
import jax
from jax import random

# Create a key
key = random.PRNGKey(seed=42)

# Split for parallel operations
key, subkey = random.split(key)
x = random.normal(subkey, shape=(10,))

# Split for multiple random operations
key, subkey1, subkey2 = random.split(key, 3)
```

### 3. Handling Large Models

For large models that don't fit in a single device memory:

```python
# Using model parallelism with jax.pjit
from jax.experimental import pjit
from jax.sharding import Mesh, PartitionSpec as P

# Define the device mesh
devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 4), ('data', 'model'))

# Define the sharding strategy
in_spec = P('data', None)
out_spec = P('data', None)
param_spec = P(None, 'model')

# Define the function to execute
sharded_fn = pjit.pjit(
    forward_fn,
    in_shardings=in_spec,
    out_shardings=out_spec,
    static_argnums=(1,),
)
```

## Getting Started

### Installation

JAX installation is hardware-specific:

```bash
# CPU-only
pip install jax

# GPU support (CUDA 11+)
pip install jax[cuda11] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# TPU support
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Key Resources

- **Official Documentation**: [JAX Documentation](https://jax.readthedocs.io/)
- **GitHub Repository**: [google/jax](https://github.com/google/jax)
- **Tutorials**: [JAX Tutorials](https://docs.jax.dev/en/latest/tutorials.html)
- **Examples**: [JAX Examples](https://github.com/google/jax/tree/main/examples)

## Recent Developments (2024)

Recent additions to the JAX ecosystem include:

- **Pallas**: JAX's kernel language for writing custom kernels for GPU and TPU
- **Shape polymorphism**: Improved handling of dynamic shapes
- **Transfer guard**: Safety features for cross-device data transfers
- **JAX FFI**: Foreign function interface for integrating with external libraries

### Exploring Pallas: JAX's Kernel Language

Pallas, introduced in 2023-2024, represents a significant advancement for JAX, enabling developers to write custom kernels for both GPU and TPU hardware. This fills a critical gap that previously existed in TPU programming.

#### Key Features of Pallas

1. **Custom Kernel Creation**: Write hardware-specific optimized operations
2. **High-level Python Interface**: Uses familiar JAX/NumPy syntax while providing low-level control
3. **Cross-Platform Support**: Works on both GPUs (via Triton) and TPUs (via Mosaic)
4. **Integration with PyTorch/XLA**: Custom kernels can be used in PyTorch via the XLA bridge

#### Pallas Programming Model

Pallas introduces a novel programming model with concepts like:

```python
from functools import partial
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

# A simple vector addition kernel
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...] 
  o_ref[...] = x + y

# Using the kernel in a JAX program
@jax.jit
def add_vectors(x, y):
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
```

#### Advanced Use Case: Matrix Multiplication with Activation

Pallas enables operator fusion and customized implementations:

```python
def matmul_kernel(x_ref, y_ref, z_ref, *, activation):
  z_ref[...] = activation(x_ref[...] @ y_ref[...])

def matmul(x, y, *, activation):
  return pl.pallas_call(
    partial(matmul_kernel, activation=activation),
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(2, 2),
    in_specs=[
        pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
    ],
    out_specs=pl.BlockSpec(
        (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j)
    ),
  )(x, y)
```

#### Impact on ML Research and Production

Pallas has enabled significant advancements in areas like:

1. **Transformer Optimization**: Flash Attention implementations for TPU
2. **Custom Architecture Components**: Supporting novel operations not natively available
3. **Performance Optimization**: Achieving GPU-like programmability on TPUs

While Pallas is still experimental and rapidly evolving, it represents a major step toward making TPUs more accessible and competitive for custom ML workloads.

## Conclusion

JAX represents a significant advancement in ML frameworks, bringing together the simplicity of NumPy with the power of modern hardware accelerators. Its functional design enables powerful program transformations that would be difficult or impossible in traditional imperative frameworks. While it has a steeper learning curve than some alternatives, the performance benefits and composable nature make it increasingly popular for research and production ML systems.

---

JAX's design philosophy can be summarized as: "Compose transformations to efficiently express complex computations on accelerators with minimal code changes from NumPy." 