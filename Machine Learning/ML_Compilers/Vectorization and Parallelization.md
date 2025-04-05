---
aliases:
  - Vectorization
  - Parallelism ML
  - SIMD ML
  - SPMD ML
---

# Vectorization and Parallelization in ML

Vectorization and parallelization are crucial techniques for achieving high performance in machine learning computations, especially on modern hardware like GPUs and TPUs which are designed for parallel data processing.

## Vectorization (SIMD)

Vectorization refers to performing the same operation on multiple data points simultaneously. This often maps to **Single Instruction, Multiple Data (SIMD)** instructions available on modern CPUs and is the fundamental operational mode for GPUs/TPUs.

- **Concept**: Instead of looping through elements of an array/vector one by one, a vectorized operation processes entire vectors or chunks of data in a single instruction.
    - *Scalar*: `c = a + b` (one operation)
    - *Loop*: `for i in range(N): c[i] = a[i] + b[i]` (N operations, sequential)
    - *Vectorized*: `c = a + b` (where a, b, c are vectors; one operation conceptually, executed in parallel on hardware)
- **Benefits**: Massively reduces computational overhead, leverages parallel hardware capabilities.
- **ML Frameworks**: Libraries like NumPy, TensorFlow, PyTorch, and [[JAX]] heavily rely on vectorized operations implemented via optimized low-level libraries (BLAS, cuBLAS, etc.). Writing code using high-level array operations instead of explicit Python loops is key to performance.

### `vmap` in JAX

[[JAX]] provides a powerful `jax.vmap` transformation that automatically vectorizes a function written for single data points to operate efficiently on batches of data.

```python
import jax
import jax.numpy as jnp

def predict(params, image): # Operates on a single image
  # ... complex computation ...
  return output

# Automatically create a function that operates on a batch of images
batch_predict = jax.vmap(predict, in_axes=(None, 0), out_axes=0)
# 'params' are not batched (None)
# 'images' are batched along axis 0
# 'outputs' will be batched along axis 0

# Run on a batch
batch_outputs = batch_predict(params, batch_of_images)
```
`vmap` allows writing clean code for single examples while achieving high performance on batches through automatic vectorization.

## Parallelization

Parallelization involves distributing computations across multiple processing units (CPU cores, GPUs, TPUs) or even multiple machines.

### Data Parallelism

- **Concept**: The most common form in deep learning. Replicate the model on multiple devices. Split a large batch of training data across the devices. Each device computes gradients for its mini-batch in parallel. Gradients are then aggregated (e.g., averaged) to update the model parameters (which are kept synchronized across devices).
- **Implementation**: Frameworks provide abstractions like PyTorch's `DistributedDataParallel` (DDP) or TensorFlow's `MirroredStrategy`.
- **[[JAX]] (`pmap`)**: JAX uses `jax.pmap` for SPMD-style (Single Program, Multiple Data) data parallelism. It compiles the function once and runs the *same program* on multiple devices (e.g., multiple GPUs or TPU cores), each operating on its shard of the data. Collective communication operations (like `psum` for summing gradients) are used to synchronize across devices.

### Model Parallelism

- **Concept**: Used when a model is too large to fit on a single device. The model itself is split across multiple devices. Different parts of the model execute on different devices, with activations/gradients communicated between them.
- **Types**: Tensor parallelism (splitting individual large layers), pipeline parallelism (assigning different layers to different devices and pipelining batches through them).
- **Implementation**: More complex to implement than data parallelism. Frameworks offer varying levels of support (e.g., Megatron-LM, DeepSpeed, JAX with `pjit`/`sharding`).
- **[[JAX]] (`pjit`/sharding)**: `jax.pjit` combined with `jax.sharding` allows for sophisticated model parallelism by specifying how large arrays (parameters, activations) should be sharded (split) across a mesh of devices.

### SPMD (Single Program, Multiple Data)

- **Concept**: A parallel programming paradigm where the *same* program code is executed simultaneously on multiple processors, but each processor operates on a different subset of the data. Communication between processors happens via explicit collective operations (e.g., all-reduce, all-gather, broadcast).
- **Relevance**: This is the primary model used by JAX (`pmap`, `pjit`) for scaling computations across multiple TPU cores or GPUs. It provides a clean way to express complex parallel computations.

## Relationship to AD and Compilation

- **[[Automatic Differentiation in ML|AD]]**: Parallelism interacts with AD. For data parallelism, gradients are computed independently on each device using [[Reverse Mode Automatic Differentiation|reverse mode AD (backpropagation)]] and then aggregated.
- **[[Just In Time Compilation|JIT Compilation]]**: Compilers (like XLA used by JAX and TensorFlow) are crucial for generating efficient parallel code. They can automatically fuse operations, optimize memory layouts, and generate kernels suitable for SIMD execution and parallel hardware. JIT compilation within `pmap` or `pjit` in JAX allows the compiler to optimize the *entire* parallel computation, including communication.

## Conclusion

Vectorization and parallelization are essential for training large machine learning models efficiently. Vectorization leverages SIMD capabilities within devices, while parallelization distributes work across multiple devices or machines. Frameworks like JAX offer powerful abstractions (`vmap`, `pmap`, `pjit`) that integrate seamlessly with automatic differentiation and JIT compilation to simplify the writing of high-performance, parallel ML code. 