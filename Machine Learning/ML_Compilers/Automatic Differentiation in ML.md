---
aliases:
  - Autodiff
  - Algorithmic Differentiation ML
---

# Automatic Differentiation in Machine Learning

**Automatic Differentiation (AD)**, also known as *autodiff* or *algorithmic differentiation*, is a computational technique for efficiently and accurately calculating derivatives of functions implemented as computer programs. It is a cornerstone technology in modern machine learning systems, enabling gradient-based optimization of complex models with billions of parameters.

## Foundational Principles

### Core Concept

AD is based on a key insight: any function computation can be decomposed into a sequence of elementary operations for which derivatives are known. By systematically applying the chain rule to these operations, AD can compute exact derivatives of arbitrary computational graphs without the limitations of numerical approximation or symbolic differentiation.

### Position in ML Systems Stack

AD systems sit at a critical junction in the ML compiler stack:
- **Above**: High-level ML frameworks (PyTorch, TensorFlow)
- **Below**: Hardware-specific optimizers and code generators
- **Role**: Transforming computational graphs to include derivative computation

## Two Primary AD Algorithms

### [[Forward Mode Automatic Differentiation]]

- **Mechanism**: Propagates derivatives forward alongside function evaluation
- **Mathematical Formulation**: Computes directional derivatives $D_v f(x) = \nabla f(x) \cdot v$
- **Implementation**: Often uses [[Dual Numbers and Forward AD|dual numbers]] $(a + b\varepsilon)$, where $\varepsilon^2 = 0$
- **Memory Efficiency**: Requires constant memory overhead
- **Computational Cost**: $O(n)$ for computing full gradient of $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$
- **Use Case**: Efficient when $n \ll m$ (few inputs, many outputs)
- **Process**: One pass through the [[Forward Primal Trace]]

### [[Reverse Mode Automatic Differentiation]]

- **Mechanism**: Forward pass to build the computational graph, backward pass to compute derivatives
- **Mathematical Formulation**: Computes vector-Jacobian products $v^T J_f$
- **Implementation**: Uses the [[Computation Graphs for AD|computation graph]] and adjoint accumulation
- **Memory Requirements**: Needs to store intermediate values and graph structure
- **Computational Cost**: $O(m)$ for computing full gradient of $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$
- **Use Case**: Efficient when $m \ll n$ (many inputs, few outputs) - the typical case in ML
- **Process**: [[Forward Primal Trace]] followed by [[Reverse Adjoint Trace]]

## AD Implementation Paradigms

### Static vs. Dynamic Computation Graphs

The [[Computation Graphs for AD|computation graph]] representation significantly impacts AD implementation:

#### Static Graphs (Define-then-Run)
- **Examples**: TensorFlow 1.x, JAX with `@jit`
- **Advantages**: 
  - Enables aggressive optimization by compiler
  - Better performance through [[Just In Time Compilation]]
  - Easier deployment across hardware targets
- **Disadvantages**: 
  - Less intuitive for complex control flow
  - Harder to debug
  - Less flexible for dynamic models

#### Dynamic Graphs (Define-by-Run)
- **Examples**: PyTorch, TensorFlow Eager
- **Advantages**:
  - Natural Python control flow
  - Easier debugging
  - More intuitive development
- **Disadvantages**:
  - Fewer optimization opportunities
  - Potential performance tradeoffs
  - Harder to optimize for specialized hardware

### Implementation Methods

1. **Operator Overloading**:
   - Redefines arithmetic operators to also track derivative information
   - Popular in Python-based frameworks
   - Easier to implement but may have runtime overhead

2. **Source Code Transformation**:
   - Analyzes source code to generate derivative code
   - Can produce optimized, specialized derivative implementations
   - Used in some AD libraries and compilers

## AD in Modern ML Compiler Systems

### JAX's Approach to AD

[[JAX]] represents a modern synthesis of AD with compiler technology:

```python
import jax
import jax.numpy as jnp

def loss_fn(params, data, targets):
    predictions = model(params, data)
    return jnp.mean((predictions - targets) ** 2)

# AD transformation
grad_fn = jax.grad(loss_fn)  # Automatically differentiates loss_fn

# Compilation transformation
compiled_grad_fn = jax.jit(grad_fn)  # Compiles the gradient computation

# Vectorization transformation
batch_grad_fn = jax.vmap(compiled_grad_fn, in_axes=(None, 0, 0))  # Vectorizes across batch
```

JAX's design demonstrates how AD is now deeply integrated with other compiler optimizations:
- **Composable transformations**: AD, JIT, vectorization, parallelization
- **Pipeline integration**: AD happens early in compilation pipeline
- **XLA backend**: Optimizes AD patterns for hardware acceleration

### Compiler-Level Optimizations for AD

1. **Operation Fusion**: Combining multiple operations to reduce memory traffic
2. **Memory Layout Optimization**: Ensuring derivative buffers have optimal memory access patterns
3. **Kernel Selection**: Choosing specialized implementations for gradient operations
4. **Checkpointing**: Strategically recomputing forward-pass values to save memory
5. **Parallelism Exploitation**: Identifying independent AD computations

## Advanced AD Techniques in ML Compilers

### Vectorized and Parallelized AD

- **Vectorization** ([[Vectorization and Parallelization|SIMD vectorization]]): Computing gradients for multiple inputs simultaneously
- **Parallelization**: Distributing AD computation across devices
- **Implementation**: Frameworks like JAX expose this via `vmap` and `pmap` transformations

### Higher-Order Derivatives

Computing derivatives of derivatives (Hessians, etc.) through nested AD:

```python
# Computing a Hessian-vector product in JAX
def f(x):
    return jnp.sum(jnp.sin(x))

# First-order gradient
grad_f = jax.grad(f)

# Hessian-vector product function
def hvp(x, v):
    return jax.jvp(grad_f, [x], [v])[1]
```

### AD for Special Computational Patterns

1. **Implicit Differentiation**: Differentiating through fixed-point iterations and solutions to optimization problems
2. **Custom Gradient Definitions**: Overriding automatic gradients with hand-crafted ones
3. **Checkpointing**: Trading computation for memory in deep networks
4. **Mixed-Precision AD**: Using different numerical precision for forward and backward passes

## Performance and Optimization Considerations

### Memory-Computation Tradeoffs

- **Recomputation vs. Storage**: When to recompute vs. store intermediate values
- **Checkpointing Strategies**: Selective storage of computational checkpoints
- **Memory Access Patterns**: Optimizing for cache locality in gradient computation

### Precision Considerations

- **Numerical Stability**: Preventing gradient overflow/underflow
- **Mixed Precision**: Using lower precision for some operations to improve speed
- **Gradient Clipping**: Preventing exploding gradients

## AD in Different ML Domains

### AD for Recurrent Neural Networks

- **Backpropagation Through Time**: Unrolling RNNs for gradient computation
- **Truncated Backpropagation**: Limiting unrolling to manage computational cost
- **Memory-Efficient Gradient**: Specialized algorithms for long sequences

### AD for Convolutional Networks

- **Specialized Convolution Gradients**: Optimized implementations for both forward and backward passes
- **Memory Layout Optimization**: Transposition and layout optimizations for convolutional gradients

### AD for Graph Neural Networks

- **Message Passing Gradients**: Differentiating through message passing operations
- **Sparse Matrix Operations**: Specialized AD for sparse operations

## Conclusion

Automatic differentiation sits at the heart of modern machine learning systems and compiler stacks. Its efficient implementation enables the training of increasingly complex models while its integration with compilation techniques allows these models to run efficiently across diverse hardware platforms. Understanding AD—both its mathematical foundations and implementation details—is essential for anyone working on ML frameworks, compilers, or pushing the boundaries of model architecture design.

---

**References**:
1. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, *18*, 1-43. [Link](https://www.jmlr.org/papers/v18/17-468.html)
2. Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs. [GitHub](https://github.com/google/jax)
3. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.
4. Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.
5. Moses, W. S., & Churavy, V. (2020). Instead of rewriting foreign code for machine learning, automatically synthesize fast gradients. *NeurIPS 2020*. 