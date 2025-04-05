# ML Compilers and Frameworks Notes

Index for notes related to ML Compilers, Frameworks, and underlying technologies.

## Core Frameworks

- [[JAX]] - High-performance numerical computing and ML research library.
- [[JAX vs Others]] - Comparison of JAX with PyTorch and TensorFlow.

## Automatic Differentiation (AD)

- [[Automatic Differentiation in ML]] - Overview of AD concepts in machine learning.
- [[Forward Mode Automatic Differentiation]] - Tangent mode AD, dual numbers, JVPs.
- [[Reverse Mode Automatic Differentiation]] - Adjoint mode AD, backpropagation, VJPs.
- [[Computation Graphs for AD]] - Role of graphs, static vs. dynamic graphs.
- [[Forward Primal Trace]] - The forward value computation pass.
- [[Reverse Adjoint Trace]] - The backward gradient computation pass.
- [[Dual Numbers and Forward AD]] - Deeper dive into dual numbers and their connection to forward AD.

## Compilation and Optimization

- [[Just In Time Compilation]] - JIT compilation in ML frameworks (JAX, TF, PyTorch).
- [[XLA]] - Accelerated Linear Algebra compiler (used by JAX, TF).
- [[Vectorization and Parallelization]] - SIMD, SPMD, Data/Model Parallelism, vmap, pmap.
- [[Shape Polymorphism]] - Handling dynamic shapes in compiled code.

## Sparsity and Pruning

- [[Sparsity and Pruning Overview]] - Introduction to model sparsity and pruning.
- [[Sparse Tensor Formats]] - Data structures like COO, CSR, CSC, DOK, LIL.
- [[Network Pruning Techniques]] - Granularity, criteria, and schedules (pruning strategies).
- [[Quantization and Pruning]] - Combining sparsity with reduced precision techniques.

## Advanced Topics

- [[Pallas]] - JAX's kernel language for GPU/TPU.
- [[MLIR]] - Multi-Level Intermediate Representation.

---

*This index provides links to detailed notes covering the concepts and technologies behind modern machine learning compilers and frameworks, with a particular focus on Automatic Differentiation and performance optimization techniques.* 