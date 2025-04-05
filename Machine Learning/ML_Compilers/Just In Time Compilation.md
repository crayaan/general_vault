---
aliases:
  - JIT Compilation
  - Dynamic Compilation ML
---

# Just In Time Compilation in ML

**Just-In-Time (JIT) Compilation** is a technique used in programming languages and frameworks where code (often in an intermediate representation or bytecode) is compiled into machine code *during runtime*, shortly before it is executed, rather than entirely ahead-of-time (AOT).

In the context of Machine Learning frameworks like [[JAX]], PyTorch (TorchScript), and TensorFlow (`tf.function`), JIT compilation plays a crucial role in bridging the gap between high-level, flexible Python APIs and high-performance execution on hardware accelerators (GPUs, TPUs).

## Motivation in ML Frameworks

- **Performance**: Python, while excellent for rapid prototyping and research, can be slow for computationally intensive numerical tasks common in ML. Executing operations individually via Python's interpreter incurs significant overhead.
- **Optimization**: Compiling larger segments of computation allows for powerful optimizations (operation fusion, memory optimization, constant folding) that are difficult or impossible to perform when executing operations one by one.
- **Hardware Acceleration**: Native Python code cannot run directly on GPUs/TPUs. JIT compilation translates the high-level operations into optimized kernels executable on these accelerators (often via intermediate compilers like [[XLA]]).
- **Flexibility vs. Speed**: JIT offers a compromise between the flexibility of interpreted languages (like Python used in dynamic graph modes) and the performance of statically compiled languages.

## How JIT Works in ML (Conceptual)

1.  **Tracing/Graph Capture**: When a function marked for JIT compilation (e.g., with `@jax.jit`, `@tf.function`, `torch.jit.script` or `torch.jit.trace`) is called for the first time with specific input types/shapes, the framework *traces* its execution. This involves recording the sequence of framework-specific operations performed within the function, effectively building a [[Computation Graphs for AD|static computation graph]] representing that function for those input types.
2.  **Intermediate Representation (IR)**: The traced sequence of operations is often converted into a hardware-agnostic Intermediate Representation (e.g., XLA HLO, TorchScript IR).
3.  **Optimization**: The compiler performs various optimizations on this IR graph.
4.  **Code Generation**: The optimized IR is then compiled into low-level machine code specific to the target hardware (CPU, GPU, TPU).
5.  **Caching & Execution**: The compiled code is cached. Subsequent calls to the JIT-compiled function *with the same input types/shapes* can directly execute the cached, optimized machine code, bypassing the Python interpreter overhead and tracing step.

## JIT in Specific Frameworks

- **[[JAX]] (`@jax.jit`)**: JIT compilation via [[XLA]] is central to JAX's performance model. It compiles Python functions containing JAX operations into optimized XLA computations executable on CPU/GPU/TPU. Tracing occurs based on argument types and shapes.
- **TensorFlow (`@tf.function`)**: Decorator used to convert Python functions performing TensorFlow operations into callable TensorFlow graphs. This enables graph optimizations and improves performance over pure Eager execution.
- **PyTorch (`torch.jit`)**: Provides `torch.jit.script` (analyzes Python code) and `torch.jit.trace` (records operations based on example inputs) to convert PyTorch models/functions into TorchScript, an optimizable and serializable IR.

## Advantages

- **Significant Speedups**: Reduces Python overhead and enables powerful compiler optimizations.
- **Hardware Utilization**: Allows ML computations to run efficiently on accelerators.
- **Portability**: Compiled graphs/models can often be deployed more easily than raw Python code.

## Disadvantages/Considerations

- **Tracing Overhead**: The initial tracing and compilation step can take time, especially for complex functions.
- **Static Nature**: Once compiled for specific input types/shapes, the graph is fixed. If the function is called with different types/shapes, a *re-compilation* (another trace) might be triggered, incurring overhead. This makes handling highly dynamic control flow or variable shapes challenging within JIT-compiled code (though frameworks are improving support for [[Shape Polymorphism]]).
- **Debugging**: Debugging issues within JIT-compiled code can be more difficult than debugging standard Python.
- **Side Effects**: Python side effects (like printing, modifying external variables) may not behave as expected within JIT-compiled functions, as they might only run during the initial trace or be optimized away.

## Conclusion

JIT compilation is a vital technique in modern ML frameworks, enabling high performance for numerical computation written in flexible high-level languages like Python. By tracing Python functions and compiling them into optimized, hardware-specific code via intermediate representations like XLA, JIT compilers allow frameworks like JAX, TensorFlow, and PyTorch to achieve performance close to that of lower-level languages while retaining much of Python's ease of use. 