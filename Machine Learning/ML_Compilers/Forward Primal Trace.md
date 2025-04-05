---
aliases:
  - Forward Primal Trace
  - Forward Pass Values
  - Primal Computation
---

# Forward Primal Trace in AD

The **Forward Primal Trace** refers to the process of evaluating a function $f(x)$ from inputs to outputs and recording the sequence of elementary operations and their corresponding intermediate values. This trace is a fundamental component of many [[Automatic Differentiation in ML|Automatic Differentiation (AD)]] systems, particularly [[Reverse Mode Automatic Differentiation]].

## Purpose

The primary purpose of the forward primal trace in the context of [[Reverse Mode Automatic Differentiation|reverse mode AD]] is to:

1.  **Build the [[Computation Graphs for AD|Computation Graph]]**: It defines the structure (nodes and edges) representing the function's calculation.
2.  **Store Intermediate Values**: Many intermediate values calculated during the forward pass are needed to compute the local partial derivatives during the subsequent backward pass (e.g., if $y=x^2$, $\frac{dy}{dx}=2x$, so the value of $x$ from the forward pass is needed).
3.  **Provide Context for the Backward Pass**: The backward pass ([[Reverse Adjoint Trace]]) traverses this recorded trace in reverse order.

## Implementation Details

- **Tape**: In tape-based AD systems, the forward primal trace generates a "tape" or Wengert list, which is a linear sequence of instructions recording each elementary operation performed.
- **Graph**: In graph-based AD systems, the forward pass explicitly constructs the graph data structure.
- **Static vs. Dynamic**: 
    - In systems with *static* computation graphs (like [[JAX]] when using `@jit` or TensorFlow 1.x), the graph structure is determined and potentially optimized *before* the forward trace executes. The trace then primarily involves computing and storing necessary intermediate values within that fixed structure.
    - In systems with *dynamic* computation graphs (like PyTorch or TensorFlow Eager mode), the graph structure *is* the trace, built step-by-step as the forward computation proceeds.

## Relationship to Forward Mode AD

[[Forward Mode Automatic Differentiation]] can be seen as augmenting the forward primal trace. Instead of just calculating and recording the *value* of each intermediate variable (the primal value), it simultaneously calculates and propagates the *derivative* (the tangent or dual value) alongside the primal value at each step. It doesn't typically require storing the entire trace structure for later use in the same way reverse mode does.

## Memory Considerations

Storing the forward primal trace (specifically, the intermediate values needed for the backward pass) is the main reason [[Reverse Mode Automatic Differentiation]] has higher memory requirements than [[Forward Mode Automatic Differentiation]]. Techniques like *checkpointing* (recomputing parts of the forward trace during the backward pass instead of storing everything) are used to manage this memory consumption for very large models.

## Conclusion

The forward primal trace is the essential first step in [[Reverse Mode Automatic Differentiation]], establishing the computational path and storing necessary intermediate results. It lays the groundwork for the efficient gradient calculation performed during the subsequent [[Reverse Adjoint Trace|backward pass]]. 