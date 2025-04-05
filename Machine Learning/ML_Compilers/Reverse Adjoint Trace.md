---
aliases:
  - Backward Pass
  - Adjoint Trace
  - Gradient Propagation
---

# Reverse Adjoint Trace in AD

The **Reverse Adjoint Trace**, often simply called the **backward pass**, is the crucial second phase of [[Reverse Mode Automatic Differentiation]]. It follows the [[Forward Primal Trace]] and is responsible for efficiently computing the derivatives (gradients) of the final output(s) with respect to all preceding variables in the [[Computation Graphs for AD|computation graph]], working backward from output to input.

## Purpose

The goal of the reverse adjoint trace is to compute the *adjoint* $\bar{v} = \frac{\partial L}{\partial v}$ for each variable $v$ in the computation graph, where $L$ is the final output (typically the scalar loss function in ML). The adjoint represents the sensitivity of the final output $L$ to changes in the variable $v$.

Ultimately, it calculates the gradient $\nabla_{\theta} L$ needed for optimizing model parameters $\theta$.

## Mechanism: Applying the Chain Rule Backward

1.  **Initialization**: The trace starts at the final output node $L$. Its adjoint is initialized to $\bar{L} = \frac{\partial L}{\partial L} = 1$.
2.  **Backward Propagation**: The algorithm traverses the computation graph (or the recorded tape) in reverse topological order.
3.  **Chain Rule Application**: For each node $u$ that was an input to a node $v$ during the forward pass (i.e., $v = f(..., u, ...)$), the adjoint $\bar{u}$ is updated by propagating the adjoint $\bar{v}$ backward through the operation $f$. Specifically, the contribution to $\bar{u}$ from the path through $v$ is calculated as:
    $$ \Delta \bar{u} = \bar{v} \cdot \frac{\partial v}{\partial u} $$
    where $\frac{\partial v}{\partial u}$ is the local partial derivative of the elementary operation $v = f(..., u, ...)$ with respect to $u$.
4.  **Accumulation**: If a variable $u$ contributes to multiple subsequent operations in the forward pass, its total adjoint $\bar{u}$ is the *sum* of the adjoint contributions propagated back from all its "child" nodes in the computation graph. This summation directly implements the multivariate chain rule.
5.  **Final Gradients**: Once the backward pass reaches the original input nodes (e.g., model parameters $\theta$), their accumulated adjoints $\bar{\theta}$ represent the desired gradient $\nabla_{\theta} L$.

## Relationship to Vector-Jacobian Products (VJPs)

Mathematically, the reverse adjoint trace computes Vector-Jacobian Products (VJPs). For an elementary operation $y = f(x)$, the backward step computes:
$$ \bar{x}^T = \bar{y}^T J_f $$
where $\bar{x}^T$ and $\bar{y}^T$ are row vectors of adjoints, and $J_f$ is the Jacobian of the elementary function $f$. This is computationally efficient when the output dimension (dimension of $\bar{y}$) is small.

## Implementation Details

-   Requires the computation graph structure or tape created during the [[Forward Primal Trace]].
-   Requires the values of certain intermediate variables from the forward pass to compute the local partial derivatives $\frac{\partial v}{\partial u}$.
-   Efficient implementations often involve careful memory management and potentially graph optimizations.

## Connection to Backpropagation

In the context of neural networks, **backpropagation** *is* the application of reverse mode AD (specifically, the reverse adjoint trace) to compute the gradient of the loss function with respect to the network's weights and biases.

-   The [[Forward Primal Trace]] corresponds to the forward pass through the network, calculating activations.
-   The [[Reverse Adjoint Trace]] corresponds to the backward pass, propagating the error signal (adjoints/derivatives of the loss) back through the layers.

## Conclusion

The reverse adjoint trace is the core computational step in [[Reverse Mode Automatic Differentiation]] and backpropagation. By systematically applying the chain rule backward through the computation recorded in the [[Forward Primal Trace]], it allows for the highly efficient computation of gradients essential for training modern machine learning models. 