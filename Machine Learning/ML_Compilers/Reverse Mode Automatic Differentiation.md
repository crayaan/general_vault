---
aliases:
  - Reverse Mode AD
  - Backpropagation
  - Adjoint Mode AD
---

# Reverse Mode Automatic Differentiation

Reverse Mode Automatic Differentiation (AD), also known as adjoint mode AD or reverse accumulation, is the most widely used AD technique in machine learning, forming the basis of the **backpropagation** algorithm used to train neural networks.

## Core Idea

Reverse mode AD computes the gradient of a scalar output (e.g., loss function $L$) with respect to all inputs (e.g., model parameters $\theta$) efficiently. It works in two phases:

1.  **Forward Pass ([[Forward Primal Trace]])**: The original function is evaluated from inputs to output(s), computing and storing the values of all intermediate variables in a [[Computation Graphs for AD|computation graph]]. This process is often called the *forward primal trace*.
2.  **Backward Pass ([[Reverse Adjoint Trace]])**: Starting from the final output (where the derivative is $\frac{\partial L}{\partial L} = 1$), the algorithm traverses the computation graph *backward*, applying the chain rule at each node to compute the derivatives (adjoints) of the output with respect to intermediate variables and ultimately the inputs. This phase is often called the *reverse adjoint trace*.

## Mathematical Foundation

Reverse mode computes the **vector-Jacobian product (VJP)**, $v^T J$, where $J$ is the Jacobian matrix of the function $f: \mathbb{R}^n \to \mathbb{R}^m$. For a scalar loss function $L = f(x)$, where $f: \mathbb{R}^n \to \mathbb{R}$ (so $m=1$), the Jacobian $J$ is a row vector $\nabla f(x)^T$. Setting $v=1$, the VJP becomes $1 \cdot \nabla f(x)^T = \nabla f(x)^T$, giving the gradient of the loss with respect to all inputs $x$.

The backward pass propagates *adjoints* (denoted $\bar{v} = \frac{\partial L}{\partial v}$) using the chain rule. If $z = g(y)$ and $y = h(x)$, then:
$$ \bar{x} = \frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial y} \frac{\partial y}{\partial x} = \bar{z} \frac{\partial z}{\partial y} \frac{\partial y}{\partial x} = \bar{y} \frac{\partial y}{\partial x} $$
At each node in the backward pass, the adjoint is calculated by summing the contributions from all paths leading *from* that node towards the final output.

## Implementation: Tapes and Graphs

Reverse mode requires storing the computation graph structure and potentially the values of intermediate variables from the forward pass. This is often done using a **tape** (or Wengert list) that records the sequence of operations.

- **Tape-Based AD**: Records each elementary operation during the forward pass. The backward pass replays this tape in reverse to compute derivatives.
- **Graph-Based AD**: Explicitly constructs the [[Computation Graphs for AD|computation graph]] during the forward pass. The backward pass traverses this graph.

## Computational Cost

- **Cost**: The computational cost of reverse mode AD to compute the gradient of a scalar output with respect to *all* $n$ inputs is roughly a small constant factor (typically 2-5) times the cost of evaluating the original function *once*. Cost($\nabla f$) $\approx O(1) \times$ Cost($f$).
- **Efficiency**: This makes reverse mode extremely efficient when the number of outputs is much smaller than the number of inputs ($m \ll n$), which is precisely the case in training neural networks ($m=1$ for the scalar loss, $n$ = millions/billions of parameters).

## Use Cases in Machine Learning

- **Backpropagation**: The standard algorithm for training deep neural networks is exactly reverse mode AD applied to the network's computation graph.
- **Gradient-Based Optimization**: Essential for algorithms like SGD, Adam, etc., used to minimize loss functions.
- **Sensitivity Analysis**: Calculating the sensitivity of a scalar output to all input parameters.

## Comparison with Forward Mode

| Feature                                             | [[Forward Mode Automatic Differentiation]] | Reverse Mode AD                                         |
| --------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| **Direction**                                       | Computes derivatives forward               | Computes derivatives backward                           |
| **Efficiency ($f: \mathbb{R}^n \to \mathbb{R}^m$)** | Best when $n \ll m$                        | Best when $m \ll n$                                     |
| **Computes**                                        | Jacobian-Vector Product ($Jv$)             | Vector-Jacobian Product ($v^T J$)                       |
| **Memory**                                          | Low, constant overhead                     | High, requires storing intermediate values (tape/graph) |
| **ML Training**                                     | Less common                                | Standard (Backpropagation)                              |


## Conclusion

Reverse mode AD is the powerhouse behind modern deep learning training. Its ability to efficiently compute the gradient of a scalar loss function with respect to a vast number of parameters makes large-scale optimization feasible. While it has higher memory requirements than forward mode due to the need to store the computation graph/tape, its computational efficiency for the typical ML scenario ($m=1$) is unparalleled.

---

**References**:
1. Baydin, A. G., et al. (2018). Automatic differentiation in machine learning: a survey. *JMLR*, *18*, 1-43.
2. Griewank, A., & Walther, A. (2008). *Evaluating Derivatives*. SIAM.
3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning internal representations by error propagation. *Nature*, *323*(6088), 533-536. 