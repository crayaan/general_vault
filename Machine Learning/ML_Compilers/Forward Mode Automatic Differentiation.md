---
aliases:
  - Forward Mode AD
  - Tangent Mode AD
  - Forward Pass Differentiation
---

# Forward Mode Automatic Differentiation

Forward Mode Automatic Differentiation (AD), also known as tangent mode AD or forward accumulation, is one of the primary methods for computing derivatives automatically. It calculates the derivative alongside the function value by propagating derivative information *forward* through the [[Computation Graphs for AD|computation graph]].

## Core Idea

Forward mode computes the directional derivative $D_v f(x) = \nabla f(x) \cdot v$ for a chosen direction vector $v$. It essentially computes the effect of perturbing a single input (or a linear combination defined by $v$) on all intermediate variables and the final output.

## Implementation: Dual Numbers

Forward mode is elegantly implemented using **dual numbers**. A dual number is of the form $a + b\varepsilon$, where $a$ is the real part (function value) and $b$ is the dual part (derivative value), with the property $\varepsilon^2 = 0$.

1.  **Initialization**: To compute the derivative of $f(x_1, ..., x_n)$ with respect to $x_i$, initialize the inputs as dual numbers:
    -   $x_j = x_j + 0\varepsilon$ for $j \neq i$
    -   $x_i = x_i + 1\varepsilon$ (The '1' in the dual part signifies we are differentiating w.r.t. $x_i$)
2.  **Propagation**: Perform all arithmetic operations using dual number arithmetic:
    -   $(a + b\varepsilon) + (c + d\varepsilon) = (a+c) + (b+d)\varepsilon$
    -   $(a + b\varepsilon) \times (c + d\varepsilon) = (ac) + (ad+bc)\varepsilon$
    -   $f(a + b\varepsilon) = f(a) + b f'(a) \varepsilon$ (using Taylor expansion and $\varepsilon^2=0$)
3.  **Result**: The final output will be a dual number $f(x) + f'(x)\varepsilon$. The dual part contains the exact derivative of the function with respect to the chosen input $x_i$.

(See also: [[Dual Numbers and Forward AD]])

## Computational Cost

-   **Cost**: The computational cost of forward mode AD to compute one directional derivative is roughly a small constant factor (typically 2-4) times the cost of evaluating the original function.
-   **Jacobian Calculation**: To compute the full Jacobian matrix $J \in \mathbb{R}^{m \times n}$ for a function $f: \mathbb{R}^n \to \mathbb{R}^m$, one needs to run forward mode $n$ times, each time setting the dual part of a different input $x_i$ to 1. The total cost is $O(n \times \text{cost}(f))$.

## Use Cases in Machine Learning

Forward mode AD is generally **less efficient** than [[Reverse Mode Automatic Differentiation]] for training typical deep learning models where we need the gradient of a scalar loss with respect to many parameters ($n \gg m$). However, it has specific advantages:

1.  **Computing Jacobian-Vector Products (JVPs)**: Forward mode directly computes $Jv$. This is useful in certain optimization algorithms (e.g., Hessian-free optimization) or sensitivity analysis.
2.  **Few Inputs, Many Outputs**: Efficient when $n \ll m$. For example, calculating the tangent vector of a trajectory defined by an ODE.
3.  **Online / Streaming Computation**: Derivatives can be computed concurrently with the function evaluation without needing to store intermediate values for a backward pass.
4.  **Higher-Order Derivatives**: Conceptually simpler to extend to higher-order derivatives by using nested dual numbers (or truncated Taylor series).
5.  **Implementation Simplicity**: Often simpler to implement than reverse mode.

## Forward Primal Trace

In the context of AD implementations, the sequence of operations performed during the standard function evaluation (calculating the values) is sometimes referred to as the **forward primal trace**. Forward mode AD essentially augments this trace by computing the derivative values (the dual parts) concurrently.

## Comparison with Reverse Mode

| Feature | Forward Mode AD | [[Reverse Mode Automatic Differentiation]] |
|---|---|---|
| **Direction** | Computes derivatives forward | Computes derivatives backward |
| **Efficiency ($f: \mathbb{R}^n \to \mathbb{R}^m$)** | Best when $n \ll m$ | Best when $m \ll n$ |
| **Computes** | Jacobian-Vector Product ($Jv$) | Vector-Jacobian Product ($v^T J$) |
| **Memory** | Low, constant overhead | High, requires storing intermediate values (tape) |
| **ML Training** | Less common | Standard (Backpropagation) |

## Conclusion

Forward mode AD is a fundamental technique for automatic differentiation, intrinsically linked to dual numbers and the forward evaluation of a function. While less commonly used for standard deep learning gradient computation due to cost scaling with the number of inputs, it remains valuable for specific applications like Jacobian-vector products and situations with few inputs.

---

**References**:
1. Baydin, A. G., et al. (2018). Automatic differentiation in machine learning: a survey. *JMLR*, *18*, 1-43.
2. Griewank, A., & Walther, A. (2008). *Evaluating Derivatives*. SIAM. 