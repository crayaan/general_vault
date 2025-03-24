---
aliases:
  - Variational Calculus
  - Functional Derivatives
  - Variational Principles
---

# Calculus of Variations

## Overview

Calculus of Variations is a field of mathematical analysis that deals with finding extremal values of functionals - functions that take functions as inputs and return numbers. It forms the mathematical foundation for many fundamental principles in physics, including the [[Principle of Least Action]], [[Hamilton's Principle]], and [[Fermat's Principle]].

## Mathematical Foundations

### Functionals
A functional $J[y]$ maps a function $y(x)$ to a real number:

$$
J[y] = \int_{x_1}^{x_2} F(x, y, y') dx
$$

where:
- $F$ is called the integrand
- $y'$ represents the derivative of $y$
- $[x_1, x_2]$ is the interval of interest

### First Variation
The first variation $\delta J$ is analogous to the first derivative in ordinary calculus:

$$
\delta J = \lim_{\epsilon \to 0} \frac{J[y + \epsilon\eta] - J[y]}{\epsilon}
$$

where $\eta(x)$ is an arbitrary function vanishing at the endpoints.

## Fundamental Results

### Euler-Lagrange Equation
The condition for $J[y]$ to be stationary leads to:

$$
\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0
$$

This is the [[Euler-Lagrange Equation]], fundamental to physics.

### Higher Dimensions
For multiple variables:

$$
\frac{\partial F}{\partial y_i} - \sum_j \frac{\partial}{\partial x_j}\frac{\partial F}{\partial y_{i,j}} = 0
$$

## Applications in Physics

### Classical Mechanics
1. [[Principle of Least Action]]
   $$
   S = \int_{t_1}^{t_2} L(q, \dot{q}, t) dt
   $$

2. [[Hamilton's Principle]]
   - Stationary action principle
   - Equations of motion

3. [[Fermat's Principle]]
   - Path of light rays
   - Optical systems

### Field Theory
- [[Field Theory Variations]]
- [[Electromagnetic Field Theory]]
- [[Klein-Gordon Field]]
- [[Yang-Mills Theory]]

## Advanced Topics

### Multiple Integrals
- [[Multiple Variable Functionals]]
- [[Field Theory Functionals]]
- [[Surface Integrals]]

### Constraints
- [[Holonomic Constraints]]
- [[Non-holonomic Constraints]]
- [[Lagrange Multipliers]]

### Second Variation
- [[Sufficient Conditions]]
- [[Stability Analysis]]
- [[Conjugate Points]]

## Mathematical Methods

### Direct Methods
- [[Ritz Method]]
- [[Galerkin Method]]
- [[Finite Element Method]]

### Indirect Methods
- [[Euler-Lagrange Equation]]
- [[Hamilton-Jacobi Theory]]
- [[Canonical Transformations]]

## Applications Beyond Physics

### Engineering
- [[Optimal Control]]
- [[Shape Optimization]]
- [[Structural Design]]

### Economics
- [[Economic Optimization]]
- [[Resource Allocation]]
- [[Growth Models]]

### Biology
- [[Biological Systems]]
- [[Evolution Models]]
- [[Population Dynamics]]

## Computational Aspects

### Numerical Methods
- [[Finite Difference Methods]]
- [[Variational Integrators]]
- [[Optimization Algorithms]]

### Software Tools
- [[Computer Algebra Systems]]
- [[Numerical Libraries]]
- [[Visualization Tools]]

## Historical Development

### Early Work
- [[Bernoulli Brothers]]
- [[Euler's Contributions]]
- [[Lagrange's Work]]

### Modern Developments
- [[Hilbert's Work]]
- [[Modern Theory]]
- [[Recent Advances]]

## Common Problems

### Classical Problems
1. **Brachistochrone**
   - Finding curve of fastest descent
   - [[Brachistochrone Solution]]

2. **Minimal Surface**
   - Soap film problems
   - [[Plateau's Problem]]

3. **Geodesics**
   - Shortest paths
   - [[Differential Geometry]]

### Physical Problems
- [[Mechanical Systems]]
- [[Field Theories]]
- [[Quantum Systems]]

## Mathematical Tools

### Required Background
- [[Functional Analysis]]
- [[Differential Equations]]
- [[Real Analysis]]

### Advanced Concepts
- [[Functional Derivatives]]
- [[Weak Solutions]]
- [[Sobolev Spaces]]

## Common Techniques

### Solution Methods
1. Direct computation
2. Symmetry arguments
3. Conservation laws
4. [[Numerical Methods]]

### Verification
- [[Second Variation]]
- [[Legendre Condition]]
- [[Jacobi's Condition]]

## Further Reading

### Textbooks
- [[Variational Calculus Texts]]
- [[Mathematical Physics Books]]
- [[Applied Mathematics Resources]]

### Research Papers
- [[Historical Papers]]
- [[Modern Applications]]
- [[Current Research]]

---

Calculus of Variations provides the mathematical framework for understanding how nature optimizes various physical quantities. Its applications range from fundamental physics to modern engineering and optimization theory. 