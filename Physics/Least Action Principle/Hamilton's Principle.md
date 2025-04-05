---
aliases:
  - Principle of Stationary Action
  - Hamilton's Variational Principle
  - Least Action Principle
---

# Hamilton's Principle (Principle of Stationary Action)

## Overview

Hamilton's Principle, also known as the **Principle of Stationary Action** (and often historically referred to as the Principle of Least Action), is a fundamental variational principle that underlies much of physics, from [[Classical Mechanics]] to [[Quantum Field Theory]]. It provides an alternative, powerful, and often more elegant formulation of the laws of motion compared to Newton's laws, stating that the path a physical system takes between two points in time is the one for which the [[Action (Physics)|action]] is stationary (usually, but not always, minimized).

## Statement of the Principle

Hamilton's Principle states that the actual path followed by a physical system between two specified states $q(t_1)$ and $q(t_2)$ at fixed times $t_1$ and $t_2$ is the one that makes the **action integral** $S$ **stationary** (i.e., its first variation $\delta S$ is zero) with respect to variations in the path $\delta q(t)$ that vanish at the endpoints ($\\delta q(t_1) = \delta q(t_2) = 0$).

Mathematically:
$$ \delta S = \delta \int_{t_1}^{t_2} L(q, \dot{q}, t) \, dt = 0 $$

Where:
- $S$ is the **[[Action (Physics)|action]]**, a functional representing the time integral of the Lagrangian.
- $L(q, \dot{q}, t)$ is the [[Lagrangian]] of the system, typically defined as $L = T - V$, where $T$ is the kinetic energy and $V$ is the potential energy for conservative systems.
- $q$ represents the generalized coordinates of the system.
- $\dot{q}$ represents the generalized velocities (time derivatives of $q$).
- $t$ is time.
- $\delta$ denotes a variation in the path.

## Derivation of Lagrange's Equations

Hamilton's Principle is the foundation from which the [[Euler-Lagrange Equation|Lagrange's equations of motion]] can be derived using the [[Calculus of Variations]].

1.  Consider a variation in the path: $q(t) \rightarrow q(t) + \delta q(t)$, which implies $\dot{q}(t) \rightarrow \dot{q}(t) + \delta \dot{q}(t) = \dot{q}(t) + \frac{d}{dt}(\delta q(t))$.
2.  The variation of the action is:
    $$ \delta S = \int_{t_1}^{t_2} \delta L \, dt = \int_{t_1}^{t_2} \sum_i \left( \frac{\partial L}{\partial q_i} \delta q_i + \frac{\partial L}{\partial \dot{q}_i} \delta \dot{q}_i \right) \, dt $$
3.  Integrate the second term by parts:
    $$ \int_{t_1}^{t_2} \frac{\partial L}{\partial \dot{q}_i} \delta \dot{q}_i \, dt = \left[ \frac{\partial L}{\partial \dot{q}_i} \delta q_i \right]_{t_1}^{t_2} - \int_{t_1}^{t_2} \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) \delta q_i \, dt $$
4.  The boundary term $\left[ \dots \right]_{t_1}^{t_2}$ vanishes because the variations are fixed at the endpoints ($\\delta q_i(t_1) = \delta q_i(t_2) = 0$).
5.  Substituting back into the expression for $\delta S$:
    $$ \delta S = \int_{t_1}^{t_2} \sum_i \left( \frac{\partial L}{\partial q_i} - \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) \right) \delta q_i \, dt $$
6.  For $\delta S$ to be zero for *arbitrary* variations $\delta q_i$, the term in the parentheses must be zero for each $i$. This yields the **[[Euler-Lagrange Equation]]**:
    $$ \frac{\partial L}{\partial q_i} - \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) = 0 $$

These are Lagrange's equations of motion, demonstrating that they are a direct consequence of Hamilton's Principle.

## Key Concepts

-   **[[Action (Physics)|Action ]]($S$)**: A scalar quantity whose stationarity determines the physical path.
-   **[[Lagrangian]] ($L$)**: The function $L = T - V$ that characterizes the system's dynamics.
-   **[[Calculus of Variations]]**: The mathematical framework for finding paths that make functionals like the action stationary.
-   **Stationary Path**: The actual physical path where $\delta S = 0$. Often a minimum for short paths, but can be a saddle point for longer paths.
-   **Generalized Coordinates**: Independent coordinates used to describe the system's configuration, useful for handling constraints.

## Significance and Advantages

-   **Universality**: Unifies classical mechanics, optics ([[Fermat's Principle]]), electromagnetism, relativity, and quantum mechanics under a single variational framework.
-   **Elegance and Coordinate Independence**: The scalar nature of the Lagrangian and action simplifies problem setup compared to Newton's vector approach.
-   **Constraints**: Naturally incorporates constraints via generalized coordinates.
-   **[[Symmetries and Conservation Laws]]**: [[Noether's Theorem]] links continuous symmetries of the Lagrangian (and thus the action) directly to conserved quantities (e.g., time invariance $\rightarrow$ energy conservation).
-   **Foundation for Advanced Physics**: Essential for [[Lagrangian Field Theory]], [[Quantum Field Theory]] (via the [[Path Integral Formulation]]), [[String Theory Action]], and the [[General Relativity and Action|Einstein-Hilbert action]] in General Relativity.

## Historical Development

- [[Fermat's Principle]] of least time in optics (1662)
- [[Maupertuis Principle]] of least action (1744) - related but distinct (fixed energy, variable time).
- [[Euler's Work]] formalizing the [[Calculus of Variations]].
- [[Lagrange's Contributions]] applying variational calculus to mechanics (Lagrangian Mechanics).
- [[Hamilton's Formulation]] providing the principle in its modern, widely used form (fixed time, variable energy).

## Applications Beyond Basic Mechanics

- [[Field Theory and Least Action]]
- [[Electromagnetic Field Theory]] (Maxwell's equations can be derived from an action)
- [[Quantum Mechanics and Path Integrals]] (Feynman's formulation sums over all possible paths, weighted by $e^{iS/\hbar}$)
- [[General Relativity and Action]] (Einstein-Hilbert action)

## Extended Hamilton's Principle

The principle can be extended to include non-conservative forces ($Q_{nc}$) by modifying the variational statement:
$$ \int_{t_1}^{t_2} (\delta L + \sum_i Q_{nc,i} \delta q_i) \, dt = 0 $$
This leads to Lagrange's equations with non-conservative forces:
$$ \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = Q_{nc,i} $$

## Common Misconceptions

- *"The path always minimizes the action"*: It only needs to be *stationary* (a minimum, maximum, or saddle point). "Least Action" is historical but technically less precise than "Stationary Action".
- *"It's just a mathematical reformulation"*: While mathematically equivalent to Newtonian mechanics for many systems, the action principle offers deeper insights into symmetries and conservation laws and provides the foundation for modern physics.

## Conclusion

Hamilton's Principle (the Principle of Stationary Action) offers a profound and unifying perspective on physics, reformulating dynamics in terms of optimizing a single scalar quantity – the action. Its elegance, power in handling complex systems, and deep connections to symmetries and conservation laws make it a cornerstone of classical and modern theoretical physics.

---

**References**:
1. Goldstein, H., Poole, C. P., & Safko, J. L. (2002). *Classical Mechanics* (3rd ed.). Addison Wesley.
2. Taylor, J. R. (2005). *Classical Mechanics*. University Science Books.
3. Lanczos, C. (1970). *The Variational Principles of Mechanics*. Dover Publications.
4. Gray, C. G. (2009). Principle of least action. *Scholarpedia*, 4(12):8291. [http://www.scholarpedia.org/article/Principle_of_least_action](http://www.scholarpedia.org/article/Principle_of_least_action) 