---
aliases:
  - Euler-Lagrange Equations
  - Lagrange's Equations
  - Equations of Motion
---

# Euler-Lagrange Equation

## Overview

The Euler-Lagrange equation is a fundamental result in [[Calculus of Variations]] that provides the equations of motion for a physical system described by a [[Lagrangian]]. It emerges from the [[Principle of Least Action]] and represents the path that makes the action stationary.

## Mathematical Form

### General Form
For a system with generalized coordinates $q_i$, the equation is:

$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = 0
$$

where:
- $L$ is the [[Lagrangian]]
- $q_i$ are the [[Generalized Coordinates]]
- $\dot{q}_i$ are the generalized velocities

## Derivation

### From Variational Principle
1. Start with the action integral:
   $$
   S = \int_{t_1}^{t_2} L(q, \dot{q}, t) dt
   $$

2. Consider variation $\delta q$:
   $$
   \delta S = \int_{t_1}^{t_2} \left(\frac{\partial L}{\partial q}\delta q + \frac{\partial L}{\partial \dot{q}}\delta\dot{q}\right) dt
   $$

3. Integration by parts:
   $$
   \delta S = \left[\frac{\partial L}{\partial \dot{q}}\delta q\right]_{t_1}^{t_2} + \int_{t_1}^{t_2} \left(\frac{\partial L}{\partial q} - \frac{d}{dt}\frac{\partial L}{\partial \dot{q}}\right)\delta q\, dt
   $$

4. Apply boundary conditions ($\delta q(t_1) = \delta q(t_2) = 0$)

5. For stationary action ($\delta S = 0$), obtain Euler-Lagrange equation

## Applications

### Classical Mechanics
1. **Free Particle**
   - $L = \frac{1}{2}m\dot{x}^2$
   - Yields $m\ddot{x} = 0$

2. **Harmonic Oscillator**
   - $L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2$
   - Yields $m\ddot{x} + kx = 0$

3. **Central Force**
   - See [[Central Force Motion]]
   - Conservation of angular momentum emerges naturally

### Field Theory
- [[Field Theory Applications]]
- [[Electromagnetic Field Equations]]
- [[Klein-Gordon Equation]]

## Properties and Features

### 1. Conservation Laws
- Connection to [[Noether's Theorem]]
- [[Symmetries and Conservation Laws]]
- [[Conserved Currents]]

### 2. Coordinate Independence
- [[Generalized Coordinates]]
- [[Coordinate Transformations]]
- [[Gauge Invariance]]

### 3. Multiple Dimensions
- Systems with many degrees of freedom
- [[Coupled Systems]]
- [[Field Theories]]

## Extensions and Generalizations

### Higher Derivatives
- [[Higher-Order Lagrangians]]
- $$
  \frac{\partial L}{\partial q} - \frac{d}{dt}\frac{\partial L}{\partial \dot{q}} + \frac{d^2}{dt^2}\frac{\partial L}{\partial \ddot{q}} = 0
  $$

### Constrained Systems
- [[Holonomic Constraints]]
- [[Non-holonomic Constraints]]
- [[Lagrange Multipliers]]

### Dissipative Systems
- [[Rayleigh Dissipation Function]]
- [[Friction in Lagrangian Mechanics]]
- Modified equations of motion

## Computational Methods

### Numerical Solutions
- [[Numerical Integration Techniques]]
- [[Variational Integrators]]
- [[Error Analysis in Numerical Solutions]]

### Computer Algebra
- [[Symbolic Computation]]
- [[Automated Derivation]]
- [[Computer-Aided Analysis]]

## Common Problems and Solutions

### 1. Problem Types
- [[Initial Value Problems]]
- [[Boundary Value Problems]]
- [[Mixed Conditions]]

### 2. Solution Strategies
- [[Direct Integration]]
- [[Perturbation Methods]]
- [[Numerical Methods]]

## Advanced Topics

### Quantum Mechanics
- [[Path Integral Formulation]]
- [[Quantum Action Principle]]
- [[Quantum Field Theory]]

### Relativistic Systems
- [[Relativistic Particle]]
- [[Field Theory]]
- [[String Theory]]

## Historical Context

### Development
- [[Euler's Contributions]]
- [[Lagrange's Work]]
- [[Historical Evolution]]

### Impact
- [[Modern Physics Applications]]
- [[Engineering Applications]]
- [[Control Theory]]

## Common Mistakes and Pitfalls

1. Forgetting total time derivatives
2. Incorrect handling of constraints
3. Missing generalized forces
- [[Common Errors in Lagrangian Mechanics]]

## Further Reading

### Textbooks
- [[Classical Mechanics Texts]]
- [[Variational Calculus Books]]
- [[Mathematical Physics Resources]]

### Research Papers
- [[Historical Papers]]
- [[Modern Developments]]
- [[Current Applications]]

---

The Euler-Lagrange equation represents one of the most elegant and powerful results in theoretical physics, providing a universal way to derive equations of motion from a system's Lagrangian. 