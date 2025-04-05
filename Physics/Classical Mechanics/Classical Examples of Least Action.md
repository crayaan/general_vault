---
aliases:
  - Classical Action Examples
  - Least Action Problems
  - Variational Problems
---

# Classical Examples of Least Action

## Overview

This note presents key examples and problems that illustrate the [[Hamilton's Principle|Principle of Least Action]] and its applications in classical mechanics. Each example demonstrates how the principle leads to equations of motion and reveals important physical insights.

## Fundamental Examples

### 1. Free Particle
**Lagrangian**: $L = \frac{1}{2}m\dot{x}^2$

**Action**:
$$
S = \int_{t_1}^{t_2} \frac{1}{2}m\dot{x}^2 dt
$$

**Solution**:
- [[Euler-Lagrange Equation]] gives $\ddot{x} = 0$
- Linear motion: $x(t) = x_0 + vt$
- Demonstrates conservation of momentum

### 2. Harmonic Oscillator
**Lagrangian**: $L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2$

**Action**:
$$
S = \int_{t_1}^{t_2} (\frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2) dt
$$

**Solution**:
- Equation of motion: $m\ddot{x} + kx = 0$
- General solution: $x(t) = A\cos(\omega t + \phi)$
- Where $\omega = \sqrt{k/m}$

## Classical Problems

### 1. [[Brachistochrone Problem]]
**Problem**: Find the curve of fastest descent under gravity

**Solution**:
1. Parametric equations:
   $$
   x = a(t - \sin t)
   $$
   $$
   y = a(1 - \cos t)
   $$

2. Time of descent:
   $$
   T = \sqrt{\frac{2a}{g}}\int_0^{\theta} dt = \sqrt{\frac{2a}{g}}\theta
   $$

3. Shows cycloid is optimal path

### 2. [[Geodesics on a Sphere]]
**Problem**: Find shortest path between two points on a sphere

**Lagrangian**:
$$
L = \sqrt{R^2\sin^2\theta\,\dot{\phi}^2 + R^2\dot{\theta}^2}
$$

**Solution**:
- Great circles are geodesics
- Conservation of angular momentum: $R^2\sin^2\theta\,\dot{\phi} = \text{constant}$

## Advanced Examples

### 1. Double Pendulum
**Lagrangian**:
$$
L = \frac{1}{2}m_1l_1^2\dot{\theta}_1^2 + \frac{1}{2}m_2[l_1^2\dot{\theta}_1^2 + l_2^2\dot{\theta}_2^2 + 2l_1l_2\dot{\theta}_1\dot{\theta}_2\cos(\theta_1-\theta_2)] - (m_1+m_2)gl_1(1-\cos\theta_1) - m_2gl_2(1-\cos\theta_2)
$$

**Features**:
- Chaotic motion
- Multiple conservation laws
- Numerical solutions needed

### 2. Charged Particle in EM Field
**Lagrangian**:
$$
L = \frac{1}{2}m\dot{\mathbf{r}}^2 + \frac{e}{c}\mathbf{A}\cdot\dot{\mathbf{r}} - e\phi
$$

**Results**:
- Lorentz force law
- Gauge invariance
- Conservation of generalized momentum

## Computational Examples

### 1. Numerical Action Minimization
For particle in potential $V(x)$:

```python
def action(path, dt):
    T = np.sum(0.5 * m * np.diff(path)**2/dt - dt * V(path[:-1]))
    return T

def minimize_action(initial_path):
    return scipy.optimize.minimize(action, initial_path)
```

### 2. Variational Integrator
Discrete action:
$$
S_d = \sum_{k=0}^{N-1} L_d(q_k, q_{k+1})
$$

Implementation:
```python
def discrete_euler_lagrange(q_prev, q_curr, q_next, dt):
    D2Ld = partial_derivative(Ld, 2)
    D1Ld = partial_derivative(Ld, 1)
    return D2Ld(q_prev, q_curr) + D1Ld(q_curr, q_next)
```

## Special Cases

### 1. Constrained Motion
**Example**: Bead on a wire

**Lagrangian with constraint**:
$$
L = \frac{1}{2}m(\dot{x}^2 + \dot{y}^2) - mgy - \lambda(f(x,y))
$$

where $f(x,y)=0$ is the constraint.

### 2. Relativistic Particle
**Lagrangian**:
$$
L = -mc^2\sqrt{1-\frac{v^2}{c^2}}
$$

**Action**:
$$
S = -mc\int ds
$$

where $ds$ is proper time element.

## Common Techniques

### 1. Finding Constants of Motion
1. Cyclic coordinates: $\frac{\partial L}{\partial q_i} = 0$
2. Time independence: $H = \text{constant}$
3. Symmetries via [[Noether's Theorem]]

### 2. Phase Space Analysis
1. Canonical momenta: $p_i = \frac{\partial L}{\partial \dot{q}_i}$
2. Hamiltonian: $H = \sum_i p_i\dot{q}_i - L$
3. Phase portraits

## Problem-Solving Strategy

1. **Identify System**
   - Degrees of freedom
   - Constraints
   - Symmetries

2. **Write Lagrangian**
   - Kinetic energy
   - Potential energy
   - Constraints

3. **Apply Principle**
   - Euler-Lagrange equations
   - Conservation laws
   - Boundary conditions

4. **Solve Equations**
   - Analytical methods
   - Numerical methods
   - Approximations

## Mathematical Tools

### 1. Integration Techniques
- [[Action-Angle Variables]]
- [[Hamilton-Jacobi Theory]]
- [[Canonical Transformations]]

### 2. Numerical Methods
- [[Variational Integrators]]
- [[Symplectic Integration]]
- [[Energy-Momentum Methods]]

---

These examples demonstrate how the principle of least action provides a powerful and elegant framework for solving mechanical problems, from simple particles to complex dynamical systems. 