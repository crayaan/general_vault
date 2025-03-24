---
aliases:
  - Discrete Mechanics
  - Geometric Integrators
  - Structure-Preserving Integration
---

# Variational Integrators

## Overview

Variational integrators are numerical methods that preserve the geometric structure of mechanical systems by discretizing the [[Principle of Least Action]] rather than the equations of motion directly. This approach leads to superior long-term behavior in numerical simulations.

## Mathematical Foundation

### Discrete Action Principle
For a continuous Lagrangian $L(q,\dot{q})$, define discrete Lagrangian:
$$
L_d(q_k,q_{k+1}) \approx \int_{t_k}^{t_{k+1}} L(q(t),\dot{q}(t))dt
$$

### Discrete Euler-Lagrange Equations
$$
D_2L_d(q_{k-1},q_k) + D_1L_d(q_k,q_{k+1}) = 0
$$

## Implementation

### 1. Basic Structure
```python
class VariationalIntegrator:
    def __init__(self, L_d, dt):
        self.L_d = L_d
        self.dt = dt
    
    def step(self, q_prev, q_curr):
        def residual(q_next):
            return D2L_d(q_prev, q_curr) + D1L_d(q_curr, q_next)
        
        return newton_solve(residual)
```

### 2. Common Discretizations

#### Midpoint Rule
$$
L_d(q_k,q_{k+1}) = h L\left(\frac{q_k + q_{k+1}}{2}, \frac{q_{k+1}-q_k}{h}\right)
$$

#### Störmer-Verlet
$$
L_d(q_k,q_{k+1}) = h L\left(q_k, \frac{q_{k+1}-q_k}{h}\right)
$$

## Conservation Properties

### 1. Momentum Maps
- Preserved exactly for symmetries of $L_d$
- Linear and angular momentum conservation
- Discrete Noether's theorem

### 2. Energy Behavior
- Near-conservation of energy
- Bounded energy oscillation
- No secular growth

## Applications

### 1. Molecular Dynamics
```python
def molecular_dynamics_step(positions, velocities, forces):
    q_next = variational_step(positions, velocities)
    v_next = (q_next - positions) / dt
    return q_next, v_next
```

### 2. Celestial Mechanics
- N-body simulations
- Orbital dynamics
- Long-term stability

### 3. Rigid Body Dynamics
- Rotation matrices
- Quaternions
- Lie group methods

## Advanced Topics

### 1. Constrained Systems
- Discrete constraints
- Lagrange multipliers
- Holonomic constraints
- Non-holonomic systems

### 2. Multisymplectic Integrators
- Space-time discretization
- Field theories
- Wave equations
- Maxwell's equations

### 3. Backward Error Analysis
- Modified equations
- Structure preservation
- Long-term behavior
- Geometric interpretation

## Practical Implementation

### 1. Error Analysis
```python
def compute_energy_error(traj):
    E0 = compute_energy(traj[0])
    errors = [abs(compute_energy(q) - E0) for q in traj]
    return np.array(errors)
```

### 2. Adaptive Methods
- Time step selection
- Error estimates
- Symplectic composition
- Multiple time scales

## Example Problems

### 1. Simple Pendulum
```python
def pendulum_L_d(q1, q2, h):
    """Discrete Lagrangian for pendulum"""
    v = (q2 - q1) / h
    T = 0.5 * m * v**2
    V = m * g * (1 - cos(q1))  # Using q1 for position
    return h * (T - V)
```

### 2. Double Pendulum
- Chaotic dynamics
- Energy preservation
- Phase space structure
- Poincaré sections

## Comparison with Traditional Methods

### 1. Advantages
- Long-term stability
- Conservation laws
- Geometric structure
- Phase space preservation

### 2. Challenges
- Implementation complexity
- Computational cost
- Implicit equations
- Constraint handling

## Software Tools

### 1. Python Libraries
```python
# Example using sympy for automatic derivation
from sympy import *

def derive_discrete_eom(L_d):
    q_k = Symbol('q_k')
    q_kp1 = Symbol('q_kp1')
    
    # Compute partial derivatives
    D1L = diff(L_d, q_k)
    D2L = diff(L_d, q_kp1)
    
    return D1L, D2L
```

### 2. Specialized Software
- GeomVI
- Basilisk
- FEniCS
- DiffEqBase.jl

---

Variational integrators represent a powerful approach to numerical simulation that respects the geometric structure of mechanical systems. Their foundation in the principle of least action ensures superior long-term behavior compared to traditional numerical methods. 