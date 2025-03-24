---
aliases:
  - String Actions
  - Worldsheet Theory
  - String Dynamics
---

# String Theory and Action

## Overview

String theory represents a fundamental attempt to unify quantum mechanics and gravity by replacing point particles with one-dimensional strings. The dynamics of these strings are governed by the [[Principle of Least Action]], with several equivalent formulations of the action.

## Fundamental Actions

### 1. Nambu-Goto Action
The area of the worldsheet:
$$
S_{NG} = -T\int d\tau d\sigma \sqrt{-\det(\partial_\alpha X^\mu \partial_\beta X_\mu)}
$$

where:
- $T$ is the string tension
- $X^\mu(\tau,\sigma)$ are spacetime coordinates
- $\tau,\sigma$ are worldsheet coordinates

### 2. Polyakov Action
The equivalent but more tractable form:
$$
S_P = -\frac{T}{2}\int d\tau d\sigma \sqrt{-h}h^{\alpha\beta}\partial_\alpha X^\mu \partial_\beta X_\mu
$$

where $h_{\alpha\beta}$ is the worldsheet metric.

## Symmetries and Conservation Laws

### 1. Worldsheet Symmetries
- Diffeomorphism invariance
- Weyl invariance
- Conformal invariance
- Reparametrization invariance

### 2. Target Space Symmetries
- Poincaré invariance
- Gauge symmetries
- Supersymmetry (in superstring theory)

## Equations of Motion

### 1. Classical String Equations
From [[Euler-Lagrange Equation]]:
$$
\partial_\alpha(\sqrt{-h}h^{\alpha\beta}\partial_\beta X^\mu) = 0
$$

### 2. Constraints
- Virasoro constraints
- Light-cone gauge
- Conformal gauge conditions

## Quantization Methods

### 1. Light-Cone Quantization
```python
def compute_string_spectrum(N_max):
    """Compute string spectrum up to level N_max"""
    spectrum = []
    for N in range(N_max + 1):
        states = generate_states(N)
        masses = compute_masses(states)
        spectrum.extend(masses)
    return spectrum
```

### 2. BRST Quantization
- Ghost fields
- Physical state conditions
- Nilpotent BRST operator
- Cohomology classes

## String Perturbation Theory

### 1. Worldsheet Path Integral
$$
Z = \int \mathcal{D}X\mathcal{D}h\, e^{-S_P[X,h]}
$$

### 2. String Diagrams
- Tree level
- Loop corrections
- Moduli space
- String vertices

## Advanced Topics

### 1. D-branes
- Born-Infeld action
- Gauge theory on branes
- Intersecting branes
- Boundary conditions

### 2. T-duality
- Compact dimensions
- Winding modes
- R ↔ 1/R symmetry
- Enhanced symmetry points

### 3. String Field Theory
- Cubic interaction
- Background independence
- Tachyon condensation
- Non-perturbative effects

## Applications

### 1. AdS/CFT Correspondence
- Holographic principle
- Wilson loops
- Correlation functions
- Entanglement entropy

### 2. String Cosmology
- Inflation models
- Moduli stabilization
- Flux compactification
- String landscape

## Computational Methods

### 1. Numerical String Evolution
```python
def evolve_classical_string(initial_data, dt, steps):
    """Evolve classical string using discretized equations"""
    X = initial_data.copy()
    for step in range(steps):
        X = update_string_position(X, dt)
        apply_constraints(X)
    return X
```

### 2. Mode Expansion
- Oscillator modes
- Mass spectrum
- Level matching
- GSO projection

## Mathematical Tools

### 1. Complex Analysis
- Conformal mappings
- Modular forms
- Theta functions
- Partition functions

### 2. Differential Geometry
- Riemann surfaces
- Calabi-Yau manifolds
- Characteristic classes
- Index theorems

## Physical Implications

### 1. Unification
- Gravity and gauge forces
- Supersymmetry
- Extra dimensions
- Dualities

### 2. Quantum Gravity
- UV completeness
- Black hole microstates
- Information paradox
- Holography

## Modern Developments

### 1. M-theory
- 11 dimensions
- Membrane actions
- U-duality
- Matrix theory

### 2. F-theory
- Non-perturbative aspects
- Geometric engineering
- Singularity resolution
- Phenomenology

---

String theory, through its various action principles, provides a rich framework for understanding fundamental physics. The principle of least action remains central to its formulation, leading to profound connections between geometry, quantum mechanics, and gravity. 