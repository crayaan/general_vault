---
aliases:
  - Field Theory Actions
  - QFT Dynamics
  - Quantum Fields
---

# Quantum Field Theory and Action

## Overview

Quantum Field Theory (QFT) represents the marriage of quantum mechanics with special relativity, where the [[Principle of Least Action]] plays a central role in determining both classical field equations and quantum dynamics through the path integral formulation.

## Field Theory Actions

### 1. Scalar Field
Klein-Gordon action:
$$
S[\phi] = \int d^4x \left(\frac{1}{2}\partial_\mu\phi\partial^\mu\phi - \frac{1}{2}m^2\phi^2 - V(\phi)\right)
$$

### 2. Dirac Field
$$
S[\psi,\bar{\psi}] = \int d^4x \,\bar{\psi}(i\gamma^\mu\partial_\mu - m)\psi
$$

### 3. Gauge Fields
Yang-Mills action:
$$
S[A] = -\frac{1}{4}\int d^4x\, F_{\mu\nu}^aF^{a\mu\nu}
$$

## Path Integral Formulation

### 1. Generating Functional
$$
Z[J] = \int \mathcal{D}\phi\, \exp\left(iS[\phi] + i\int d^4x\, J(x)\phi(x)\right)
$$

### 2. Correlation Functions
```python
def compute_correlator(points, action, sources):
    """Compute n-point correlation function"""
    Z = path_integral(action, sources)
    result = functional_derivative(Z, points)
    return result / Z[0]  # Normalize by partition function
```

## Quantum Effects

### 1. Loop Corrections
- One-loop effective action
- Renormalization
- Beta functions
- Anomalous dimensions

### 2. Feynman Diagrams
```python
def generate_feynman_diagrams(process, order):
    """Generate Feynman diagrams for a given process"""
    diagrams = []
    for topology in get_topologies(order):
        if valid_for_process(topology, process):
            diagrams.append(compute_amplitude(topology))
    return diagrams
```

## Symmetries and Conservation Laws

### 1. Noether's Theorem
From [[Noether's Theorem]]:
- Energy-momentum tensor
- Current conservation
- Ward identities
- Charge algebra

### 2. Gauge Symmetries
- BRST symmetry
- Ward-Takahashi identities
- Slavnov-Taylor identities
- Background field method

## Renormalization

### 1. Regularization Methods
- Dimensional regularization
- Pauli-Villars
- Lattice cutoff
- Heat kernel

### 2. Renormalization Group
$$
\beta(g) = \mu\frac{\partial g}{\partial \mu}
$$

## Advanced Topics

### 1. Effective Field Theory
- Wilson's approach
- Operator product expansion
- Matching conditions
- Power counting

### 2. Anomalies
- Chiral anomaly
- Trace anomaly
- Gravitational anomalies
- 't Hooft matching

### 3. Instantons
- Classical solutions
- Tunneling
- Theta vacua
- Index theorems

## Computational Methods

### 1. Lattice Field Theory
```python
def lattice_action(field_config, coupling):
    """Compute lattice action for scalar field"""
    action = 0
    for site in lattice_sites:
        for mu in range(4):
            action += (field_config[site + mu] - field_config[site])**2
    return action / (2 * coupling)
```

### 2. Perturbative Calculations
- Feynman rules
- Loop integrals
- Counterterms
- Renormalization conditions

## Applications

### 1. Standard Model
- Electroweak theory
- QCD
- Higgs mechanism
- CKM matrix

### 2. Condensed Matter
- Critical phenomena
- Superconductivity
- Quantum Hall effect
- Topological phases

## Mathematical Tools

### 1. Functional Methods
- Functional derivatives
- Grassmann variables
- Heat kernel expansion
- Zeta function regularization

### 2. Group Theory
- Lie algebras
- Representations
- Casimir operators
- Character formulas

## Modern Developments

### 1. Conformal Field Theory
- Operator product expansion
- Virasoro algebra
- Minimal models
- Bootstrap methods

### 2. Topological Field Theory
- Chern-Simons theory
- BF theory
- Witten-type theories
- TQFT axioms

## Computational Examples

### 1. Phi-Fourth Theory
```python
def phi4_action(phi, m, lambda_):
    """Action for φ⁴ theory"""
    kinetic = 0.5 * np.sum(gradient(phi)**2)
    mass = 0.5 * m**2 * np.sum(phi**2)
    interaction = (lambda_/4!) * np.sum(phi**4)
    return kinetic + mass + interaction
```

### 2. Monte Carlo Methods
```python
def metropolis_update(config, action, beta):
    """Metropolis algorithm for field configurations"""
    new_config = propose_update(config)
    delta_S = action(new_config) - action(config)
    if random() < exp(-beta * delta_S):
        return new_config
    return config
```

## Physical Implications

### 1. Particle Physics
- Mass generation
- Coupling unification
- CP violation
- Neutrino masses

### 2. Early Universe
- Phase transitions
- Inflation
- Baryogenesis
- Dark matter

---

Quantum Field Theory, built upon the principle of least action, provides our deepest understanding of fundamental particles and their interactions. The action principle guides both classical field equations and quantum dynamics through the path integral formulation. 