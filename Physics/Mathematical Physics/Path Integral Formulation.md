---
aliases:
  - Feynman Path Integrals
  - Path Integral Quantization
  - Quantum Paths
---

# Path Integral Formulation

## Overview

The path integral formulation provides a powerful alternative to canonical quantization, showing how quantum mechanics emerges from considering all possible paths between two points, not just the classical path of [[Principle of Least Action]]. Each path contributes with a phase factor determined by its classical action.

## Mathematical Framework

### Quantum Amplitude
The probability amplitude for a particle to go from point $x_a$ at time $t_a$ to point $x_b$ at time $t_b$ is:

$$
K(b,a) = \int \mathcal{D}[x(t)] \exp\left(\frac{i}{\hbar}S[x(t)]\right)
$$

where:
- $\mathcal{D}[x(t)]$ is the path integral measure
- $S[x(t)]$ is the [[Action (Physics)|action]] along path $x(t)$
- $\hbar$ is Planck's constant

### Connection to Classical Mechanics
In the limit $\hbar \to 0$, the dominant contribution comes from paths near the classical path where:

$$
\frac{\delta S}{\delta x} = 0
$$

This recovers the [[Principle of Least Action]].

## Explicit Calculations

### Free Particle
For a free particle with Lagrangian $L = \frac{1}{2}m\dot{x}^2$:

$$
K(x_b,t_b;x_a,t_a) = \sqrt{\frac{m}{2\pi i\hbar(t_b-t_a)}} \exp\left(\frac{im(x_b-x_a)^2}{2\hbar(t_b-t_a)}\right)
$$

### Harmonic Oscillator
For $L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}m\omega^2x^2$:

$$
K(x_b,t_b;x_a,t_a) = \sqrt{\frac{m\omega}{2\pi i\hbar\sin(\omega T)}} \exp\left(\frac{im\omega}{2\hbar\sin(\omega T)}[(x_a^2+x_b^2)\cos(\omega T)-2x_ax_b]\right)
$$

where $T = t_b-t_a$

## Applications

### Quantum Field Theory
- [[Field Theory Path Integrals]]
- Generating functionals
- Feynman diagrams
- Perturbation theory

### Statistical Mechanics
Connection to partition function:
$$
Z = \int \mathcal{D}[\phi] \exp(-S_E[\phi]/\hbar)
$$
where $S_E$ is the Euclidean action.

### Quantum Tunneling
Calculation of tunneling amplitudes through:
1. WKB approximation
2. Instanton methods
3. Complex classical paths

## Advanced Topics

### Gauge Theories
- [[Path Integrals in Gauge Theory]]
- Faddeev-Popov ghosts
- BRST symmetry

### Topological Effects
- [[Quantum Topology]]
- Aharonov-Bohm effect
- Instantons and theta vacua

## Computational Methods

### Numerical Techniques
1. **Monte Carlo Methods**
$$
⟨O⟩ = \frac{\int \mathcal{D}[\phi] O[\phi] e^{-S_E[\phi]}}{\int \mathcal{D}[\phi] e^{-S_E[\phi]}}
$$

2. **Lattice Regularization**
   - Discretized paths
   - Finite difference methods

### Series Methods
1. **Semiclassical Expansion**
$$
K = K_{cl}\exp\left(\sum_{n=1}^{\infty} \hbar^n S_n\right)
$$

2. **Perturbation Theory**
   - Loop expansion
   - Feynman rules

## Example Problems

### Double Slit Experiment
Path integral explanation:
1. Sum over all paths through both slits
2. Interference from action differences
3. Recovery of wave-particle duality

### Quantum Tunneling
Calculation for potential barrier:
$$
V(x) = \begin{cases} 
V_0 & 0 < x < a \\
0 & \text{otherwise}
\end{cases}
$$

Transmission amplitude:
$$
T = \frac{4k_1k_2}{(k_1+k_2)^2}e^{-2\kappa a}
$$

where:
- $k_1 = \sqrt{2mE}/\hbar$
- $k_2 = \sqrt{2m(E-V_0)}/\hbar$
- $\kappa = \sqrt{2m(V_0-E)}/\hbar$

### Aharonov-Bohm Effect
Phase difference for paths around solenoid:
$$
\Delta\phi = \frac{e}{\hbar c}\oint \mathbf{A}\cdot d\mathbf{x} = \frac{e}{\hbar c}\Phi
$$

## Common Techniques

### Stationary Phase Approximation
For large $S/\hbar$:
$$
I = \int dx\, e^{iS(x)/\hbar} \approx \sqrt{\frac{2\pi i\hbar}{S''(x_0)}}e^{iS(x_0)/\hbar}
$$

### Wick Rotation
Transform to Euclidean time:
$t \to -i\tau$
$S \to -S_E$

## Mathematical Tools

### Functional Integration
- [[Measure Theory]]
- [[Gaussian Integrals]]
- [[Functional Determinants]]

### Complex Analysis
- [[Steepest Descent]]
- [[Contour Integration]]
- [[Branch Cuts]]

---

The path integral formulation provides deep insights into quantum mechanics and its connection to classical physics through the principle of least action. Its mathematical structure reveals the profound relationship between quantum probability amplitudes and classical dynamics. 