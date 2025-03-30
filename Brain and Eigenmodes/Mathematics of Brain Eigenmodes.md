---
aliases:
  - Eigenmode Mathematics
  - Connectome Harmonics Math
  - Spectral Graph Theory Brain
---

# Mathematics of Brain Eigenmodes

This note explores the mathematical foundations used to derive and understand [[Brain Eigenmodes Overview]]. The core idea is to apply principles from spectral graph theory or differential geometry to models of brain structure.

## 1. Connectome-Based Eigenmodes (Graph Laplacian Approach)

This approach focuses on the network of connections between brain regions.

1.  **Representing [[The Connectome]] as a Graph**: The brain is modelled as a graph $G = (V, E)$, where $V$ is the set of vertices (representing brain regions, often derived from an atlas or parcellation) and $E$ is the set of edges (representing anatomical connections, typically estimated from [[Diffusion MRI]] tractography).

2.  **Constructing the [[Graph Laplacian]]**: From the graph $G$, we construct the Graph Laplacian matrix $L$. As detailed in the [[Graph Laplacian]] note, $L = D - A$, where $D$ is the degree matrix and $A$ is the adjacency matrix of the connectome graph.

3.  **The Eigenvalue Problem**: The core mathematical step is solving the eigenvalue problem for the Laplacian matrix:
    $$ L \phi_k = \lambda_k \phi_k $$
    where:
    -   $\phi_k$ is the $k$-th eigenvector.
    -   $\lambda_k$ is the corresponding $k$-th eigenvalue.

4.  **Interpreting Eigenvectors and Eigenvalues**:
    -   **Eigenvectors ($\phi_k$)**: Each eigenvector $\phi_k$ is a vector where each element corresponds to a brain region. It represents a fundamental spatial pattern of activity across the brain regions – this is the **connectome eigenmode** or **connectome harmonic**. [\[1\]](https://www.nature.com/articles/ncomms10340)
    -   **Eigenvalues ($\lambda_k$)**: Each eigenvalue $\lambda_k$ represents the spatial frequency (or smoothness) of the corresponding eigenmode $\phi_k$. Lower eigenvalues correspond to smoother, large-scale patterns, while higher eigenvalues correspond to more rapidly varying, complex spatial patterns.

5.  **Decomposing [[Brain Activity Patterns]]**: Any observed pattern of brain activity $A(t)$ across the $n$ regions at time $t$ (represented as a vector) can be decomposed as a linear combination (superposition) of these eigenmodes:
    $$ A(t) = \sum_{k=1}^{n} c_k(t) \phi_k $$
    where $c_k(t)$ are time-varying coefficients indicating how much each eigenmode $\phi_k$ contributes to the overall activity pattern at time $t$.

## 2. Geometry-Based Eigenmodes (Laplace-Beltrami Approach)

This approach, strongly motivated by [[Neural Field Theory]], prioritizes the physical shape and continuous geometry of the cortical surface.

1.  **Representing Brain Geometry**: The cortical surface is modelled as a continuous 2D manifold (surface) embedded in 3D space, often derived from structural MRI scans.

2.  **The Laplace-Beltrami Operator ($\Delta_{LB}$)**: This operator is the generalization of the standard Laplacian to curved surfaces and manifolds. It measures the local curvature or "divergence of the gradient" on the cortical surface.

3.  **The Eigenvalue Problem**: Similar to the graph approach, we solve the eigenvalue problem for the Laplace-Beltrami operator on the cortical surface $\mathcal{M}$:
    $$ \Delta_{LB} \psi_k = -\kappa_k^2 \psi_k $$
    where:
    -   $\psi_k$ is the $k$-th eigenfunction (a continuous spatial pattern on the cortical surface).
    -   $-\kappa_k^2$ is the corresponding eigenvalue (often written with a negative sign and square by convention in this context).
    *Note: Appropriate boundary conditions are needed.* [\[2\]](https://www.nature.com/articles/s41586-023-06098-1)

4.  **Interpreting Eigenfunctions and Eigenvalues**:
    -   **Eigenfunctions ($\psi_k$)**: These are the **geometric eigenmodes**. They represent fundamental spatial patterns of standing waves determined purely by the shape (geometry) of the cortical surface. [\[2\]](https://www.nature.com/articles/s41586-023-06098-1)
    -   **Eigenvalues ($\kappa_k^2$)**: These relate to the spatial frequency of the geometric eigenmodes. Lower eigenvalues correspond to large-scale, slowly varying wave patterns, while higher eigenvalues represent finer, more complex patterns.

5.  **Decomposing Brain Activity**: Similar to the connectome approach, continuous patterns of cortical activity $A(x, t)$ (where $x$ is a point on the cortical surface) can be decomposed:
    $$ A(x, t) = \sum_{k=1}^{\infty} d_k(t) \psi_k(x) $$
    where $d_k(t)$ are time-varying coefficients.

## Connecting to Dynamics (Neural Field Theory)

[[Neural Field Theory]] often models cortical activity using wave equations where spatial coupling is key. A simplified form might look like:
$$ \frac{\partial A}{\partial t} = \mathcal{F}(A) + D \nabla^2 A + \text{Noise} $$
where $\nabla^2$ is the Laplacian operator (approximated by $L$ or $\Delta_{LB}$). The eigenmodes arise naturally as the solutions to the spatial part of such equations, representing the spatial patterns that are sustained by the system's structure and local dynamics.

## Geometric vs. Connectome Eigenmodes Debate

Recent research directly compares these two mathematical frameworks. Some studies suggest that **geometric eigenmodes** (derived from the Laplace-Beltrami operator) provide a more fundamental and parsimonious explanation for observed fMRI activity patterns than connectome eigenmodes (derived from the Graph Laplacian of dMRI tractography), particularly for large-scale patterns. [\[2\]](https://www.nature.com/articles/s41586-023-06098-1) This implies that the brain's overall shape might be a more dominant constraint on large-scale dynamics than the intricate details of long-range connections, although both likely play a role.

---

**References**:
1.  Atasoy, S., Donnelly, I., & Pearson, J. (2016). Human brain networks function in connectome-specific harmonic waves. *Nature Communications*, *7*, 10340. [https://www.nature.com/articles/ncomms10340](https://www.nature.com/articles/ncomms10340)
2.  Pang, J. C., Aquino, K. M., et al. (2023). Geometric constraints on human brain function. *Nature*, *618*, 566–574. [https://www.nature.com/articles/s41586-023-06098-1](https://www.nature.com/articles/s41586-023-06098-1) 