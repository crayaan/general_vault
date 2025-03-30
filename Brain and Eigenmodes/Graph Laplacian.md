---
aliases:
  - Laplacian Matrix
---

# Graph Laplacian

The **Graph Laplacian** is a matrix representation of a graph that is fundamental in spectral graph theory and has important applications in various fields, including machine learning, image processing, and the study of [[Brain Eigenmodes Overview|brain connectomics]]. Its properties are central to the [[Mathematics of Brain Eigenmodes]].

## Definition

For a simple graph $G$ with $n$ vertices, the Laplacian matrix $L$ is an $n \times n$ matrix defined as:

$L = D - A$

where:
- $D$ is the **degree matrix**: a diagonal matrix where $D_{ii}$ is the degree of vertex $i$ (the number of edges connected to it), and all off-diagonal elements are zero.
- $A$ is the **adjacency matrix**: an $n \times n$ matrix where $A_{ij} = 1$ if there is an edge between vertex $i$ and vertex $j$, and $A_{ij} = 0$ otherwise.

Alternatively, the elements of $L$ can be defined directly as:

$$ L_{ij} = 
\begin{cases} 
\text{deg}(v_i) & \text{if } i = j \\
-1 & \text{if } i \neq j \text{ and } v_i \text{ is adjacent to } v_j \\
0 & \text{otherwise} 
\end{cases} $$

## Properties

- $L$ is always symmetric for undirected graphs.
- $L$ is positive semi-definite.
- The smallest eigenvalue of $L$ is always 0, corresponding to the eigenvector where all entries are 1.
- The number of times 0 appears as an eigenvalue equals the number of connected components in the graph.

## Relation to Brain Eigenmodes

In the context of brain connectomics, as detailed in [[Mathematics of Brain Eigenmodes]]:
1.  The brain's structural network ([[The Connectome]]) is represented as a graph, where brain regions are vertices and anatomical connections (e.g., white matter tracts estimated from [[Diffusion MRI]]) are edges.
2.  The Graph Laplacian $L$ is constructed for this connectome graph.
3.  The **eigenvectors** of $L$ correspond to the **brain eigenmodes** (or connectome harmonics). These represent the fundamental spatial patterns of activity that the brain's structure can support. [\[1\]](https://www.nature.com/articles/ncomms10340)
4.  The **eigenvalues** of $L$ correspond to the spatial frequencies of these eigenmodes. Smaller eigenvalues correspond to smoother, large-scale patterns, while larger eigenvalues correspond to more complex, rapidly varying patterns.

Essentially, the Graph Laplacian captures how activity might "diffuse" or spread across the connectome structure. Its eigenvectors provide a natural basis set (like a Fourier basis, but adapted to the specific connectome structure) to decompose and analyze complex [[Brain Activity Patterns]].

## Variants

- **Normalized Laplacian**: Often used to account for variations in node degrees. Common forms include the symmetric normalized Laplacian ($L_{sym} = D^{-1/2} L D^{-1/2}$) and the random-walk normalized Laplacian ($L_{rw} = D^{-1} L$).

---

**References**:
1.  Atasoy, S., Donnelly, I., & Pearson, J. (2016). Human brain networks function in connectome-specific harmonic waves. *Nature Communications*, *7*, 10340. [https://www.nature.com/articles/ncomms10340](https://www.nature.com/articles/ncomms10340) 