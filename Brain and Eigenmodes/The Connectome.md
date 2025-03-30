---
aliases:
  - Brain Connectivity
  - Neural Wiring
---

# The Connectome

The **connectome** refers to the comprehensive map of neural connections in the brain, often described as its "wiring diagram". It encompasses the physical pathways (primarily white matter tracts composed of axons) that link different brain regions.

## Levels of Connectomics

1.  **Macroscale**: Maps large brain regions and the major fibre bundles connecting them. Typically studied using non-invasive neuroimaging techniques like [[Diffusion MRI]]. This is the scale relevant to [[Brain Eigenmodes Overview]].
2.  **Mesoscale**: Focuses on the connectivity of local circuits within specific brain regions, mapping connections between different types of neurons.
3.  **Microscale**: Aims to map every single synaptic connection between individual neurons. Currently only feasible for very small organisms or tiny portions of larger brains due to immense complexity.

## Role in Brain Function

The connectome provides the physical substrate upon which neural communication and computation occur. Its structure constrains:

-   How signals propagate through the brain.
-   The speed of communication between regions.
-   The possible patterns of synchronous activity ([[Brain Activity Patterns]]).
-   The natural resonant frequencies or [[Brain Eigenmodes Overview|eigenmodes]] of brain activity.

## Relation to Eigenmodes

The specific structure of the macroscale connectome determines the shape and frequency of the brain's eigenmodes. Just as the shape and material of a bell determine how it rings, the pattern of connections in the brain determines its fundamental modes of oscillation. Techniques like graph theory are used to analyze the connectome, often representing it as a network graph where brain regions are nodes and connections are edges. The [[Graph Laplacian]] of this network is key to calculating the eigenmodes. [\[1\]](https://www.nature.com/articles/ncomms10340)

## Studying the Connectome

-   **[[Diffusion MRI]] (dMRI/DTI)**: Measures the diffusion of water molecules to infer the orientation of white matter tracts non-invasively in humans.
-   **Tractography Algorithms**: Computational methods used to reconstruct fibre pathways from dMRI data.
-   **Tracer Studies**: Invasive techniques used in animal models where substances are injected and traced along neuronal pathways (considered the gold standard but not applicable to humans in vivo).

---

**References**:
1.  Sporns, O. (2013). The human connectome: origins and challenges. *NeuroImage*, *80*, 53–61. [https://pubmed.ncbi.nlm.nih.gov/23528922/](https://pubmed.ncbi.nlm.nih.gov/23528922/) 