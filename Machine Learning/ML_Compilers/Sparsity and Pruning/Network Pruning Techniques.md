---
aliases:
  - Pruning Methods
  - Network Pruning Strategies
  - Pruning Schedules
---

# Network Pruning Techniques

Network pruning is the process of removing parameters (weights, neurons, channels, etc.) from a trained or partially trained [[Sparsity and Pruning Overview|dense network]] to create a smaller, more efficient [[Sparsity and Pruning Overview|sparse network]]. Various techniques exist, differing in *what* they prune (**pruning granularity**), *how* they decide what to prune (**importance criteria**), and *when* they prune (**pruning schedule**).

## 1. Pruning Granularity

This refers to the level at which pruning occurs:

-   **Unstructured Pruning (Fine-grained)**:
    -   Individual weights are removed (set to zero) based on some importance score.
    -   **Pros**: Can achieve the highest sparsity levels for a given accuracy drop.
    -   **Cons**: Results in irregular [[Sparse Tensor Formats|sparsity patterns]] that are difficult to accelerate on standard hardware (CPUs/GPUs) without specialized libraries/kernels.
-   **Structured Pruning (Coarse-grained)**:
    -   Entire structured blocks of weights are removed simultaneously.
    -   **Types**:
        -   **Neuron/Filter Pruning**: Removing entire neurons (columns in weight matrices) or convolution filters (along with their corresponding connections).
        -   **Channel Pruning**: Removing entire channels in convolutional layers.
        -   **Layer Pruning**: Removing entire layers.
        -   **Block Pruning**: Removing pre-defined blocks within weight matrices (e.g., N:M sparsity where N out of M weights in a block are zero).
    -   **Pros**: Creates regular sparsity patterns that are easier to implement efficiently on hardware, often leading to direct speedups and memory savings.
    -   **Cons**: Typically results in a larger accuracy drop for the same level of parameter reduction compared to unstructured pruning.
-   **Intra-Kernel Pruning vs. Kernel Pruning**:
    -   *Intra-kernel pruning* usually refers to unstructured or block pruning *within* a single kernel/layer's weight tensor.
    -   *Kernel pruning* (or filter/neuron pruning) removes the entire kernel/filter/neuron structure.

## 2. Importance Criteria

How to decide which elements are "unimportant" and can be pruned?

-   **Magnitude-Based Pruning**: The simplest and often surprisingly effective method. Assumes parameters with small absolute values contribute less to the output.
    -   *Local*: Prune weights below a threshold within each layer.
    -   *Global*: Prune weights below a single threshold across the entire network.
-   **Gradient-Based Pruning**: Uses gradient information (often combined with weight magnitude) to estimate importance. Example: Pruning weights whose removal causes the smallest increase in the loss function (e.g., Optimal Brain Damage, Optimal Brain Surgeon - computationally expensive).
-   **Activation-Based Pruning**: Pruning based on the activation values produced by neurons or filters (e.g., neurons with low average activation across a dataset).
-   **Other Criteria**: Taylor expansion-based methods, Hessian-based methods, sparsity-inducing regularization during training (e.g., L1 regularization).

## 3. Pruning Schedules

When and how often is pruning performed during the training process?

-   **One-Shot Pruning**: Train a dense model fully, then prune it once to the target sparsity, followed by fine-tuning.
-   **Iterative Pruning**: Repeatedly prune a small percentage of weights and fine-tune the network. This often achieves better accuracy for high sparsity levels than one-shot pruning.
    -   *Schedule Variations*: The amount pruned per iteration and the fine-tuning duration can vary (e.g., constant sparsity increase, gradual increase).
-   **Pruning During Training (Dynamic Pruning)**:
    -   Pruning occurs periodically *during* the initial training process.
    -   Some methods allow pruned weights to potentially regrow later if they become important again.
    -   Can sometimes reach target sparsity faster than post-training pruning.
-   **Pruning at Initialization (e.g., Lottery Ticket Hypothesis)**: Pruning occurs *before* any training, based on initial weight values or specific initialization schemes. The remaining subnetwork is then trained from scratch.

## Combining Techniques

Effective pruning often involves combining these aspects:

-   *Example Strategy*: Iteratively prune 20% of the remaining weights globally based on magnitude every 2 epochs, followed by 1 epoch of fine-tuning.
-   *Structured + Unstructured*: Apply filter pruning first, then fine-grained unstructured pruning within the remaining filters.

## Relationship with Compilers and Schedulers

-   **Sparse Kernels**: Achieving performance gains from sparsity relies on compilers generating efficient code (kernels) for sparse operations using appropriate [[Sparse Tensor Formats]].
-   **Auto-Schedulers**: Advanced compilers might incorporate auto-schedulers (like Halide, TVM, or specialized ones for sparse computations [\[1\]](https://arxiv.org/abs/2311.09549)) to automatically optimize the implementation of sparse kernels based on the specific sparsity pattern and target hardware.
-   **Framework Support**: Libraries like PyTorch and TensorFlow are increasingly adding support for sparse tensors and structured sparsity patterns to facilitate efficient execution.
-   **Progressive Optimization**: Concepts like Slapo [\[2\]](https://arxiv.org/abs/2302.08005) aim to decouple the model definition from optimization schedules, allowing progressive application of techniques like pruning and parallelism.

## Conclusion

Choosing the right network pruning technique involves balancing the desired sparsity level, acceptable accuracy loss, computational cost of the pruning process itself, and the potential for actual inference speedup on target hardware. Structured pruning is often preferred for hardware efficiency, while iterative, magnitude-based pruning is a common and effective baseline. The interaction between the pruning method, the resulting [[Sparse Tensor Formats|sparsity pattern]], and hardware-specific code generation is crucial for realizing the practical benefits of pruning.

---

**References**:
1. Dias, A., Anderson, L., Sundararajah, K., Pelenitsyn, A., & Kulkarni, M. (2024). SparseAuto: An Auto-Scheduler for Sparse Tensor Computations Using Recursive Loop Nest Restructuring. *arXiv preprint arXiv:2311.09549*. [https://arxiv.org/abs/2311.09549](https://arxiv.org/abs/2311.09549)
2. Chen, H., Yu, C. H., Zheng, S., Zhang, Z., Zhang, Z., & Wang, Y. (2023). Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training. *arXiv preprint arXiv:2302.08005*. [https://arxiv.org/abs/2302.08005](https://arxiv.org/abs/2302.08005) 