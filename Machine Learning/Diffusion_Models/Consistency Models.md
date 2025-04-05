# Consistency Models

Consistency Models, introduced by Song et al. in 2023, represent a significant breakthrough in [[Diffusion Models Overview|diffusion models]] by enabling high-quality generation in a single step or very few steps. They address the primary limitation of traditional diffusion models: the need for many sequential denoising steps.

## Core Innovation

The key innovation of Consistency Models is the introduction of a self-consistency framework that maps any point on a diffusion trajectory directly to the endpoint. This enables:

1. **One-step generation**: Creating high-quality samples in a single step
2. **Few-step sampling**: Progressive refinement with just a few consistency steps
3. **Trajectory consistency**: Ensuring all points on a trajectory map to the same endpoint

## Conceptual Framework

Consistency Models are built on a fundamental insight: instead of learning the step-by-step denoising process, learn a mapping that directly projects any noisy point on a trajectory to its origin.

Given a trajectory $\{\mathbf{x}_t \vert t \in [\epsilon, T]\}$, the **consistency function** $f$ is defined as:

$$f: (\mathbf{x}_t, t) \mapsto \mathbf{x}_\epsilon$$

Where the self-consistency property ensures:

$$f(\mathbf{x}_t, t) = f(\mathbf{x}_{t'}, t') = \mathbf{x}_\epsilon$$

for all $t, t' \in [\epsilon, T]$.

## Mathematical Formulation

### Model Parameterization

Consistency Models parameterize the consistency function as:

$$f_\theta(\mathbf{x}, t) = c_\text{skip}(t)\mathbf{x} + c_\text{out}(t) F_\theta(\mathbf{x}, t)$$

Where:
- $c_\text{skip}(t)$ and $c_\text{out}(t)$ are designed such that $c_\text{skip}(\epsilon) = 1, c_\text{out}(\epsilon) = 0$
- $F_\theta(\mathbf{x}, t)$ is a neural network with parameters $\theta$

This ensures that when $t=\epsilon$, $f_\theta$ becomes an identity function.

### Training Objectives

Consistency Models can be trained using two main approaches:

#### 1. Distillation Training

Uses a pre-trained diffusion model as a teacher:

$$\mathcal{L}_\text{distill}(\theta) = \mathbb{E}_{t, \mathbf{x}_t, \mathbf{x}_\epsilon} \left[ d\left(f_\theta(\mathbf{x}_t, t), \mathbf{x}_\epsilon \right) \right]$$

Where:
- $\mathbf{x}_\epsilon$ is obtained from a pre-trained diffusion model
- $d$ is a distance function (e.g., L2 or LPIPS)

#### 2. Consistency Training

Directly enforces the self-consistency property without a teacher model:

$$\mathcal{L}_\text{consist}(\theta) = \mathbb{E}_{t, t', \mathbf{x}} \left[ d\left(f_\theta(CD(f_\theta(\mathbf{x}, t), t, t'), t'), f_\theta(\mathbf{x}, t) \right) \right]$$

Where $CD$ is a function that adds noise from time $t$ to $t'$.

## Generation Process

### One-Step Generation

Sampling from a Consistency Model is remarkably simple:

1. Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ from a standard Gaussian
2. Apply the consistency function: $\mathbf{x}_0 = f_\theta(\mathbf{x}_T, T)$

### Multi-Step Generation (PF-ODE)

For higher quality results, the probability flow ODE can be used with just a few steps:

1. Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. Discretize $[0, T]$ into $n$ steps: $T = s_n > s_{n-1} > ... > s_1 > s_0 = 0$
3. For $i = n, n-1, ..., 1$:
   - $\mathbf{x}_{s_{i-1}} = f_\theta(\mathbf{x}_{s_i}, s_i)$
4. Return $\mathbf{x}_0 = \mathbf{x}_{s_0}$

Even with $n=2$ or $n=4$, this produces high-quality results.

## Advantages Over Traditional Diffusion Models

Consistency Models offer several significant advantages:

- **Speed**: 10-50× faster generation than traditional diffusion models
- **Efficiency**: Substantially reduced computational requirements
- **Quality**: Competitive sample quality, especially with a few refinement steps
- **Reduced Memory**: Lower memory footprint during inference
- **Latent Space Structure**: Preserves semantic structure of the latent space

## Applications

Consistency Models have shown promising results in several domains:

- **Image Generation**: High-quality image synthesis in a single step
- **Text-to-Image**: Conditioning on text for controlled generation
- **Image Editing**: Fast modifications of existing images
- **Video Generation**: Potential for rapid video synthesis

## Comparison with Other Fast Diffusion Methods

Several methods exist for accelerating diffusion models:

| Method | Approach | Steps | Quality | Training |
|--------|----------|-------|---------|----------|
| DDIM | Non-Markovian sampling | 20-50 | High | Standard |
| Progressive Distillation | Iterative knowledge distillation | 4-8 | High | Multiple stages |
| Consistency Models | Direct mapping | 1-4 | High | Single stage |

## Limitations

Despite their advantages, Consistency Models have some limitations:

- **Quality Gap**: One-step generation still has a slight quality gap compared to many-step diffusion
- **Training Complexity**: Especially in the consistency training regime
- **Conditional Generation**: May require additional techniques for complex conditional generation

## Recent Developments

Building upon Consistency Models:

- **Consistency Distillation**: Improved training approaches
- **Adversarial Consistency Models**: Combining with GAN-like training
- **Latent Consistency Models**: Applying consistency models in latent space

## References

1. Song, Y., Durkan, C., et al. (2023). "Consistency Models." arXiv Preprint arXiv:2303.01469.
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
3. Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." arXiv Preprint arXiv:2010.02502.

## Related Topics

- [[Progressive Distillation]]
- [[Denoising Diffusion Implicit Models|DDIM]]
- [[Diffusion Models with Fast Sampling]]
- [[Classifier-Free Guidance]] 