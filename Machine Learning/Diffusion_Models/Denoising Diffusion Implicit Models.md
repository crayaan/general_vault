---
aliases:
  - DDIM
---

# Denoising Diffusion Implicit Models

Denoising Diffusion Implicit Models (DDIM) is a significant advancement in [[Diffusion Models Overview|diffusion models]] introduced by Song et al. in 2020. DDIM addresses one of the primary limitations of [[Denoising Diffusion Probabilistic Models|DDPM]] by dramatically accelerating the sampling process while maintaining high sample quality.

## Key Innovation

The key innovation in DDIM is the formulation of a non-Markovian diffusion process that allows for deterministic generation and faster sampling. Unlike the stochastic Markovian process in DDPM, DDIM enables:

1. **Accelerated sampling**: Generate high-quality samples with 10-50x fewer steps
2. **Deterministic generation**: Create consistent samples from the same latent point
3. **Latent space interpolation**: Meaningful interpolation between samples

## Mathematical Formulation

DDIM generalizes the [[Reverse Diffusion Process]] by introducing a family of non-Markovian inference processes parameterized by $\sigma_t$:

$$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\underbrace{\frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\alpha_t}}}_{\text{predicted }\mathbf{x}_0} + \underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(\mathbf{x}_t, t) + \sigma_t\epsilon_t}_{\text{direction pointing to }\mathbf{x}_t}$$

Where:
- $\alpha_t = 1 - \beta_t$ (from the forward process)
- $\epsilon_\theta(\mathbf{x}_t, t)$ is the noise prediction model
- $\epsilon_t \sim \mathcal{N}(0, \mathbf{I})$ is random noise
- $\sigma_t$ controls the stochasticity of the inference process

By setting $\sigma_t = 0$, we get the deterministic DDIM sampling process:

$$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t-1}}\cdot\epsilon_\theta(\mathbf{x}_t, t)$$

## Accelerated Sampling

DDIM enables accelerated sampling by taking larger steps in the reverse process, effectively skipping intermediate timesteps. Given a trained DDPM model with $T$ timesteps, DDIM can generate samples with only $S < T$ steps by using:

1. A subset of timesteps $\{\tau_1, \tau_2, ..., \tau_S\}$ where $\tau_1 = 1$ and $\tau_S = T$
2. A generalized non-Markovian reverse process that jumps directly between these timesteps

This allows for significantly faster sampling without retraining the model.

## Consistency Properties

The deterministic nature of DDIM enables several desirable properties:

### Consistency

Starting from the same noise $\mathbf{x}_T$, DDIM will always generate the same sample $\mathbf{x}_0$, regardless of the number of sampling steps (as long as the subset of timesteps is consistent).

### Interpolation in Latent Space

DDIM enables semantically meaningful interpolation between images by:
1. Encoding two images into their respective latent representations $\mathbf{x}_T^{(1)}$ and $\mathbf{x}_T^{(2)}$
2. Linearly interpolating between these points: $\mathbf{x}_T^{(\lambda)} = (1-\lambda)\mathbf{x}_T^{(1)} + \lambda\mathbf{x}_T^{(2)}$
3. Generating images from these interpolated latents using the deterministic DDIM process

This produces more coherent interpolations than possible with stochastic diffusion models.

## Generalization of DDPM

DDIM can be viewed as a generalization of DDPM, with both being special cases of a more general non-Markovian process:

- When $\sigma_t = \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t}$ (maximum stochasticity), we recover DDPM
- When $\sigma_t = 0$ (no stochasticity), we get deterministic DDIM

This parameter allows for a controlled trade-off between sample diversity and generation speed.

## Practical Implementation

To implement DDIM sampling:

1. Train a standard noise prediction model $\epsilon_\theta(\mathbf{x}_t, t)$ as in DDPM
2. Select a subset of timesteps for accelerated sampling
3. Use the deterministic DDIM update rule for generation

The same pre-trained DDPM model can be used with DDIM sampling without modification.

## Empirical Results

DDIM demonstrates several empirical advantages:

- Comparable sample quality to DDPM with 10-50x fewer sampling steps
- Better FID scores when using very few sampling steps (e.g., 10-20 steps)
- Consistent generation from the same latent, enabling applications like image editing and interpolation

## Applications

DDIM's innovations have enabled several practical applications:

- **Fast image generation**: High-quality samples with significantly reduced compute
- **Image editing**: Consistent generation allows for controlled manipulation
- **Latent space exploration**: Semantic interpolation between concepts
- **Foundation for further accelerated methods**: Basis for techniques like [[Progressive Distillation]]

## Limitations

Despite its advantages, DDIM has some limitations:

- Still slower than one-step generative models like GANs
- Quality degradation when using extremely few steps
- Less sample diversity compared to fully stochastic models

## Relationship to Other Methods

DDIM shares connections with:

- [[Score-based Generative Models|NCSN]]: Both leverage score functions in their formulation
- [[Consistency Models]]: Further acceleration of the sampling process
- [[Progressive Distillation]]: Using DDIM as a teacher model for distillation

## References

1. Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." arXiv Preprint arXiv:2010.02502.
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.

## Related Topics

- [[Diffusion Models with Fast Sampling]]
- [[Latent Diffusion Models]]
- [[Progressive Distillation]]
- [[Consistency Models]] 