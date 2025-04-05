# Reverse Diffusion Process

The reverse diffusion process is the heart of [[Diffusion Models Overview|diffusion models]], involving the step-by-step denoising of random noise to generate data samples. This process reverses the [[Forward Diffusion Process]] and is what enables diffusion models to generate new data.

## Core Concept

While the forward process gradually adds noise to data, the reverse process learns to gradually remove noise, starting from pure noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ and working backward to recover a clean sample $\mathbf{x}_0$.

## Mathematical Formulation

The true reverse process follows the posterior distribution:

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$$

Where:
- $\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0$
- $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

Since we don't know $\mathbf{x}_0$ during generation, we approximate the reverse process with a learned model $p_\theta$:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

Where:
- $\mu_\theta(\mathbf{x}_t, t)$ is the predicted mean
- $\Sigma_\theta(\mathbf{x}_t, t)$ is the predicted variance

## Parameterization Approaches

### Predicting the Noise

The most common approach (DDPM) is to train a neural network $\epsilon_\theta(\mathbf{x}_t, t)$ to predict the noise component that was added during the forward process. This leads to:

$$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right)$$

### Predicting $\mathbf{x}_0$ Directly

Alternatively, some models directly predict $\mathbf{x}_0$ from $\mathbf{x}_t$:

$$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

### Variance Choices

Several choices exist for $\Sigma_\theta(\mathbf{x}_t, t)$:

1. **Fixed variance**: $\Sigma_\theta = \beta_t \mathbf{I}$
2. **Learned diagonal variance**: $\Sigma_\theta = \text{diag}(\sigma^2_\theta(\mathbf{x}_t, t))$
3. **Interpolated variance**: $\Sigma_\theta = \beta_t \mathbf{I}$ or $\tilde{\beta}_t \mathbf{I}$

## Sampling Algorithm

The standard sampling algorithm for the reverse diffusion process is:

1. Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. For $t = T, T-1, \ldots, 1$:
   - Sample $\mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$
3. Return $\mathbf{x}_0$

## Deterministic Sampling

[[Denoising Diffusion Implicit Models|DDIM]] introduced a deterministic sampling approach:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(\mathbf{x}_t, t)$$

This allows for faster sampling with fewer steps and enables latent space interpolation.

## Training Objective

The reverse process is typically trained by minimizing a variant of the following objective:

$$L_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2 \right]$$

Where:
- $t$ is uniformly sampled from $\{1, 2, \ldots, T\}$
- $\mathbf{x}_0$ is sampled from the training data
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$

## Acceleration Techniques

Several techniques have been proposed to accelerate the reverse diffusion process:

- [[Progressive Distillation]] - Reduces the number of required sampling steps through iterative knowledge distillation
- [[Consistency Models]] - Maps any noisy latent directly to the denoised sample in one step
- [[Diffusion Models with Fast Sampling|Fast sampling methods]] - Various approaches to reduce the number of steps needed

## Conditioning Mechanisms

The reverse process can be conditioned on additional information $\mathbf{y}$ (such as class labels, text descriptions, etc.):

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{y})$$

This allows for controlled generation by guiding the denoising process toward samples that match the conditioning information.

## Related Concepts

- [[Score Matching]] - Connection between diffusion models and score-based generative models
- [[Classifier Guided Diffusion]] - Using a classifier to guide the reverse process
- [[Classifier-Free Guidance]] - Guiding the reverse process without an explicit classifier

## References

1. Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
2. Song et al. (2020) - "Denoising Diffusion Implicit Models"
3. Dhariwal & Nichol (2021) - "Diffusion Models Beat GANs" 