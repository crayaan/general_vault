---
aliases:
  - DDPM
---

# Denoising Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPM) is a groundbreaking approach to generative modeling introduced by Ho et al. in 2020. It represents one of the most influential implementations of the [[Diffusion Models Overview|diffusion model]] framework.

## Key Innovation

DDPM formalized a tractable approach to training and sampling from diffusion models that achieved state-of-the-art results in image generation. The key innovation was framing the problem as learning to reverse a fixed Markov chain of gradual noise addition.

## Architecture Overview

DDPM consists of:

1. A **forward diffusion process** that gradually adds Gaussian noise to data
2. A **reverse diffusion process** that learns to denoise by predicting the noise added at each step
3. A **U-Net architecture** with time conditioning to model the denoising function

## Mathematical Framework

### Forward Process

The forward process follows the standard [[Forward Diffusion Process]] approach, with a variance schedule $\beta_1,...,\beta_T$ that controls the noise addition rate.

### Reverse Process

DDPM parameterizes the [[Reverse Diffusion Process]] by training a neural network $\epsilon_\theta(\mathbf{x}_t, t)$ to predict the noise component added to the data. The model is trained with the simplified objective:

$$L_\text{simple} = \mathbb{E}_{t,\mathbf{x}_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

This is equivalent to optimizing a variational bound on the negative log-likelihood.

### Sampling

To generate new samples, DDPM follows this procedure:

1. Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. For $t = T, T-1, \ldots, 1$:
   - Sample $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ if $t > 1$, else $\mathbf{z} = 0$
   - Compute $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right) + \sigma_t\mathbf{z}$
3. Return $\mathbf{x}_0$

## Architectural Details

The noise prediction network $\epsilon_\theta$ in DDPM is typically implemented as a U-Net with:

- Time embedding to condition the model on the diffusion timestep
- Residual blocks for better gradient flow
- Attention mechanisms in certain layers
- Skip connections between corresponding layers

## Variance Schedule

DDPM typically uses a linear variance schedule:

$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

Where common values are $\beta_1 = 10^{-4}$ and $\beta_T = 0.02$.

## Comparison with Other Approaches

### Advantages over GANs

- More stable training
- No mode collapse
- Better coverage of data distribution
- No adversarial training

### Comparison with VAEs

- Higher quality samples
- No evidence lower bound (ELBO) gap
- More flexible architecture

## Limitations

- Slow sampling due to the sequential nature of the reverse process
- Computationally expensive inference
- Requires thousands of diffusion steps in the original formulation

## Extensions and Improvements

Several improvements to the original DDPM have been proposed:

- [[Improved DDPM]] - Better noise scheduling and architectural improvements
- [[Denoising Diffusion Implicit Models|DDIM]] - Accelerated sampling through non-Markovian diffusion
- [[Classifier Guided Diffusion]] - Improved sample quality through classifier guidance
- [[Latent Diffusion Models|LDM]] - Running diffusion in a compressed latent space for efficiency

## Applications

DDPM has been successfully applied to:

- High-quality image generation
- Conditional image synthesis
- Image inpainting and restoration
- Audio generation

## Impact and Legacy

DDPM marked a significant moment in generative modeling, demonstrating that diffusion models could outperform GANs in image quality. This work laid the foundation for many subsequent developments in the field, including:

- Text-to-image models like DALL-E 2 and Stable Diffusion
- Video generation models
- 3D content generation

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.

## Related Topics
- [[Score Matching]]
- [[U-Net Architecture]]
- [[Classifier-Free Guidance]]
- [[Diffusion Models with Fast Sampling]] 