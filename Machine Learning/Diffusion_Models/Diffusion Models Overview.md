---
aliases:
  - Diffusion Models
  - Diffusion Probabilistic Models
  - DDPM
---

# Diffusion Models Overview

Diffusion models represent one of the most significant developments in [[Generative AI]] in recent years. These models have revolutionized image synthesis, text generation, and other creative applications by providing high-quality generation capabilities with stable training dynamics.

## What Are Diffusion Models?

Diffusion models are a class of generative models inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise.

Unlike [[Variational Autoencoders|VAEs]] or [[Flow-based Models|flow models]], diffusion models are learned with a fixed procedure, and the latent variable has high dimensionality (same as the original data).

The key components of diffusion models include:

- [[Forward Diffusion Process|Forward diffusion process]] - gradually adding noise to data
- [[Reverse Diffusion Process|Reverse diffusion process]] - learning to denoise and recover the original data
- [[Score Matching|Score matching]] techniques for training

## Core Types of Diffusion Models

- [[Denoising Diffusion Probabilistic Models|DDPM]] (Ho et al., 2020)
- [[Score-based Generative Models|NCSN]] (Yang & Ermon, 2019)
- [[Denoising Diffusion Implicit Models|DDIM]] (Song et al., 2020)

## Key Features

- **Stable Training**: More stable training compared to GANs
- **High-Quality Generation**: Capable of producing detailed, high-fidelity samples
- **Flexible Architecture**: Adaptable to various domains and conditional generation tasks
- **Mathematical Tractability**: Both analytically tractable and flexible

## Advantages and Limitations

### Advantages
- High sample quality that rivals or exceeds GANs in many domains
- Stable training procedure
- Flexible conditioning mechanisms
- Strong theoretical foundations

### Limitations
- Slow sampling speed due to iterative denoising process
- Computationally expensive during inference
- Requires many diffusion steps for high-quality results

## Model Architectures

The two primary backbone architectures for diffusion models are:

- [[U-Net Architecture|U-Net]] - Commonly used for image-based diffusion models
- [[Diffusion Transformer|DiT]] - Transformer-based approach for higher efficiency

## Applications

- [[Image Generation|Image generation and editing]]
- [[Text to Image|Text-to-image synthesis]]
- [[Audio Generation|Audio generation]]
- [[Video Synthesis|Video synthesis]]
- [[3D Content Generation|3D content generation]]

## Recent Developments

- [[Latent Diffusion Models|Latent Diffusion Models (LDM)]]
- [[Consistency Models]]
- [[Progressive Distillation]]
- [[ControlNet]]

## References

1. Sohl-Dickstein et al. (2015) - "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
2. Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
3. Song et al. (2020) - "Denoising Diffusion Implicit Models"
4. Dhariwal & Nichol (2021) - "Diffusion Models Beat GANs"
5. Rombach & Blattmann et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
6. Song et al. (2023) - "Consistency Models"

## Related Topics
- [[Generative AI]]
- [[Machine Learning]]
- [[Deep Learning]]
- [[Neural Networks]]
- [[Generative Adversarial Networks|GANs]]
- [[Variational Autoencoders|VAEs]]