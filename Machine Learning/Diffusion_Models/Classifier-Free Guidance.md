# Classifier-Free Guidance

Classifier-Free Guidance (CFG) is a powerful technique introduced by Ho and Salimans in 2021 that significantly improves the quality and controllability of conditional [[Diffusion Models Overview|diffusion models]]. It enables better adherence to conditioning inputs without requiring a separate classifier model.

## Core Concept

The key insight of Classifier-Free Guidance is to jointly train both conditional and unconditional diffusion models, then use a weighted combination of their scores during sampling to amplify the influence of the conditioning information.

This approach creates a tunable parameter (guidance scale) that controls the trade-off between sample quality and diversity, allowing for precise control over how strictly the model follows the conditioning input.

## Mathematical Formulation

### Traditional Classifier Guidance

Traditional classifier guidance uses a separate classifier $p(y|\mathbf{x}_t)$ to guide the reverse diffusion process:

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t)$$

### Classifier-Free Approach

Classifier-Free Guidance directly combines the conditional and unconditional score functions:

$$\tilde{\nabla}_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + w \cdot \left( \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \right)$$

Which simplifies to:

$$\tilde{\nabla}_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) = (1-w) \cdot \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + w \cdot \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|y)$$

Where $w \geq 1$ is the guidance scale that controls the strength of the conditioning.

### Implementation in Practice

In diffusion models that predict noise $\epsilon_\theta$, CFG is implemented as:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t, y) = \epsilon_\theta(\mathbf{x}_t, t) + w \cdot (\epsilon_\theta(\mathbf{x}_t, t, y) - \epsilon_\theta(\mathbf{x}_t, t))$$

Where:
- $\epsilon_\theta(\mathbf{x}_t, t)$ is the unconditional noise prediction
- $\epsilon_\theta(\mathbf{x}_t, t, y)$ is the conditional noise prediction
- $w$ is the guidance scale

## Training Approach

The implementation of CFG requires training a model that can function both conditionally and unconditionally:

### Joint Training

The most common approach is to use a single model trained with conditioning dropout:

1. During training, randomly set the conditioning input $y$ to a null token with probability $p$ (typically 0.1-0.2)
2. When the conditioning is dropped, the model learns to predict unconditionally
3. When conditioning is provided, the model learns to use it

### Implementation in Sampling

During the sampling process:

1. Perform two forward passes through the model for each denoising step:
   - One with the conditioning input $y$
   - One with the null conditioning
2. Combine the results using the guidance scale formula
3. Use the combined prediction for the denoising step

## Effects of Guidance Scale

The guidance scale $w$ has several important effects:

- **$w = 1$**: Equivalent to using standard conditional generation
- **$w < 1$**: Reduces the impact of conditioning, increasing diversity
- **$w > 1$**: Amplifies the influence of conditioning, improving adherence
- **$w \gg 1$**: Creates high-fidelity but potentially less diverse samples

Most implementations typically use values between 2.5 and 15, with 7.5 being a common default that balances quality and diversity.

## Advantages and Benefits

Classifier-Free Guidance offers several key advantages:

- **No Separate Classifier**: Eliminates the need to train a separate classifier model
- **Computational Efficiency**: More efficient than classifier guidance, requiring only one additional forward pass
- **Better Sample Quality**: Significantly improves FID scores and visual quality
- **Tunable Control**: Allows dynamic adjustment of conditioning strength at inference time
- **Broad Applicability**: Works across various conditioning types (text, images, class labels, etc.)

## Applications in Different Domains

CFG has been successfully applied across various domains:

### Text-to-Image Generation

- **Stable Diffusion**: Widely uses CFG with scales around 7.5-9.0
- **DALL-E 2**: Employs CFG to improve text alignment
- **Imagen**: Uses high guidance scales (up to 20) for better text adherence

### Class-Conditional Generation

- **Improved sample quality**: Better FID scores on ImageNet class-conditional generation
- **Style control**: Stronger adherence to style labels

### Image-to-Image Translation

- **Inpainting**: Better adherence to context in image completion
- **Super-resolution**: Improved fidelity to low-resolution inputs
- **Editing**: More precise control in image editing tasks

## Practical Considerations

When implementing CFG, several practical aspects should be considered:

### Null Conditioning

The choice of null conditioning representation is important:
- For text conditioning: Empty string, special token, or random drop
- For class conditioning: Special null class or averaging all classes
- For image conditioning: Zero image or noise

### Computational Trade-offs

CFG requires two forward passes instead of one:
- Approximately doubles the computation time per step
- Memory requirements increase due to storing both outputs
- Can be optimized by sharing computation between conditional and unconditional paths

### Guidance Scale Selection

The optimal guidance scale depends on the task and model:
- **Text-to-image**: Typically 5-10 for good results
- **Class-conditional**: Often lower, around 3-7
- **Image-to-image**: Can be lower (1.5-3) as the conditioning is stronger

## Variations and Extensions

Several variations of CFG have been proposed:

### Dynamic Guidance

- **Noise-dependent guidance**: Different scales at different noise levels
- **Content-dependent guidance**: Adapting guidance based on content

### Multi-guidance Approaches

- **Multiple condition types**: Combining text and image guidance
- **Compositional guidance**: Different guidance for different aspects of generation

### Negative Prompting

- Using negative conditioning with CFG to avoid certain attributes
- Implemented as a form of additional guidance

## Limitations and Challenges

Despite its effectiveness, CFG has some limitations:

- **Quality-Diversity Trade-off**: Stronger guidance reduces sample diversity
- **Saturation Effects**: Extremely high guidance scales can cause artifacts
- **Training Complexity**: Requires careful balancing of conditional/unconditional training
- **Conditioning Strength Variation**: Different models may require different guidance scales

## Implementation Example (Pseudocode)

```python
class CFGDiffusionModel:
    def __init__(self, model, guidance_scale=7.5):
        self.model = model
        self.guidance_scale = guidance_scale
    
    def denoise_step(self, x_t, t, condition):
        # Get unconditional prediction
        unconditional_pred = self.model(x_t, t, condition=None)
        
        # Get conditional prediction
        conditional_pred = self.model(x_t, t, condition=condition)
        
        # Apply classifier-free guidance
        guided_pred = unconditional_pred + self.guidance_scale * (conditional_pred - unconditional_pred)
        
        return guided_pred
    
    def sample(self, condition, steps=50):
        # Start with random noise
        x_t = torch.randn(1, 3, 64, 64)
        
        # Reverse diffusion process with guidance
        for t in reversed(range(steps)):
            noise_pred = self.denoise_step(x_t, t, condition)
            x_t = self.update_x(x_t, noise_pred, t)
            
        return x_t
```

## References

1. Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.
2. Nichol, A., & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
3. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.
4. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.

## Related Topics

- [[Conditional Diffusion Models]]
- [[Text to Image]]
- [[Latent Diffusion Models]]
- [[ControlNet]]
- [[Negative Prompting]] 