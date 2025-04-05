# ML Compiler Architectures

Overview of major ML compiler frameworks and their architectural designs.

## MLIR (Multi-Level Intermediate Representation)
- **Core Concepts**
  - Dialect-based design
  - Progressive lowering
  - Extensible framework
- **Components**
  - [[MLIR Dialects]]
  - [[MLIR Passes]]
  - [[MLIR Infrastructure]]

## XLA (Accelerated Linear Algebra)
- **Features**
  - JIT compilation
  - Platform-specific optimization
  - TensorFlow integration
- **Key Components**
  - [[XLA HLO]] (High Level Optimizer)
  - [[XLA Backend Integration]]
  - [[XLA Optimization Passes]]

## Other Frameworks
- **TVM (Tensor Virtual Machine)**
  - [[TVM Stack]]
  - Automated optimization
  - Hardware targets
- **Glow**
  - Facebook's ML compiler
  - Quantization support
  - Neural network focus

## Common Architecture Patterns
- **IR Hierarchy**
  - High-level representation
  - Mid-level optimization
  - Low-level code generation
- **Optimization Passes**
  - [[Common Compiler Optimizations]]
  - ML-specific transformations
  - Target-specific passes

## Related Topics
- [[ML Fundamental Concepts]] - Mathematical foundations
- [[Parallel Computing ML]] - Hardware optimization
- [[ML Compiler Tools]] - Development tools

## Implementation Considerations
- [[Compiler Pipeline Design]]
- [[Optimization Strategy]]
- [[Backend Integration]]

---
Tags: #ml-compilers #architecture #frameworks #infrastructure 