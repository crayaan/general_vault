# StableHLO

StableHLO is a portable intermediate representation (IR) for machine learning operations that serves as a key compatibility layer between ML frameworks and compilers within the [[XLA Compiler|OpenXLA]] ecosystem.

## Overview

StableHLO functions as a portable dialect of operations designed to bridge the gap between various ML frameworks (such as TensorFlow, PyTorch, and JAX) and ML compilers (such as XLA and IREE). It provides strong compatibility guarantees that make it ideal for model exchange and long-term deployment scenarios.

Essentially, StableHLO works as a standardized language for representing ML computations that can be:
1. Generated from any supported ML framework
2. Optimized by compatible compilers
3. Deployed across diverse hardware targets
4. Maintained over extended time periods with compatibility assurances

## Key Features

- **Strong Compatibility Guarantees**: 
  - 5 years of backward compatibility
  - 2 years of forward compatibility
  - Versioned serialization format using MLIR bytecode

- **Comprehensive Operation Set**:
  - ~100 fully specified operations
  - Support for both static and dynamic shapes
  - Quantization capabilities
  - Extensibility via custom calls and composite operations

- **Reference Implementation**:
  - High-quality C++ and Python APIs
  - Reference interpreter for validation and testing
  - Compatibility testing infrastructure

- **Standardized Transformations**:
  - Hardware-independent optimizations
  - Shape refinement for dynamic shapes
  - Conversions to other MLIR dialects (linalg, TOSA)

## Architecture

### StableHLO Programs

A StableHLO program consists of:
- **Module**: The top-level container
- **Functions**: Collections of operations with inputs and outputs
- **Operations**: Individual computational steps with defined semantics
- **Types**: Data representations (tensors, scalars, etc.)
- **Attributes**: Compile-time constants and metadata

### Relationship to MLIR

StableHLO is implemented as an [[MLIR Framework|MLIR]] dialect, leveraging the MLIR infrastructure for:
- IR representation and manipulation
- Type system
- Transformation infrastructure
- Serialization (via MLIR bytecode)

### Evolution from MHLO

StableHLO evolved from the MHLO (MLIR HLO) dialect, adding:
- Formal specification
- Versioning mechanism
- Compatibility guarantees
- Reference interpreter
- Enhanced dynamism support

## Integration with ML Frameworks

### TensorFlow
- TensorFlow graphs can be exported to StableHLO
- Used as part of the XLA compilation path
- Enables deployment to various XLA backends

### PyTorch
- PyTorch models can be exported to StableHLO via torch.export
- Enables XLA-based acceleration for PyTorch
- Supports distributed training on TPUs and other accelerators

### JAX
- StableHLO is the primary IR used by JAX
- JAX functions can be JIT-compiled to StableHLO
- Enables hardware acceleration across JAX's supported platforms

## Compatibility Mechanisms

StableHLO's compatibility guarantees are implemented through:

### Versioning
- Semantic versioning scheme (MAJOR.MINOR.PATCH)
- Minor version bumps for any changes to the operation set or serialization format
- Patch version bumps for bug fixes and downstream integrations

### VHLO Dialect
- Internal versioned representation for serialization/deserialization
- Tracks operation semantics across versions
- Enables converting between different StableHLO versions

### Portable Artifacts
- Serialized programs using MLIR bytecode
- Include version information
- Support translation between compatible versions

## Use Cases

- **Model Exchange**: Enabling different frameworks to share models
- **Compiler Development**: Providing a stable target for ML compiler backends
- **Long-term Deployment**: Ensuring models remain functional over time
- **Hardware Targeting**: Allowing models to be compiled for diverse hardware
- **ML Ecosystem Interoperability**: Creating bridges between ML frameworks

## Relationship to Other ML Compiler Projects

- **XLA**: StableHLO serves as XLA's portable model format
- **IREE**: Uses StableHLO as an input format for its compilation pipeline
- **MLIR**: Built on MLIR infrastructure and tooling
- **TVM**: Serves a similar role but with different design choices and ecosystem

## Next Steps
→ Learn about the [[XLA Compiler]] which uses StableHLO as its portable format
→ Explore [[MLIR Framework]] which provides the infrastructure for StableHLO
→ See [[Model Optimization Strategies]] for how StableHLO programs can be optimized

---
Tags: #ml-compilers #stablehlo #intermediate-representation #mlir #xla 