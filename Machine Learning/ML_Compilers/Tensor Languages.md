# Tensor Languages

Programming languages and domain-specific languages (DSLs) designed for expressing tensor computations in machine learning.

## Tensor Language Categories

### High-Level ML Frameworks
- **PyTorch**
  - Eager execution model
  - Dynamic computation graphs
  - Python-based tensor operations
  - TorchScript for graph capture
- **TensorFlow**
  - Static computation graphs (TF1.x)
  - Eager execution support (TF2.x)
  - Python frontend with C++ backend
  - SavedModel and Graph formats

### ML-Specific DSLs
- **TensorIR**
  - Tensor expression abstraction
  - Schedule primitives
  - Auto-scheduling capabilities
  - Part of Apache TVM stack
- **Halide**
  - Separation of algorithm and schedule
  - Image processing optimizations
  - Stencil computation patterns
  - Influences on ML compilers
- **TACO (Tensor Algebra Compiler)**
  - Sparse tensor algebra
  - Format-aware computation
  - Index notation abstractions

### Embedded IR Languages
- **MLIR Tensor Dialect**
  - Tensor operations representation
  - Progressive lowering approach
  - Multi-level optimization capabilities
  - Integration with other MLIR dialects
- **XLA HLO**
  - High-level operations for linear algebra
  - TensorFlow/JAX integration
  - Platform-specific lowering
  - Operation fusion capabilities

## Language Design Principles

### Abstraction Levels
- **Declarative vs. Imperative**
  - Expression of what to compute vs. how
  - Optimization opportunities in declarative models
  - Ease of use in imperative approaches
- **Static vs. Dynamic**
  - Compile-time vs. runtime decisions
  - Shape inference capabilities
  - Trade-offs between flexibility and performance

### Tensor Expression Patterns
- **Einstein Notation**
  - Implicit summation over repeated indices
  - Concise representation of complex operations
  - Translation to explicit loops
- **Linear Algebra Operations**
  - Matrix multiplication, convolution, etc.
  - High-level operation composition
  - Algebraic optimization opportunities

### Scheduling and Execution Models
- **Scheduling Languages**
  - Explicit schedule descriptions
  - Loop transformation primitives
  - Memory hierarchy mapping
- **Execution Strategies**
  - Lazy vs. eager execution
  - Operator fusion approaches
  - Memory allocation patterns

## Implementation Features

### Type Systems
- **Shape Typing**
  - Static shape constraints
  - Dynamic shape handling
  - Shape inference mechanisms
- **Data Type Handling**
  - Numeric precision specification
  - Custom data types
  - Type inference and conversion

### Optimization Hooks
- **Transformation Interfaces**
  - Pattern-based rewrites
  - Algebraic optimizations
  - Target-specific transformations
- **Auto-Tuning Integration**
  - Cost model interfaces
  - Search space definition
  - Performance feedback loops

### Hardware Abstractions
- **Target Specification**
  - Generic vs. specialized hardware descriptions
  - Capability-based abstractions
  - Backend code generation

## Related Topics
- [[Tensor Programming]] - General tensor programming concepts
- [[ML Compiler Frameworks]] - Compiler frameworks using tensor languages
- [[MLIR Framework]] - Multi-level IR implementation
- [[Auto-Scheduling]] - Automated tensor program optimization

## Next Steps
→ Explore [[ML Compiler Frameworks]] to understand how tensor languages are integrated into compilers

---
Tags: #ml-compilers #tensor-languages #dsl #programming-languages 