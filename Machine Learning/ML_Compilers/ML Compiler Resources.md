# ML Compiler Resources

Comprehensive collection of research papers, courses, tutorials, and community resources for learning about ML compilers.

## Research Papers

### Foundational Papers
- **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning**
  - Chen et al., OSDI 2018
  - Introduces the TVM stack
  - End-to-end compilation and optimization
  - Auto-tuning approach
- **XLA: Optimizing Compiler for Machine Learning**
  - Google
  - TensorFlow's compiler architecture
  - Just-in-time compilation
  - Operator fusion and buffer analysis
- **MLIR: A Compiler Infrastructure for the End of Moore's Law**
  - Lattner et al., 2020
  - Multi-level IR design
  - Dialect-based extensibility
  - Progressive lowering approach

### Optimization Techniques
- **Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions**
  - Vasilache et al., 2018
  - Polyhedral compilation
  - Auto-tuning for GPUs
  - Framework integration
- **Equality Saturation: A New Approach to Optimization**
  - Tate et al., POPL 2009
  - E-graph based optimization
  - Rewrite rule application
  - Influence on ML compiler design
- **Automatic Generation of Peephole Superoptimizers**
  - Bansal et al., ASPLOS 2006
  - Search-based optimization
  - Machine learning for compiler optimization
  - Automated rule discovery

### Hardware Acceleration
- **In-Datacenter Performance Analysis of a Tensor Processing Unit**
  - Jouppi et al., ISCA 2017
  - TPU architecture and design
  - Workload analysis
  - Performance comparisons
- **Efficiently Scaling Transformer Inference**
  - Pope et al., MLSys 2022
  - Compiler techniques for transformers
  - Memory optimization
  - Hardware-specific adaptation

## Online Courses and Tutorials

### University Courses
- **Stanford CS217: Hardware Accelerators for Machine Learning**
  - Accelerator architectures
  - Compiler techniques
  - Memory systems
  - [Course Website](https://cs217.stanford.edu/)
- **UC Berkeley CS294: ML Systems**
  - System design for ML
  - Compiler optimization
  - Hardware/software co-design
  - [Course Resources](https://ucbrise.github.io/cs294-ai-sys-sp19/)

### Industry Tutorials
- **MLIR Tutorials**
  - Getting started with MLIR
  - Creating custom dialects
  - Implementing passes
  - [MLIR Documentation](https://mlir.llvm.org/docs/Tutorials/)
- **TVM Tutorials**
  - TVM compiler basics
  - Auto-tuning and scheduling
  - Deployment workflows
  - [TVM Documentation](https://tvm.apache.org/docs/tutorial/)
- **XLA Overview and Tutorials**
  - XLA concepts
  - Integration with TensorFlow
  - Custom operations
  - [XLA Documentation](https://www.tensorflow.org/xla)

### Video Resources
- **Conference Recordings**
  - LLVM Developer Meetings
  - Compiler Design for ML workshops
  - MLSys conference presentations
  - [LLVM YouTube Channel](https://www.youtube.com/channel/UCv2_41bSAa5Y_8BacJUZfjQ)
- **Online Lectures**
  - Deep learning systems design
  - Hardware acceleration
  - Compiler optimization techniques

## Community and Forums

### Discussion Forums
- **LLVM Discussion Groups**
  - MLIR developer mailing list
  - LLVM compiler infrastructure
  - Target-specific optimization
  - [LLVM Discourse](https://llvm.discourse.group/)
- **TVM Community**
  - Development discussions
  - User questions
  - Hardware support
  - [TVM Discuss](https://discuss.tvm.apache.org/)

### GitHub Repositories
- **Framework Repositories**
  - [MLIR GitHub](https://github.com/llvm/llvm-project/tree/main/mlir)
  - [TVM GitHub](https://github.com/apache/tvm)
  - [XLA GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla)
- **Example Projects**
  - Compiler extensions
  - Custom optimizations
  - Hardware backend implementations

### Conferences and Workshops
- **Major Conferences**
  - MLSys (Machine Learning and Systems)
  - ASPLOS (Architectural Support for Programming Languages and Operating Systems)
  - PLDI (Programming Language Design and Implementation)
  - OSDI (Operating Systems Design and Implementation)
- **Specialized Workshops**
  - Compilers for Machine Learning (C4ML)
  - Machine Learning and Programming Languages (MAPL)
  - AI Systems workshops

## Books and Long-Form Resources

### Textbooks
- **The LLVM Compiler Infrastructure**
  - Compiler design principles
  - Pass infrastructure
  - Backend design
- **Optimizing Compilers for Modern Architectures**
  - Morgan, K. & Muchnick, S.
  - Loop optimization
  - Instruction selection
  - Register allocation

### Blogs and Articles
- **Company Engineering Blogs**
  - Google AI Blog
  - NVIDIA Developer Blog
  - Meta AI Research Blog
- **Personal Technical Blogs**
  - Experts in compiler design
  - ML systems researchers
  - Hardware acceleration specialists

## Related Topics
- [[Compiler Construction Projects]] - Practical applications
- [[ML Compiler Tools]] - Development and deployment tools
- [[Learning Resources]] - General learning path
- [[Research Papers]] - In-depth paper analysis

## Next Steps
→ Apply your knowledge in a practical [[Compiler Construction Projects]] project

---
Tags: #ml-compilers #resources #research-papers #courses #community 