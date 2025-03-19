# PJRT (Portable JAX Runtime)

PJRT is an open, stable interface for device runtime and compiler operations within the [[XLA Compiler|OpenXLA]] ecosystem. It serves as a hardware-agnostic abstraction layer that simplifies ML hardware and framework integration.

## Overview

PJRT was designed to address the growing complexity of executing ML workloads across diverse hardware platforms and frameworks. It provides a uniform device API that enables ML frameworks to interact with different hardware accelerators through a consistent interface, making hardware accelerators "pluggable" and frameworks "hardware-agnostic."

The core vision of PJRT is two-fold:
1. Frameworks (TensorFlow, JAX, PyTorch) call PJRT through a standardized API
2. Hardware vendors implement PJRT plugins that provide device-specific functionality

This separation of concerns significantly reduces integration complexity and enables faster adoption of new hardware accelerators by ML frameworks.

## Architecture

### Core Components

PJRT consists of several key components:

- **Client**: Represents a connection to one or more devices of the same type
- **Device**: Represents a single accelerator (e.g., GPU, TPU, CPU)
- **Buffer**: Represents data allocated on a device
- **Executable**: Represents a compiled program that can be executed on devices

### API Layers

PJRT provides multiple API layers to accommodate different integration needs:

1. **PJRT C API**: The foundational layer providing ABI stability
2. **PJRT C++ API**: A higher-level C++ interface built on top of the C API
3. **Python Bindings**: Python wrappers for framework integration

### Plugin System

The PJRT plugin system allows hardware vendors to implement the PJRT interface for their specific hardware:

- Each plugin implements the core PJRT methods
- Plugins are typically distributed as Python packages
- Frameworks discover and load plugins at runtime
- Extensions mechanism allows for optional or experimental features

## Key Features

### Cross-Framework Support

PJRT serves as:
- The only interface for JAX
- The primary interface for TensorFlow
- A fully supported interface for PyTorch through PyTorch/XLA

### Hardware Portability

PJRT has been implemented for various hardware platforms:
- NVIDIA GPUs
- Google Cloud TPUs
- Apple Silicon (through Metal)
- Intel Max GPUs
- CPUs (reference implementation)

### Multi-Device & Distributed Execution

PJRT supports:
- Managing multiple devices within a single node
- Cross-device operations and memory transfers
- Multi-node execution through key-value store coordination
- Distributed training across device clusters

### Versioning and Compatibility

PJRT maintains compatibility through:
- Major and minor versioning
- Forward compatibility window (6 weeks) for minor version updates
- Coordinated updates for major version changes
- Extension mechanism for optional or experimental features

## Integration with XLA

### Compilation Flow

The typical compilation and execution flow with PJRT:

1. Framework (JAX/TensorFlow/PyTorch) generates a model representation
2. Model is lowered to [[StableHLO]]
3. StableHLO representation is passed to PJRT for compilation
4. PJRT compiles the program for the target hardware
5. Framework executes the compiled program through PJRT

### Memory Management

PJRT handles device memory operations:
- Buffer allocation and deallocation
- Host-to-device transfers
- Device-to-host transfers
- Device-to-device transfers
- Buffer donation for memory optimization

### DLPack Support

PJRT provides integration with the DLPack standard:
- Creating PJRT buffers from DLPack tensors
- Exporting PJRT buffers to DLPack format
- Reference counting for safe memory management

## Implementing a PJRT Plugin

To create a new PJRT plugin, hardware vendors must implement:

1. Core PJRT C API functions or C++ API with wrapper
2. `GetPjRtApi()` function that returns function pointers to implementations
3. Registration mechanisms for framework discovery
4. Essential functionality like compilation, execution, and memory management

The plugin typically includes:
- Device detection and initialization
- Compiler that translates StableHLO to hardware-specific code
- Runtime that manages execution and device resources
- Memory management for device buffers

## Real-World Examples

### Apple Silicon Implementation

Apple's PJRT plugin for Metal:
- Accelerates JAX models on Apple Silicon and AMD GPUs
- Converts StableHLO to MPSGraph executables
- Uses Metal runtime APIs to dispatch operations to the GPU
- Provides performance comparable to native Metal implementations

### Google Cloud TPU

The TPU PJRT implementation:
- Manages TPU device topology
- Handles distributed TPU execution
- Optimizes compilation for TPU architecture
- Supports multi-chip and multi-host TPU configurations

## Relation to Other ML Compiler Components

- **[[StableHLO]]**: Serves as the input format for PJRT compilation
- **[[XLA Compiler]]**: PJRT serves as XLA's runtime interface
- **[[MLIR Framework]]**: Used in the compilation pipeline for many PJRT plugins
- **[[Hardware-Aware Compilation]]**: PJRT plugins implement hardware-specific optimizations

## Next Steps
→ Learn about the [[XLA Compiler]] which uses PJRT as its runtime interface
→ Explore [[StableHLO]] which provides the portable format for PJRT compilation
→ See [[Hardware-Aware Compilation]] for hardware-specific optimizations

---
Tags: #ml-compilers #pjrt #runtime #xla #distributed-training 