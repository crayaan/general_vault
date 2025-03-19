# Systolic Arrays in ML Hardware

Systolic arrays represent a specialized processing architecture that has become foundational for modern machine learning hardware accelerators, most notably in Google's Tensor Processing Units (TPUs).

## Foundational Concepts

A systolic array is an arrangement of processing elements (PEs) or cells that process data in a rhythmic, wave-like fashion - similar to how the heart pumps blood through the body (hence the term "systolic"). In this architecture:

- Data flows through interconnected processing elements in a pipelined manner
- Each PE performs a specific operation (typically multiply-accumulate)
- Data is reused multiple times before returning to memory
- Processing happens in waves or pulses through the array

This design addresses one of the major computing bottlenecks in modern machine learning: **memory access energy consumption**. Research has shown that a DRAM memory access consumes 100-1000X more energy than a complex arithmetic operation. Systolic arrays significantly reduce this overhead by allowing multiple computations to be performed for each memory access.

## Visual Understanding

To understand how a systolic array works, consider this simplified example of a 3×3 systolic array performing matrix multiplication:

```
    a11   a12   a13
     ↓     ↓     ↓
b11→[PE]→[PE]→[PE]
     ↓     ↓     ↓
b21→[PE]→[PE]→[PE]
     ↓     ↓     ↓
b31→[PE]→[PE]→[PE]
     ↓     ↓     ↓
    c11   c12   c13
```

In this diagram:
- Weights (a values) flow down from the top
- Input data (b values) flows from the left
- Output results (c values) emerge at the bottom

The operation proceeds in a wave-like pattern:

**Step 1**: Values a11 and b11 enter the top-left PE, where they are multiplied together.
```
    a11    -     -
     ↓     
b11→[PE]→  -  →  -
     ↓     
 -   -     -     -
     
 -   -     -     -
```

**Step 2**: a11 moves down, b11 moves right, and a12 and b21 enter the array:
```
    a12   a11    -
     ↓     ↓     
b21→[PE]→[PE]→  -  
     ↓     ↓     
 -   -     -     -
     
 -   -     -     -
```

**Step 3**: The wave continues, with partial results accumulating in each PE:
```
    a13   a12   a11
     ↓     ↓     ↓
b31→[PE]→[PE]→[PE]
     ↓     ↓     ↓
b21→[PE]→[PE]→  -  
     ↓
 -   -     -     -
```

As the computation progresses, each PE multiplies and accumulates values, and final results emerge diagonally. This approach minimizes memory access by reusing data across multiple PEs.

## Data Flow Types

Systolic arrays can be configured with different data flow strategies:

1. **Weight Stationary (WS)**: Weights remain fixed in PEs while inputs and partial sums flow through
2. **Input Stationary (IS)**: Input values remain fixed while weights and partial results move
3. **Output Stationary (OS)**: Each PE accumulates its output while weights and inputs are streamed in

The choice of data flow pattern depends on the matrix dimensions and optimization goals.

## Architecture and Operation

In a typical systolic array for matrix multiplication:

1. Input data enters from the left side of the array
2. Weights are loaded from the top
3. Partial results flow through the array diagonally
4. Final results emerge from the bottom or right edges

Each processing element:
- Receives inputs from its neighbors
- Performs a computation (typically multiply-accumulate)
- Passes results to adjacent cells
- May store intermediate results

The key advantage is that intermediate results are automatically available when needed due to the structure of the array and the order of data flow, eliminating the need to store and fetch these results from main memory.

## Implementation in TPUs

Google's Tensor Processing Unit represents the most prominent commercial implementation of systolic arrays for machine learning. The first-generation TPU featured:

- A 256×256 systolic array (65,536 multiply-accumulate units)
- 8-bit integer matrix operations
- 92 TOPS (Tera Operations Per Second) peak throughput
- 15-30X faster than contemporary GPUs and CPUs
- 30-80X higher TOPS/Watt efficiency

The TPU's Matrix Multiply Unit (MXU) leverages the systolic array architecture to perform highly efficient matrix multiplications and convolutions, which are fundamental operations in neural networks. This approach stands in contrast to:

- **CPUs**: General-purpose scalar processors that execute one operation at a time
- **GPUs**: Vector processors that execute the same operation on multiple data elements
- **TPUs**: Matrix processors that execute hundreds of thousands of operations in a single clock cycle

## Applications in ML Compilers

Systolic arrays are particularly effective for [[ML_Compilers/Hardware-Aware Compilation|hardware-aware compilation]], as they enable optimized execution of common neural network operations:

1. **Matrix Multiplication**: The foundation of fully-connected layers in neural networks
2. **Convolution**: Fundamental for CNNs and image processing
3. **Activation Functions**: Applied efficiently after matrix operations

ML compilers like [[ML_Compilers/Compiler Frameworks|TensorFlow XLA]] and others must generate optimized code specifically for systolic architectures to achieve maximum performance. This requires:

- Efficient mapping of tensor operations to systolic arrays
- Optimal tiling and blocking strategies for large matrices
- Precise data orchestration to maintain the systolic flow
- Quantization techniques to leverage integer arithmetic

## Advantages and Limitations

**Advantages**:
- Extremely high computational density and throughput
- Energy efficiency through data reuse
- Reduced memory bandwidth requirements
- Predictable execution patterns for reliable performance

**Limitations**:
- Less flexible than general-purpose architectures
- Best suited for regular, structured computations
- Requires specialized programming and [[ML_Compilers/Model Optimization Strategies|optimization strategies]]
- May require quantization to int8 or other reduced precision formats

## Future Directions

As machine learning models continue to grow in complexity, systolic arrays are evolving to address new challenges:

1. **Increased size and dimensions**: Larger arrays for higher throughput
2. **Mixed precision support**: Combining different numeric formats
3. **Sparse matrix handling**: Addressing efficiency for sparse neural networks
4. **Reconfigurable topologies**: More flexible data flow patterns
5. **Integration with other accelerator types**: Hybrid architectures

## Relation to ML Compiler Optimization

Systolic arrays have significant implications for [[ML_Compilers/Model Optimization Strategies|model optimization strategies]]. Compiler techniques must consider:

- Operator fusion opportunities to maximize systolic execution
- Data layout transformations for optimal array utilization
- Loop tiling and blocking to match systolic array dimensions
- Memory access patterns that align with systolic data flow

## References

1. "In-Datacenter Performance Analysis of a Tensor Processing Unit" - Google Inc.
2. "An in-depth look at Google's first Tensor Processing Unit (TPU)" - Google Cloud Blog
3. "Challenges for Future Computing Systems" - Bill Dally, NVIDIA
4. "Google Workloads for Consumer Devices: Mitigating Data Movement Bottlenecks" - ASPLOS

#ml-compilers #hardware-acceleration #systolic-arrays #tpu #matrix-multiplication 