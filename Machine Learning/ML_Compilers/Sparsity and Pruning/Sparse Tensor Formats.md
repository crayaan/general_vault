---
aliases:
  - Sparse Data Structures
  - Sparse Matrix Formats
---

# Sparse Tensor Formats

When dealing with [[Sparsity and Pruning Overview|sparse networks]], where many parameters (weights) are zero, using standard dense tensor representations (like multi-dimensional arrays) is inefficient in terms of both memory storage and computation. Sparse tensor formats are specialized data structures designed to store only the non-zero elements and their locations, leading to significant savings.

## Motivation

- **Memory Efficiency**: Avoid storing large numbers of zero values.
- **Computational Efficiency**: Skip unnecessary computations involving zeros (e.g., multiplication by zero).

## Common Sparse Matrix Formats (Often extended to Tensors)

These formats are commonly described for 2D matrices but can be generalized or adapted for higher-dimensional tensors.

### 1. Coordinate List (COO)

- **Description**: Stores a list of tuples, where each tuple contains `(row_index, column_index, value)` for every non-zero element.
- **Structure**: Typically three arrays: one for row indices, one for column indices, and one for the non-zero values.
- **Example**:
  ```
  Matrix:
  [[5, 0, 0],
   [0, 0, 2],
   [1, 0, 0]]

  COO:
  rows = [0, 1, 2]
  cols = [0, 2, 0]
  vals = [5, 2, 1]
  ```
- **Pros**: Simple to construct, easy to add elements incrementally.
- **Cons**: Inefficient for matrix operations (like multiplication or row slicing) due to unsorted indices and potential random memory access.

### 2. Dictionary of Keys (DOK)

- **Description**: Uses a dictionary (hash map) where keys are tuples `(row_index, column_index)` and values are the corresponding non-zero elements.
- **Structure**: Hash map.
- **Example**:
  ```
  Matrix:
  [[5, 0, 0],
   [0, 0, 2],
   [1, 0, 0]]

  DOK:
  {(0, 0): 5,
   (1, 2): 2,
   (2, 0): 1}
  ```
- **Pros**: Very efficient for incremental construction and element lookup/modification.
- **Cons**: High memory overhead due to dictionary structure, generally inefficient for matrix arithmetic operations.

### 3. List of Lists (LIL)

- **Description**: Uses a list of lists. The outer list represents rows. Each inner list stores tuples `(column_index, value)` for the non-zero elements in that row.
- **Structure**: List of lists of tuples.
- **Example**:
  ```
  Matrix:
  [[5, 0, 0],
   [0, 0, 2],
   [1, 0, 0]]

  LIL:
  [ [(0, 5)],            # Row 0
    [(2, 2)],            # Row 1
    [(0, 1)] ]           # Row 2
  ```
- **Pros**: Efficient for row slicing and incremental construction.
- **Cons**: Can be memory-inefficient compared to CSR/CSC for arithmetic, column slicing is slow.

### 4. Compressed Sparse Row (CSR / CRS)

- **Description**: Compresses the row indices of the COO format. It uses three arrays:
    - `values`: Stores the non-zero values, ordered row by row.
    - `col_indices`: Stores the column index for each corresponding value in `values`.
    - `row_pointer`: Stores the index in `values` where each row *starts*. Length is `num_rows + 1`, with the last element being the total number of non-zero elements (`nnz`).
- **Structure**: Three arrays.
- **Example**:
  ```
  Matrix:
  [[5, 0, 0],
   [0, 0, 2],
   [1, 0, 0]]

  CSR:
  values      = [5, 2, 1]
  col_indices = [0, 2, 0]
  row_pointer = [0, 1, 2, 3]  # Row 0 starts at index 0, Row 1 at 1, Row 2 at 2, ends at 3 (nnz)
  ```
- **Pros**: Memory efficient, very fast row slicing, efficient matrix-vector multiplication and other row-oriented operations.
- **Cons**: Slow column slicing, slower incremental construction than COO/LIL/DOK.

### 5. Compressed Sparse Column (CSC / CCS)

- **Description**: The transpose of CSR format. Compresses the column indices.
    - `values`: Stores non-zero values, ordered column by column.
    - `row_indices`: Stores the row index for each value.
    - `col_pointer`: Stores the index in `values` where each column starts.
- **Structure**: Three arrays.
- **Example**:
  ```
  Matrix:
  [[5, 0, 0],
   [0, 0, 2],
   [1, 0, 0]]

  CSC:
  values      = [5, 1, 2]
  row_indices = [0, 2, 1]
  col_pointer = [0, 2, 2, 3] # Col 0 starts at 0, Col 1 at 2, Col 2 at 2, ends at 3 (nnz)
  ```
- **Pros**: Memory efficient, very fast column slicing, efficient matrix multiplication (especially A*B where A is CSC).
- **Cons**: Slow row slicing, slower incremental construction.

## Higher-Dimensional Tensors

Representing sparsity in tensors with more than two dimensions is more complex. Common approaches include:

- **Generalizations of COO/CSR/CSC**: Extending the index arrays.
- **Hierarchical Formats**: Like Block Compressed Sparse Row (BCSR) or formats that exploit tensor structures (e.g., Tucker, Tensor Train decompositions combined with sparsity).
- **Fiber-Based Formats (e.g., TACO)**: Represent tensors as hierarchies of sparse/dense fibers (vectors/arrays). [\[1\]](https://arxiv.org/abs/2311.09549)

## Choosing a Format

The best format depends on:
- **The specific operation**: Matrix-vector products favor CSR, matrix transpositions favor CSC.
- **Construction method**: Incremental updates favor DOK/LIL.
- **Sparsity pattern**: Some formats are better for specific structures.
- **Hardware/Library support**: Optimized libraries (e.g., cuSPARSE, Intel MKL) often perform best with CSR/CSC.

In deep learning inference, formats like CSR or specialized blocked formats are often preferred for efficient sparse matrix multiplications on GPUs/CPUs.

---

**References**:
1. Dias, A., Anderson, L., Sundararajah, K., Pelenitsyn, A., & Kulkarni, M. (2024). SparseAuto: An Auto-Scheduler for Sparse Tensor Computations Using Recursive Loop Nest Restructuring. *arXiv preprint arXiv:2311.09549*. [https://arxiv.org/abs/2311.09549](https://arxiv.org/abs/2311.09549) 