# SEARCH
SEARCH: Symmetric Encryption Algorithm Research

## Overview

The `DifferentialAnalyzer` Python library is designed for cryptanalysts to determine optimal differential characteristics in various contexts. The class provides several methods for differential analysis, focusing on characteristics based on given constraints. 

## Key Methods

The following methods are essential for differential analysis:

### `find_optimal_characteristic()`

This method determines the optimal differential characteristic based on standard constraints of the algorithm. It considers only permutation, substitution, and the differential distribution table (DDT).

**Signature:**

```python
find_optimal_characteristic(self, min_blocks=1, max_blocks=1, prune_level=-1, repeat=1, excluded_inputs_from_pruning=None, print_paths=True)
```

**Parameters:**

- `min_blocks`: Minimum number of active S-boxes in the final result.
- `max_blocks`: Maximum number of active S-boxes in the final result.
- `prune_level`: Pruning threshold for the DDT. If set to -1, the solver initializes with the best pruning threshold.
- `repeat`: Number of solutions to find. If not specified, only one solution is found.
- `excluded_inputs_from_pruning`: List of input values to exclude from pruning.
- `print_paths`: Whether to print intermediate solutions. Default is `True`.

### `find_optimal_characteristic_with_fixed_delta_in()`

This method finds the optimal differential characteristic with a fixed input difference.

**Signature:**

```python
find_optimal_characteristic_with_fixed_delta_in(self, delta_in, min_blocks=1, max_blocks=1, prune_level=-1, repeat=1, print_paths=True)
```

**Parameters:**

- `delta_in`: Desired input difference (integer).
- `min_blocks`, `max_blocks`, `prune_level`, `repeat`, `print_paths`: Same as `find_optimal_characteristic`.

### `find_optimal_characteristic_with_fixed_delta_out()`

This method finds the optimal differential characteristic with a fixed output difference.

**Signature:**

```python
find_optimal_characteristic_with_fixed_delta_out(self, delta_out, min_blocks=1, max_blocks=1, prune_level=-1, repeat=1, print_paths=True)
```

**Parameters:**

- `delta_out`: Desired output difference (integer).
- `min_blocks`, `max_blocks`, `prune_level`, `repeat`, `print_paths`: Same as `find_optimal_characteristic`.

### `find_optimal_characteristic_iterative()`

This method finds the optimal differential characteristic iteratively, where the input difference equals the output difference.

**Signature:**

```python
find_optimal_characteristic_iterative(self, min_blocks=1, max_blocks=1, prune_level=-1, repeat=2, print_paths=True)
```

**Parameters:**

- `min_blocks`, `max_blocks`, `prune_level`, `repeat`, `print_paths`: Same as `find_optimal_characteristic`.

### `find_best_differential()`

This method improves an already found differential characteristic by finding alternative paths.

**Signature:**

```python
find_best_differential(self, delta_in, delta_out, repeat=1, print_paths=True)
```

**Parameters:**

- `delta_in`: Fixed input difference (integer).
- `delta_out`: Fixed output difference (integer).
- `repeat`: Number of solutions to find. Default is set to `2`.
- `print_paths`: Whether to print intermediate solutions. Default is `True`.

## Usage Example

  You can take as an example main.py, rectangle.py, gift.py or present.py


Ensure you have the necessary dependencies installed and properly configured. Adjust the S-box, P-box, and number of rounds as per your requirements for differential analysis.
