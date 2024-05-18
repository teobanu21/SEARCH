from collections import Counter
from z3 import *


def calculate_difference_distribution_table(sbox: list) -> Counter:
    """
    This method is used to calculate the difference distribution table of a given S-box

    Args:
        sbox (list)

    Returns:
        Counter: A Counter dictionary that keeps the Difference Distribution Table

    Note: Counter is faster for counting the repetitions of an object than any other method of implementing this task
    """
    ddt = Counter()
    n = len(sbox)
    for input_diff in range(n):
        for plaintext in range(n):
            output_diff = sbox[plaintext] ^ sbox[plaintext ^ input_diff]
            ddt[(input_diff, output_diff)] += 1

    return ddt

def pretty_print_ddt(counter: Counter) -> None:
    max_x = max(coord[0] for coord in counter.keys()) + 1
    max_y = max(coord[1] for coord in counter.keys()) + 1
    matrix = [[0] * max_y for _ in range(max_x)]
    for coord, value in counter.items():
        x, y = coord
        matrix[x][y] = value
    for row in matrix:
        print(row)

def formatted_bits(value, block_size, sub_block_size):
    binary_value = bin(value)[2:].zfill(block_size)
    return "|".join(
        binary_value[i : i + sub_block_size]
        for i in range(0, block_size, sub_block_size)
    )


def all_smt(solver, initial_terms):
    """
    Generate all satisfying models for a given set of initial terms using an SMT solver.

    Parameters
    ----------
    solver : z3.Solver 
        An instance of an SMT solver.

    initial_terms : list
        A list of initial terms to generate satisfying models for.

    Yields
    ------
    model : Model
        A satisfying model for the set of initial terms.

    Notes
    -----
    This function uses a recursive approach to generate all satisfying models for a given set of initial terms.
    The `all_smt_rec` function is a recursive helper function that performs the actual generation of satisfying models.
    """

    def all_smt_rec(terms):
        """
        Recursive helper function to generate all satisfying models for a given set of terms.

        Parameters
        ----------
        terms : list
            A list of terms to generate satisfying models for.

        Yields
        ------
        model : Model
            A satisfying model for the set of terms.
        """
        if solver.check() == sat:
            model = solver.model()
            yield model
            for i in range(len(terms)):
                solver.push()
                solver.add(terms[i] != model.eval(terms[i]))
                for j in range(i):
                    solver.add(terms[j] == model.eval(terms[j]))
                yield from all_smt_rec(terms[i:])
                solver.pop()

    yield from all_smt_rec(list(initial_terms))
