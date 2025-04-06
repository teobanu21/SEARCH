from collections import Counter
from z3 import *
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    def all_smt_rec(terms):
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



def parity(x):
    """Calculates the parity of an integer.

    This method calculates the parity of an integer by counting the number
    of set bits in the binary representation of the integer. It returns 0 if the
    number of set bits is even, and 1 otherwise.

    Args:
        x (int): The input value for which the parity is calculated.

    Returns:
        int: 0 if the number of set bits is even, 1 otherwise.
    """
    res = 0
    while x:
        res ^= 1
        x &= (x - 1)
    return res


def calculate_linear_bias(sbox, no_sign=True, fraction=False):
    """Calculates the linear bias of an S-box.

    This method calculates the linear bias of an S-box. It iterates over
    all possible input and output mask pairs and computes the linear bias using
    the Cryptanalysis.parity method.

    Args:
        sbox (list): A list of integers representing the S-box.
        no_sign (bool, optional): If True, the absolute value of the bias is returned. Defaults to True.
        fraction (bool, optional): If True, the bias is returned as a fraction. Defaults to False.

    Returns:
        Counter: A Counter dictionary containing the linear biases for each input and output mask pair.
    """
    n = len(sbox)
    bias = Counter({(i, j): -(n // 2) for i in range(n) for j in range(n)})
    for imask in range(n):
        for omask in range(n):
            for i in range(n):
                bias[(imask, omask)] += parity((sbox[i] & omask) ^ (i & imask)) ^ 1
    if no_sign:
        for i in bias:
            bias[i] = abs(bias[i])
    if fraction:
        for i in bias:
            bias[i] /= n
    return bias


