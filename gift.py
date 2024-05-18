import DifferentialAnalyzer as DifferentialAnalyzer
from utils import pretty_print_ddt


def main():
    sbox = {
        0: 0x1,
        1: 0xA,
        2: 0x4,
        3: 0xC,
        4: 0x6,
        5: 0xF,
        6: 0x3,
        7: 0x9,
        8: 0x2,
        9: 0xD,
        0xA: 0xB,
        0xB: 0x7,
        0xC: 0x5,
        0xD: 0x0,
        0xE: 0x8,
        0xF: 0xE,
    }
    sbox_list = [sbox[key] for key in range(len(sbox))]

    pbox = [0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3, 4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
            8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11, 12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15]

    num_rounds = 1  # 28 rounds

    solver = DifferentialAnalyzer.DifferentialAnalyzer(
        sbox_list, pbox, num_rounds, "gift/1_cont_12.txt"
    )

    # pretty_print_ddt(solver.ddt)

    # optimal characteristic
    # char = solver.find_optimal_characteristic(
    #     min_blocks=1, max_blocks=2, prune_level=1, repeat=1, print_paths=True # excluded_inputs_from_pruning=[1, 3, 9, 11],
    # )

    # print(char)

    # iterative = solver.find_optimal_characteristic_iterative(min_blocks=3, max_blocks=5, prune_level=1, repeat=1, print_paths=True)
    # print(iterative)

    # delta_in =int('0000000000000000010100000000000000000000000000000101000000000000', 2)
    # delta_in = int('0000000000000000010100000000000000000000000000000101000000000000',2)
    # fixed_in = solver.find_optimal_characteristic_with_fixed_delta_in(delta_in=delta_in, min_blocks=1, max_blocks=2, prune_level=1, repeat=1, print_paths=True)
    # fixed_out = solver.find_optimal_characteristic_with_fixed_delta_out(delta_out=delta_in, min_blocks=1, max_blocks=2, prune_level=1, repeat=1, print_paths=True)

    # print(fixed_in)

    delta_in = int("0000000000000000010100000000000000000000000000000101000000000000",2)
    delta_out = int("0000100000001000000001000000010000000010000000100000000100000001",2)

    solver.find_best_differential(delta_in=delta_in, delta_out=delta_out, repeat=4, print_paths=True)

if __name__ == "__main__":
    main()
