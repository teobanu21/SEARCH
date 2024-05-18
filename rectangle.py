import DifferentialAnalyzer as DifferentialAnalyzer
from utils import pretty_print_ddt


def main():

    sbox = {
        0: 0x6,
        1: 0x5,
        2: 0xC,
        3: 0xA,
        4: 0x1,
        5: 0xE,
        6: 0x7,
        7: 0x9,
        8: 0xB,
        9: 0x0,
        0xA: 0x3,
        0xB: 0xD,
        0xC: 0x8,
        0xD: 0xF,
        0xE: 0x4,
        0xF: 0x2,
    }

    # sbox_nou = {
    #     0: 0x6,
    #     1: 0XB,
    #     2: 0x1,
    #     3: 0x8,
    #     4: 0xC,
    #     5: 0x3,
    #     6: 0x7,
    #     7: 0x4,
    #     8: 0x5,
    #     9: 0x0,
    #     0xA: 0xE,
    #     0xB: 0XF,
    #     0xC: 0xA,
    #     0xD: 0XD,
    #     0xE: 0X9,
    #     0xF: 0x2,
    # }

    # sbox_list = [sbox_nou[key] for key in range(len(sbox_nou))]
    sbox_list = [sbox[key] for key in range(len(sbox))]

    # pbox = [0, 61, 18, 15, 4, 1, 22, 19, 8, 5, 26, 23, 12, 9, 30, 27, 16, 13, 34, 31, 20, 17, 38, 35, 24, 21, 42, 39, 28, 25, 46,
    #         43, 32, 29, 50, 47, 36, 33, 54, 51, 40, 37, 58, 55, 44, 41, 62, 59, 48, 45, 2, 63, 52, 49, 6, 3, 56, 53, 10, 7, 60, 57, 14, 11]

    pbox = [12, 17, 62, 3, 16, 21, 2, 7, 20, 25, 6, 11, 24, 29, 10, 15, 28, 33, 14, 19, 32, 37, 18, 23, 36, 41, 22, 27, 40, 45, 26,
            31, 44, 49, 30, 35, 48, 53, 34, 39, 52, 57, 38, 43, 56, 61, 42, 47, 60, 1, 46, 51, 0, 5, 50, 55, 4, 9, 54, 59, 8, 13, 58, 63]

    num_rounds = 4 # rectangle is defined on 25 rounds

    solver = DifferentialAnalyzer.DifferentialAnalyzer(
        sbox_list, pbox, num_rounds, "rectangle_nou/articol.txt")
    
    # pretty_print_ddt(solver.ddt)

    # optimal_characteristic = solver.find_optimal_characteristic(
    #     min_blocks=2, max_blocks=3, prune_level=2, repeat=1, excluded_inputs_from_pruning=[8], print_paths=True)

    # print(f"Optimal characteristic {optimal_characteristic}")

    # optimal_characteristic_iterative = solver.find_optimal_characteristic_iterative(
    #     min_blocks=2, max_blocks=4, prune_level=1, repeat=1, print_paths=True)
    # print(optimal_characteristic_iterative)

    # delta_in = int('0000000000000000001100100000000000000000000000000000000000000000', 2)
    # print(delta_in)
    # solver.find_optimal_characteristic_with_fixed_delta_in(delta_in=delta_in, min_blocks=1, max_blocks=4, prune_level=1, repeat=1,print_paths=True)
    # solver.find_optimal_characteristic_with_fixed_delta_out(delta_out=delta_in, min_blocks=1, max_blocks=6, prune_level=1, repeat=1, print_paths=True)

    delta_in = int("0000000000000000000000010000000100000000000000000000000000000000", 2)
    delta_out = int("0000000000000000000000010000000100000000000000000000000000000000", 2)
    diff = solver.find_best_differential(delta_in=delta_in, delta_out=delta_out, repeat=3, print_paths=True)
    print(diff)

if __name__ == "__main__":
    main()
