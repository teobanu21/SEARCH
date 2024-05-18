import DifferentialAnalyzer as DifferentialAnalyzer
from utils import pretty_print_ddt


def main():
    sbox = {
        0: 0xC,
        1: 0x5,
        2: 0x6,
        3: 0xB,
        4: 0x9,
        5: 0x0,
        6: 0xA,
        7: 0xD,
        8: 0x3,
        9: 0xE,
        0xA: 0xF,
        0xB: 0x8,
        0xC: 0x4,
        0xD: 0x7,
        0xE: 0x1,
        0xF: 0x2,
    }
    sbox_list = [sbox[key] for key in range(len(sbox))]
    pbox = [
        0,
        16,
        32,
        48,
        1,
        17,
        33,
        49,
        2,
        18,
        34,
        50,
        3,
        19,
        35,
        51,
        4,
        20,
        36,
        52,
        5,
        21,
        37,
        53,
        6,
        22,
        38,
        54,
        7,
        23,
        39,
        55,
        8,
        24,
        40,
        56,
        9,
        25,
        41,
        57,
        10,
        26,
        42,
        58,
        11,
        27,
        43,
        59,
        12,
        28,
        44,
        60,
        13,
        29,
        45,
        61,
        14,
        30,
        46,
        62,
        15,
        31,
        47,
        63,
    ]

    num_rounds = 4

    solver = DifferentialAnalyzer.DifferentialAnalyzer(
        sbox_list, pbox, num_rounds, "present/articol.txt"
    )

    # pretty_print_ddt(solver.ddt)

    # for tests:
    # solver.init_differential_characteristic_solver()
    # solver.get_differences(num_rounds, 1, True )
    # print(solver.search_exclusive_masks(repeat=1, print_paths=False))
    # solver.search_exclusive_masks(repeat=1, print_paths=True)

    # optimal characteristic
    # char = solver.find_optimal_characteristic(
    #     min_blocks=2, max_blocks=3, prune_level=2, repeat=1, print_paths=True
    # )
    # print(char)

    # best differential
    # delta_in = 20565
    # delta_out = 13835058055282212864
    # solver.find_best_differential(
    #     delta_in, delta_out, repeat=10, num_rounds=num_rounds, print_paths=True
    # )

    # iterative = solver.find_optimal_characteristic_iterative(
    #     min_blocks=2, max_blocks=3, prune_level=1, repeat=1, print_paths=True)
    # print(iterative)


    #each of these 3 input differences can be used in determining the characteristic for 14 rounds based on 4 rounds iterative characteristtic
    # delta_in = int('0000000000000000000000010000000100000000000000000000000000000000', 2)
    # delta_in = int('0000000000000000000000000000100100000000000000000000000000001001',2)
    # delta_in = int('0000000000000000000000010000000100000000000000000000000000000000',2)

    # this difference is used for determining characteristics on 11-13 rounds
    # delta_in = int('0000000000000000000000000000100100000000000000000000000000001001',2)

    # char = solver.find_optimal_characteristic_with_fixed_delta_in(delta_in=delta_in, min_blocks=1, max_blocks=2, prune_level=1,repeat=1, print_paths=True)
    # print(char)

    # char = solver.find_optimal_characteristic_with_fixed_delta_out(delta_out=delta_in, min_blocks=1, max_blocks=2, prune_level=1, repeat=1, print_paths=True)
    # print(char)

    #only one characteristic with this properties
    delta_in_diff = int('0000000000000000000000010000000100000000000000000000000000000000',2)
    delta_out_diff = int('0000000000000000000000010000000100000000000000000000000000000000', 2)
    diff = solver.find_best_differential(delta_in=delta_in_diff, delta_out=delta_out_diff,repeat=4, print_paths=True)

if __name__ == "__main__":
    main()
