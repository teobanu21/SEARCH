import DifferentialAnalyzer as DifferentialAnalyzer


def main():
    sbox = {
        0: 0xE,
        1: 0x4,
        2: 0xD,
        3: 0x1,
        4: 0x2,
        5: 0xF,
        6: 0xB,
        7: 0x8,
        8: 0x3,
        9: 0xA,
        0xA: 0x6,
        0xB: 0xC,
        0xC: 0x5,
        0xD: 0x9,
        0xE: 0x0,
        0xF: 0x7,
    }
    sbox_list = [sbox[key] for key in range(len(sbox))]

    pbox = [
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
        4,
        8,
        12,
        16,
    ]

    for idx in range(16):
        pbox[idx] -= 1

    num_rounds = 3

    solver = DifferentialAnalyzer.DifferentialAnalyzer(
        sbox_list, pbox, num_rounds, "tutorial_didactic.txt"
    )

    # solver.init_differential_characteristic_solver()
    # solver.get_differences(num_rounds, 1, True )
    # print(solver.search_exclusive_masks(repeat=1, print_paths=False))
    # solver.search_exclusive_masks(repeat=1, print_paths=True)

    # optimal_characteristic = solver.find_optimal_characteristic(min_blocks=1, max_blocks=3, prune_level=3, repeat=1, excluded_inputs_from_pruning=[
    #                                                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15], print_paths=True)
    # print(optimal_characteristic)

    with open("tutorial_didactic.txt", 'a') as file:
        file.write("\n" + "Finding best differential" + 2*"\n")

    delta_in = 45232
    delta_out = 2056

    solver.find_best_differential(
        delta_in, delta_out, repeat=5, print_paths=True
    )

    # with open("tutorial_didactic.txt", 'a') as file:
    #     file.write("\n" + "Finding characteristic with fixed delta in" + 2*"\n")

    # characteristic_delta_in = solver.find_optimal_characteristic_with_fixed_delta_in(delta_in=delta_in, min_blocks=1, max_blocks=4, prune_level=-1, repeat=1, print_paths=True)
    # print(characteristic_delta_in)

    # with open("tutorial_didactic.txt", 'a') as file:
    #     file.write("\n" + "Finding characteristic with fixed delta out" + 2*"\n")

    # characteristic_delta_out = solver.find_optimal_characteristic_with_fixed_delta_out(delta_out=delta_out, min_blocks=1, max_blocks=4, prune_level=-1, repeat=1, print_paths=True)
    # print(characteristic_delta_out)

    # with open("tutorial_didactic.txt", 'a') as file:
    #     file.write("\n" + "Finding iterative characteristic (delta_in=delta_out)" + 2*"\n")

    # solver.find_optimal_characteristic_iterative(min_blocks=1, max_blocks=4, prune_level=-1, repeat=2, print_paths=True)


if __name__ == "__main__":
    main()
