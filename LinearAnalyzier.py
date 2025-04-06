"""
characteristic_searcher
===========

This module provides the CharacteristicSearcher class for performing characteristic search in cryptanalysis.

Classes:
- CharacteristicSearcher: Implements the search algorithm for finding linear and differential characteristics in substitution permutation network based cipher.

"""
from collections import defaultdict
from math import log2
from itertools import islice, combinations
from z3 import Optimize, BitVec, sat, Concat, And, Not, Implies, Extract, Function, BitVecSort, RealSort, Product, Or
from utils import calculate_linear_bias, all_smt, formatted_bits



class LinearAnalyzer:
    """A class for finding characteristics (linear or differential) of a substitution
    permutation network with provided S-box and P-box with a given number of rounds.

    Attributes:
        sbox: A list representing the substitution box.
        pbox: A list representing the permutation box.
        num_rounds: An integer representing the number of rounds.
        block_size: An integer representing the number of bits in the block.
        box_size: An integer representing the size of the S-box in bits.
        num_blocks: An integer representing the number of sboxes in a block
        mode: A string representing the mode, which can be 'linear' or 'differential'.
        bias: A Counter dictionary representing linear or differential bias
              of sbox input/output pairs
        solutions: A dictionary containing list of valid characteristic masks for a given
            set of included and excluded blocks
        solver: SMT solver (optimize) instance to search the characteristics
    """

    def __init__(self, sbox: list, pbox: list, num_rounds: int, result_file: str):
        """Initializes the CharacteristicSolver with the given sbox, pbox, num_rounds and mode.

        Args:
            sbox (list): The substitution box.
            pbox (list): The permutation box.
            num_rounds (int): The number of rounds.
            mode (str, optional): The mode of operation. Defaults to 'linear'.
        """
        self.sbox = sbox
        self.pbox = pbox
        self.block_size = len(pbox)
        self.num_rounds = num_rounds
        self.result_file = result_file

        self.sub_block_size = int(log2(len(sbox)))
        self.num_sub_blocks = len(pbox) // self.sub_block_size

        self.lat = calculate_linear_bias(sbox)
       
        self.solver = Optimize()
        self.solutions = defaultdict(list)
        self.prune_level = 0


        self.inputs = None
        self.outputs = None
        self.bv_input_masks = None
        self.bv_output_masks = None

        self.sboxf = None
        

        self.objectives = {
            'reduced': lambda rounds: sum([
                self.sboxf(
                    self.inputs[i // self.num_sub_blocks][i % self.num_sub_blocks],
                    self.outputs[i // self.num_sub_blocks][i % self.num_sub_blocks])
                for i in range(self.num_sub_blocks * rounds)
            ]),           
            'linear': lambda rounds: 2**(self.num_sub_blocks * rounds - 1) * Product([
                self.sboxf(
                    self.inputs[i // self.num_sub_blocks][i % self.num_sub_blocks],
                    self.outputs[i // self.num_sub_blocks][i % self.num_sub_blocks])
                for i in range(self.num_sub_blocks * rounds)
            ]) / ((2**self.sub_block_size)**(self.num_sub_blocks * rounds))
        }

    def apply_pbox_z3(self, input: BitVec, output: BitVec):
        """Function to perform the permutation -> it generates the neccessary constraints

        Args:
            input (BitVec)- the vector on which we apply the pbox
            output (BitVec) - the vector after permutation

        Output:
            constraints (list) - list of constraints for the pbox
        """

        constraints = []
        for i, v in enumerate(self.pbox):
            constraints.append(
                Extract(len(self.pbox) - 1 - i, len(self.pbox) - 1 - i, input)
                == Extract(len(self.pbox) - 1 - v, len(self.pbox) - 1 - v, output)
            )
        return constraints

    def initialize_sbox_structure(self):
        """This function transforms the algorithm structure, sbox & ppbox, into z3 constraints that the solver will be capable to operate with.

        In self.inputs we have the input of each round, on which we apply the sbox and also pbox so in self.outputs we will have
        the output of each round. The output of one round is the input to the next round

        Note: The method also creates a view of the input and output layers for viewing the results of the analysis
        """
        self.inputs = [
            [
                BitVec("r{}_i{}".format(r, i), self.sub_block_size)
                for i in range(self.num_sub_blocks)
            ]
            for r in range(self.num_rounds + 1)
        ]  
        self.outputs = [
            [
                BitVec("r{}_o{}".format(r, i), self.sub_block_size)
                for i in range(self.num_sub_blocks)
            ]
            for r in range(self.num_rounds)
        ]

        for i in range(self.num_rounds):
            self.solver.add(
                self.apply_pbox_z3(Concat(self.outputs[i]), Concat(self.inputs[i + 1]))
            )

        non_zero_conditions = [
            self.inputs[0][i] != 0 for i in range(self.num_sub_blocks)
        ]
        self.solver.add(Or(*non_zero_conditions))

        for r in range(self.num_rounds):
            for i in range(self.num_sub_blocks):
                self.solver.add(
                    Implies(self.inputs[r][i] != 0, self.outputs[r][i] != 0)
                )
                self.solver.add(
                    Implies(self.inputs[r][i] == 0, self.outputs[r][i] == 0)
                )

        self.bv_input_masks = [
            Concat(self.inputs[i]) for i in range(self.num_rounds + 1)
        ]
        self.bv_output_masks = [Concat(self.outputs[i]) for i in range(self.num_rounds)]

    def add_bias_constraints(self, prune_level: int, excluded_inputs_from_pruning: list = None):
        for i in range(2**self.sub_block_size):
            for j in range(2**self.sub_block_size):
                if excluded_inputs_from_pruning!=None:
                    if i in excluded_inputs_from_pruning:
                        if self.lat[(i, j)] >=2:
                            self.solver.add(self.sboxf(i, j) == self.lat[(i, j)])
                        else:
                            self.solver.add(self.sboxf(i, j) == 0)
                    else:
                        if self.lat[(i, j)] >= 2**prune_level:
                            self.solver.add(self.sboxf(i, j) == self.lat[(i, j)])
                        else:
                            self.solver.add(self.sboxf(i, j) == 0)
                else:
                    if self.lat[(i, j)] >= 2**prune_level:
                        self.solver.add(self.sboxf(i, j) == self.lat[(i, j)])
                    else:
                        self.solver.add(self.sboxf(i, j) == 0)
        for r in range(self.num_rounds):
            for i in range(self.num_sub_blocks):
                self.solver.add(
                    Implies(
                        And(self.inputs[r][i] != 0, self.outputs[r][i] != 0),
                        self.sboxf(self.inputs[r][i], self.outputs[r][i]) != 0,
                    )
                )

    def init_characteristic_solver(self, prune_level=-1,  excluded_inputs_from_pruning: list = None):

            self.initialize_sbox_structure()
            self.sboxf = Function(
                "sbox",
                BitVecSort(self.sub_block_size),
                BitVecSort(self.sub_block_size),
                RealSort(),
            )

            try:
                assert (
                    self.solver.check()
                )  # check if the sbox conditions can be satisfied before proceeding to add more constraints

                if prune_level < 0:
                    print(
                        "No prune level provided. Trying to find the optimal prune level!"
                    )
                    low, high = 0, len(self.sbox) // 4
                    while low <= high:
                        mid = (low + high) // 2

                        print(f"Trying to prune values smaller than 2**{mid}")

                        self.solver.push()

                        self.solver.set(timeout=10000)
                        # self.add_ddt_constraints(prune_level=mid, excluded_inputs_from_pruning=excluded_inputs_from_pruning)
                        self.add_bias_constraints(prune_level=prune_level)

                        if self.solver.check() == sat:
                            print(
                                f"Success in pruning with {mid} as prune level, trying a bigger value!"
                            )
                            low = mid + 1
                        else:
                            print(
                                f"Failure, could not prune with {mid}, trying a lower value!"
                            )
                            high = mid - 1

                        self.solver.pop()

                    self.solver.set(timeout=0)

                    self.prune_level = high
                    print(f"Best pruning level found = {self.prune_level}")
                    print()
                    self.add_bias_constraints(prune_level=prune_level)
                else:
                    self.solver.push()
                    self.add_bias_constraints(prune_level=prune_level)
                    # self.add_ddt_constraints(prune_level=prune_level, excluded_inputs_from_pruning=excluded_inputs_from_pruning)
                    if self.solver.check() == sat:
                        self.solver.pop()

                        self.prune_level = prune_level
                        self.add_bias_constraints(prune_level=prune_level)
                      
                        # self.add_ddt_constraints(prune_level=prune_level, excluded_inputs_from_pruning=excluded_inputs_from_pruning)
                    else:
                        print(
                            "The provided pruning lever results in an unsat system! Searching for optimal pruning level..."
                        )
                        print()
                        self.solver.pop()
                        self.init_characteristic_solver(prune_level=-1, excluded_inputs_from_pruning=excluded_inputs_from_pruning)

            except AssertionError:
                print("Constraints added for the sbox structure are unsatisfiable.")

    def print_paths(self, input: list, output: list):
        """Method to print te path of the differential analysis results

        The function prints the input and output of each round in a formatted manner so that the user can see in a more
        visual way how the differences are propagating. It splits the input in groups of sub_block_size bits
        split by a separator "|". The function also specifies the round associated with the input and output

        Args:
            input - list containing the inputs of each round
            output - list containting the output after applying the sbox

        """
        with open(self.result_file, "a") as file:
            num_rounds_performed = len(output)
            for i in range(num_rounds_performed):
                file.write(
                    formatted_bits(input[i], self.block_size, self.sub_block_size)
                    + "\n"
                )
                file.write(
                    " ".join(["-" * self.sub_block_size] * self.num_sub_blocks) + "\n"
                )
                file.write(
                    formatted_bits(output[i], self.block_size, self.sub_block_size)
                    + "\n\n"
                )

            # file.write(
            #     f"Value obtained after performing differential analysis on the SPN: "
            # )
            file.write(
                formatted_bits(
                    input[num_rounds_performed], self.block_size, self.sub_block_size
                )
                + "\n"
            )

    def get_masks(self, num_rounds, n=1, print_paths=True):
        """Method to return the input_masks, output_masks, total_probability, active, counter_active_blocks

        Args:
            num_rounds (int) - number of rounds on which we apply the solver
            n (int, optional) - number of masks to return, basically number of solutions the solver returns
            print_paths (bool, optional) - whether or not to print the paths of the solution

        Returns:
            list: a list of tuples, each tuple contains input_masks, output_masks, total_probability, active, counter_active_blocks
            input_mask - input of the system solved by the smt solver
            output_mask - solution of the system
            total_probability - probability of the differential characteristic
            active - list of booleans, where each element represents the state of the sub block in a block
                    True means that the block is active and False the opposite
            counter_active_blocks - number of active blocks in the full solution, basically counts for each round how many sub-blocks are active

        Note: the solver uses the differential objective defined in the __init__ function in which we add the occurences
            of a specific difference, and in the end, we divide by  (2**self.sub_block_size) ** (self.num_sub_blocks * rounds)
            so that we can recover the probability of differences to occur

        """
        masks = []

        # we use the all_smt helper function that solves the system based on self.bv_input_masks of the last round
        # the last round has the accumulated differential effect of all of the previous rounds
        for m in islice(all_smt(self.solver, [self.bv_input_masks[num_rounds - 1]]), n):
            input_masks = [
                m.eval(i).as_long() for i in self.bv_input_masks[: num_rounds + 1]
            ]
            output_masks = [
                m.eval(i).as_long() for i in self.bv_output_masks[:num_rounds]
            ]
            total_probability = m.eval(
                self.objectives["linear"](num_rounds)
            ).as_fraction()

            active = [m.eval(i).as_long() != 0 for i in self.inputs[num_rounds - 1]]

            counter_active_blocks = 0
            for j in range(num_rounds):
                active_blocks = [m.eval(i).as_long() != 0 for i in self.inputs[j]]
                for status in active_blocks:
                    if status == True:
                        counter_active_blocks += 1

            if print_paths:
                self.print_paths(input_masks, output_masks)
                with open(self.result_file, "a") as file:
                    file.write(f"Total probability is: {total_probability}" + "\n")
                    file.write(
                        f"Number of active blocks is: {counter_active_blocks}"
                        + 3 * "\n"
                    )

            masks.append(
                (
                    input_masks,
                    output_masks,
                    total_probability,
                    active,
                    counter_active_blocks,
                )
            )

        return masks
    
    def solve_for_specific_blocks(
        self,
        include_blocks=(),
        exclude_blocks=(),
        num_rounds=0,
        num_solutions=1,
        print_paths=True,
    ):
        if num_rounds == 0:
            num_rounds = self.num_rounds
        else:
            num_rounds = min(num_rounds, self.num_rounds)


        self.solver.push()

        for i in include_blocks:
            self.solver.add(self.inputs[num_rounds - 1][i] != 0)

        for i in exclude_blocks:
            self.solver.add(self.inputs[num_rounds - 1][i] == 0)

        self.solver.maximize(self.objectives["reduced"](num_rounds))

        solutions = self.get_masks(num_rounds, num_solutions, print_paths)

        self.solutions[
            (tuple(sorted(include_blocks)), tuple(sorted(exclude_blocks)))
        ].extend(solutions)

        self.solver.pop()

        return [
            (input_masks[0], input_masks[-1], calc_probability, active_blocks)
            for input_masks, _, calc_probability, _, active_blocks in solutions
        ]

    def solve_for_multiple_active_blocks(
        self,
        min_active_blocks=1,
        max_active_blocks=1,
        num_rounds=0,
        num_solutions=1,
        print_paths=True,
    ):
        """
        Function that generates satisfiable models for certain number of active sub-blocks we want to include in the analisys.
        This function is an extension of search_for_blocks() because here we include up to num_sub_blocks - 1 active blocks in the result
        The difference is, that here we generate in the function the blocks we want to include/exclude (we will have all possible combinations not only one block)

        Args:
            min_active_blocks (int, optional): minimum number of active blocks we include in the analysis, default only one
            max_active_blocks (int, optional): maximum number of active blocks we include in the analysis, default only one
            num_rounds (int, optional) - the number of rounds for which we search the characteristic, if it is set
                                        to 0, then it will use the number of rounds of the solver
            num_solution (int, optional) - number of solutions to return; by default there is only one
            print_paths (bool, optional) - whether or not to print the paths of the solution

        Returns:
            list - list of tuples, each tuple contains:
                 input_masks[0] - input_mask,
                 input_masks[-1] - the result,
                 calc_probability - the probability,
                 active_blocks - number of active blocks

        Note: If you want to use this function by itself you need to call in your main this method 'self.init_differential_characteristic_solver(prune_level)'

        """
        if num_rounds == 0:
            num_rounds = self.num_rounds
        else:
            num_rounds = min(num_rounds, self.num_rounds)

        if max_active_blocks > (self.num_sub_blocks - 1):
            max_active_blocks = self.num_sub_blocks - 1

        if min_active_blocks <= 0:
            min_active_blocks = 1


        all_solutions = []

        self.solver.push()
        for num_active in range(min_active_blocks, max_active_blocks + 1):
            for active_blocks in combinations(range(self.num_sub_blocks), num_active):
                exclude_blocks = set(range(self.num_sub_blocks)) - set(active_blocks)

                self.solver.push()
                for block in active_blocks:
                    self.solver.add(self.inputs[num_rounds - 1][block] != 0)
                for block in exclude_blocks:
                    self.solver.add(self.inputs[num_rounds - 1][block] == 0)

                self.solver.maximize(self.objectives["reduced"](num_rounds))

                solutions = self.get_masks(num_rounds, num_solutions, print_paths)
                all_solutions.extend(solutions)

                self.solutions[
                    (tuple(sorted(active_blocks)), tuple(sorted(exclude_blocks)))
                ].extend(solutions)

                self.solver.pop()

        self.solver.pop()
        return [
            (input_masks[0], input_masks[-1], calc_probability, active_blocks)
            for input_masks, _, calc_probability, _, active_blocks in all_solutions
        ]

    def search_exclusive_masks(self, prune_level=-1, repeat=1, excluded_inputs_from_pruning: list = None, print_paths=True):
        self.init_differential_characteristic_solver(prune_level, excluded_inputs_from_pruning)
        masks = []

        for i in range(self.num_sub_blocks):
            include_blocks = {i}
            exclude_blocks = set(range(self.num_sub_blocks)) - include_blocks

            masks.extend(
                self.solve_for_specific_blocks(
                    include_blocks,
                    exclude_blocks,
                    num_solutions=repeat,
                    print_paths=print_paths,
                )
            )

        return masks

    def search_exclusive_masks_extended(
        self, min_blocks=1, max_blocks=1, repeat=1, print_paths=True
    ):
      
        differences = []

        for i in range(min_blocks, max_blocks + 1):
            differences.extend(
                self.solve_for_multiple_active_blocks(
                    min_active_blocks=i,
                    max_active_blocks=i,
                    num_solutions=repeat,
                    print_paths=print_paths,
                )
            )

        return differences

    def find_optimal_approx(
        self, min_blocks=1, max_blocks=1, prune_level=-1, repeat=1, excluded_inputs_from_pruning: list = None, print_paths=True
    ):

        self.init_characteristic_solver(prune_level, excluded_inputs_from_pruning)

        all_masks = self.search_exclusive_masks_extended(
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            repeat=repeat,
            print_paths=print_paths,
        ) 

        sorted_masks = sorted(all_masks, key=lambda x: (-x[2]))
        optimal_characteristic = sorted_masks[0] if sorted_masks else None

        if optimal_characteristic:
            input_mask, output_mask, probability, active_blocks = optimal_characteristic


            with open(self.result_file, "a") as file:
                file.write("\nOptimal Trail Found:" + "\n")
                file.write(
                    f"Input mask: {formatted_bits(input_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )
                file.write(
                    f"Output mask: {formatted_bits(output_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )

                file.write(f"Probability: {probability}" + "\n")
                file.write(f"Active blocks: {active_blocks}" + "\n" + "\n" + "\n")

        return optimal_characteristic

    def find_lin_hulls(
        self, imask: int, omask: int, repeat=2, print_paths=True
    ):
        """
        Searches for the best differential (characteristic) with specific input and output differences.

        Args:
            delta_in (int): The input differential (delta) for the SPN.
            delta_out (int): The output differential (delta) for the SPN.
            repeat (int, optional): The number of solutions to find.
            num_rounds (int, optional): The number of rounds for the search. Uses self.num_rounds if set to 0.
            print_paths (bool, optional): Whether or not to print the paths of the solution.

        Returns:
            list: A list of tuples containing the characteristics found. Each tuple includes:
                input_mask, output_mask, probability, active_blocks.
        """

        num_rounds = self.num_rounds #just for writing the cod shorter 

        # self.init_differential_characteristic_solver(-1)
        self.init_characteristic_solver(0) #prune level 0 means we will include all non-zero values from DDT

        self.solver.push()

        # Adding constraints for delta_in and delta_out
        self.solver.add(self.bv_input_masks[0] == imask)
        self.solver.add(self.bv_input_masks[num_rounds] == omask)

        # Maximizing the objective function
        self.solver.maximize(self.objectives["reduced"](num_rounds))

        # Getting the masks (solutions)
        solutions = self.get_masks(num_rounds, repeat, print_paths)

        self.solver.pop()

        total_probability = sum(
            calc_probability for _, _, calc_probability, _, _ in solutions
        )

        with open(self.result_file, "a") as file:
            file.write(
                f"Total probability for ({formatted_bits(imask, self.block_size, self.sub_block_size)}, {formatted_bits(omask, self.block_size, self.sub_block_size)}) is {total_probability}"
            )
            file.write("\n\n")

        return [
            (input_masks[0], input_masks[-1], calc_probability, active_blocks)
            for input_masks, _, calc_probability, _, active_blocks in solutions
        ]

    def find_optimal_characteristic_with_fixed_inp_mask(
        self,
        imask: int,
        min_blocks=1,
        max_blocks=1,
        prune_level=-1,
        repeat=1,
        print_paths=True,
    ):
        """
        Finds and prints the optimal differential characteristic with fixed delta_in for the SPN.

        Args:
            delta_in: the input difference we want to start with
            min_blocks (int, optional) - the minimum number of active blocks in the last round of the differential analysis; default it tries only one block
            max_blocks (int, optional) - the maximum number of active blocks in the last round of the differential analysis; default it tries only one block
            prune_level (int, optional) - the prune level we use in the analisys
            repeat (int, optional) - how many solutions we want in the searching algorithm,
                                    we use the method .search_exclusive_masks_extended() which, for more solutions retrieved, it might find a better differential characteristic
                                    from the solver's implementation, the first solution should be the best, but given that the constraints are kind of flexible one solution might be better than another
            print_paths (bool, optional) - whether or not to print the paths of the solution in the analisys

        Returns:
            tuple - a tuple containing the neccessary informations of the differential characteristic such as:
                    input_mask, output_mask, probability, active_blocks
        """

        self.solver.push()

        # Adding constraints for delta_in
        self.init_characteristic_solver(prune_level)

        self.solver.add(self.bv_input_masks[0] == imask)

        all_masks = self.search_exclusive_masks_extended(
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            repeat=repeat,
            print_paths=print_paths,
        )  # we use the extended function if we want to include more than one active block in the characteristic

        sorted_masks = sorted(all_masks, key=lambda x: (-x[2]))
        optimal_characteristic = sorted_masks[0] if sorted_masks else None

        if optimal_characteristic:
            input_mask, output_mask, probability, active_blocks = optimal_characteristic

            with open(self.result_file, "a") as file:
                file.write("\nOptimal Trail Found:" + "\n")
                file.write(
                    f"Input mask: {formatted_bits(input_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )
                file.write(
                    f"Output mask: {formatted_bits(output_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )

                file.write(f"Probability: {probability}" + "\n")
                file.write(f"Active blocks: {active_blocks}" + "\n" + "\n" + "\n")

        self.solver.pop()

        return optimal_characteristic

    def find_optimal_characteristic_with_fixed_out_mask(
        self,
        omask: int,
        min_blocks=1,
        max_blocks=1,
        prune_level=-1,
        repeat=1,
        print_paths=True,
    ):
        """
        Finds and prints the optimal differential characteristic with fixed delta_in for the SPN.

        Args:
            delta_out: the output difference we want to obtain
            min_blocks (int, optional) - the minimum number of active blocks in the last round of the differential analysis; default it tries only one block
            max_blocks (int, optional) - the maximum number of active blocks in the last round of the differential analysis; default it tries only one block
            prune_level (int, optional) - the prune level we use in the analisys
            repeat (int, optional) - how many solutions we want in the searching algorithm,
                                    we use the method search_exclusive_masks_extended() which, for more solutions retrieved, it might find a better differential characteristic
                                    from the solver's implementation, the first solution should be the best, but given that the constraints are kind of flexible one solution might be better than another
            print_paths (bool, optional) - whether or not to print the paths of the solution in the analisys

        Returns:
            tuple - a tuple containing the neccessary informations of the differential characteristic such as:
                    input_mask, output_mask, probability, active_blocks
        """

        self.solver.push()
        self.init_characteristic_solver(prune_level)

        # Adding constraints for delta_out
        self.solver.add(self.bv_input_masks[self.num_rounds] == omask)

        all_masks = self.search_exclusive_masks_extended(
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            repeat=repeat,
            print_paths=print_paths,
        )  # we use the extended function if we want to include more than one active block in the characteristic

        sorted_masks = sorted(all_masks, key=lambda x: (-x[2]))
        optimal_characteristic = sorted_masks[0] if sorted_masks else None

        if optimal_characteristic:
            input_mask, output_mask, probability, active_blocks = optimal_characteristic

            with open(self.result_file, "a") as file:
                file.write("\nOptimal Trail Found:" + "\n")
                file.write(
                    f"Input mask: {formatted_bits(input_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )
                file.write(
                    f"Output mask: {formatted_bits(output_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )

                file.write(f"Probability: {probability}" + "\n")
                file.write(f"Active blocks: {active_blocks}" + "\n" + "\n" + "\n")

        self.solver.pop()

        return optimal_characteristic

    # basically delta_in = delta_out
    def find_optimal_trail_iterative(
        self, min_blocks=1, max_blocks=1, prune_level=-1, repeat=1, print_paths=True
    ):
        """
        Finds and prints the optimal differential characteristic with fixed delta_in=delta_out (ITERATIVE) for the SPN.

        Args:
            min_blocks (int, optional) - the minimum number of active blocks in the last round of the differential analysis; default it tries only one block
            max_blocks (int, optional) - the maximum number of active blocks in the last round of the differential analysis; default it tries only one block
            prune_level (int, optional) - the prune level we use in the analisys
            repeat (int, optional) - how many solutions we want in the searching algorithm,
                                    we use the method search_exclusive_masks_extended() which, for more solutions retrieved, it might find a better differential characteristic
                                    from the solver's implementation, the first solution should be the best, but given that the constraints are kind of flexible one solution might be better than another
            print_paths (bool, optional) - whether or not to print the paths of the solution in the analisys

        Returns:
            tuple - a tuple containing the neccessary informations of the differential characteristic such as:
                    input_mask, output_mask, probability, active_blocks
        """

        self.solver.push()

        self.init_characteristic_solver(prune_level)

        # Adding constraints for delta_in = delta_out
        self.solver.add(self.bv_input_masks[self.num_rounds] == self.bv_input_masks[0])

        all_masks = self.search_exclusive_masks_extended(
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            repeat=repeat,
            print_paths=print_paths,
        )  # we use the extended function if we want to include more than one active block in the characteristic

        sorted_masks = sorted(all_masks, key=lambda x: (-x[2]))
        optimal_characteristic = sorted_masks[0] if sorted_masks else None

        if optimal_characteristic:
            input_mask, output_mask, probability, active_blocks = optimal_characteristic

            with open(self.result_file, "a") as file:
                file.write("\nOptimal Trail Found:" + "\n")
                file.write(
                    f"Input mask: {formatted_bits(input_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )
                file.write(
                    f"Output mask: {formatted_bits(output_mask, self.block_size, self.sub_block_size)}"
                    + "\n"
                )

                file.write(f"Probability: {probability}" + "\n")
                file.write(f"Active blocks: {active_blocks}" + "\n" + "\n" + "\n")

        self.solver.pop()

        return optimal_characteristic