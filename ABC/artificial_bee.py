import random
import copy


class ArtificialBee:
    """
        the general representation of a bee in the ABC

    """
    INITIAL_NUM_OF_TRIALS = 0

    def __init__(self, objective_function, processing_opt):
        self.objective_function = objective_function
        self.n_trial = ArtificialBee.INITIAL_NUM_OF_TRIALS
        self.food_source = self.objective_function.random_sample()
        self.fitness_value = self.objective_function.evaluate(self.food_source)
        self.processing_opt = processing_opt

    def calc_fit(self):
        return 1 / (1 + self.fitness_value) if self.fitness_value >= 0 else 1 + abs(self.fitness_value)

    def get_fitness_value(self):
        return self.fitness_value

    def scout(self, max_trials):
        if self.n_trial >= max_trials:
            self.__scout()

    def __scout(self):
        self.food_source = [random.uniform(self.objective_function.get_min_lim(),
                                           self.objective_function.get_max_lim())
                            for _ in range(self.objective_function.get_dim())
                            ]
        self.n_trial = ArtificialBee.INITIAL_NUM_OF_TRIALS

    def produce_new_food_source(self, other_food_source):
        if self.processing_opt == 'all_dimensions':
            return [z_i_j + random.uniform(-1.0, 1.0) * (z_i_j - z_k_j)
                    for z_i_j, z_k_j in zip(self.food_source, other_food_source)]
        elif self.processing_opt == 'single_dimension':
            rand_ind = random.randint(0, len(self.food_source)-1)
            z_i_j = self.food_source[rand_ind]
            z_k_j = other_food_source[rand_ind]
            new_food_source = copy.deepcopy(self.food_source)
            new_food_source[rand_ind] = z_i_j + random.uniform(-1.0, 1.0) * (z_i_j - z_k_j)
            return new_food_source
