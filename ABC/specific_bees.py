from ABC.artificial_bee import ArtificialBee
import numpy as np


class EmployeeBee(ArtificialBee):
    def __init__(self, objective_function, processing_opt):
        ArtificialBee.__init__(self, objective_function, processing_opt)

    def search(self, other_food_source, max_tries):
        if self.n_trial >= max_tries:
            return
        alternative = ArtificialBee.produce_new_food_source(self, other_food_source)

        for ind, val in enumerate(alternative):
            if val < self.objective_function.get_min_lim():
                alternative[ind] = self.objective_function.get_min_lim()
            if val > self.objective_function.get_max_lim():
                alternative[ind] = self.objective_function.get_max_lim()

        new_fitness = self.objective_function.evaluate(alternative)

        if new_fitness >= self.fitness_value:
            self.n_trial = self.n_trial + 1
        else:
            self.fitness_value = new_fitness
            self.food_source = alternative
            self.n_trial = ArtificialBee.INITIAL_NUM_OF_TRIALS

    def get_food_source(self):
        return self.food_source


class OnLookerBee(ArtificialBee):
    def __init__(self, objective_function, processing_opt):
        ArtificialBee.__init__(self, objective_function, processing_opt)

    def search(self, probs, food_sources, max_tries):
        if self.n_trial >= max_tries:
            return
        other_food_source = np.random.choice(food_sources, p=probs)
        alternative = ArtificialBee.produce_new_food_source(self, other_food_source)

        for ind, val in enumerate(alternative):
            if val < self.objective_function.get_min_lim():
                alternative[ind] = self.objective_function.get_min_lim()
            if val > self.objective_function.get_max_lim():
                alternative[ind] = self.objective_function.get_max_lim()

        new_fitness = self.objective_function.evaluate(alternative)

        if new_fitness >= self.fitness_value:
            self.n_trial = self.n_trial + 1
        else:
            self.fitness_value = new_fitness
            self.food_source = alternative
            self.n_trial = ArtificialBee.INITIAL_NUM_OF_TRIALS
