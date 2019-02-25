from ABC.bee import Bee
import numpy as np


class EmployeeBee(Bee):
    def __init__(self, objective_function, processing_opt):
        Bee.__init__(self, objective_function, processing_opt)

    """
        producing new food source and if the solution is an improvement 
        (currently minimizing the objective function value)
        
        :return void
    """
    def search(self, other_food_source, max_tries):
        if self.n_trial >= max_tries:
            return
        alternative = Bee.produce_new_food_source(self, other_food_source)

        """
            if the alternative food source crossed the values of the objective function adapt them accordingly        
        """
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
            self.n_trial = Bee.INITIAL_NUM_OF_TRIALS

    def get_food_source(self):
        return self.food_source


class OnLookerBee(Bee):
    def __init__(self, objective_function, processing_opt):
        Bee.__init__(self, objective_function, processing_opt)

    """
        choose from all the food sources the other food source based on probability p (formula 5 in the article)
        
        :return void
    """
    def search(self, probs, food_sources, max_tries):
        if self.n_trial >= max_tries:
            return
        other_food_source = np.random.choice(food_sources, p=probs)
        alternative = Bee.produce_new_food_source(self, other_food_source)

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
            self.n_trial = Bee.INITIAL_NUM_OF_TRIALS
