from ABC.specific_bees import OnLookerBee, EmployeeBee
from ABC.artificial_bee import ArtificialBee
import random


class ABClustering:
    def __init__(self, objective_function, colony_size=30, cycles=5000, max_trials=100, processing_opt='all_dimensions'):
        self.objective_function = objective_function
        self.colony_size = colony_size
        self.cycles = cycles
        self.max_trials = max_trials
        self.processing_opt = processing_opt

        self.optimal_val = None
        self.employee_bees = []
        self.onlooker_bees = []
        self._fit_values = []
        self.optimal_value_tracking = []

    def __initialize_colony(self):
        for _ in range(self.colony_size):
            self.employee_bees.append(EmployeeBee(self.objective_function, self.processing_opt))
            self.onlooker_bees.append(OnLookerBee(self.objective_function, self.processing_opt))

    """
        calculate the probability of each of the employee bees (based on formula 5 from the article)
    """
    def __fit_evaluation(self):
        self._fit_values = map(lambda emp_bee: emp_bee.calc_fit(), self.employee_bees)
        sum_of_fits = sum(self._fit_values)
        self._fit_values = map(lambda fit_i: fit_i / sum_of_fits, self._fit_values)
    """
        choose random food source (index in the list of employee bees) different from the original one
    """
    def __get_other_employee_source(self, current_ind):
        other_ind = current_ind
        while other_ind == current_ind:
            other_ind = random.randint(0, self.colony_size - 1)
        return other_ind

    def __employees_phase(self):
        for ind, bee in enumerate(self.employee_bees):
            other_ind = self.__get_other_employee_source(ind)
            self.employee_bees[ind].search(self.employee_bees[other_ind].get_food_source(), self.max_trials)

    def __on_lookers_phase(self):
        map(lambda on_looker: on_looker.search(self.employee_bees, self._fit_values), self.onlooker_bees)
    """
        scout phase in both the employees and the onlookers 
    """
    def __scouts_phase(self):
        map(lambda bee: ArtificialBee.scout(bee, self.max_trials), self.employee_bees + self.onlooker_bees)

    def get_optimization_path(self):
        return self.optimal_value_tracking

    def optimize(self):
        self.__initialize_colony()

        for cycle_no in range(self.cycles):
            self.__employees_phase()

            self.__fit_evaluation()

            self.__on_lookers_phase()

            self.__scouts_phase()

            max_fitness_in_cycle = min(map(lambda bee: bee.get_fitness_value(),
                                           self.employee_bees + self.onlooker_bees))
            self.optimal_value_tracking.append(max_fitness_in_cycle)
            # print("cycle no {0} \t->\tbest fitness: {1}".format(cycle_no, max_fitness_in_cycle))
