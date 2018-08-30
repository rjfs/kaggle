import random
import numpy as np
import utils


class ParamsRandomSearch:

    def __init__(self, model_class, params_lists, eval_funct, n_runs=1000):
        self.model_class = model_class
        self.params_lists = params_lists
        self.eval_funct = eval_funct
        self.n_runs = n_runs
        self.goal = 'minimize'
        self.best_score = 0.0 if self.goal == 'maximize' else np.inf
        self.best_params = {}
        self.file_name = '%s-rsearch-log.txt' % utils.get_timestamp()

    def run(self):
        for i in range(self.n_runs):
            params = self.select_params()
            m = self.model_class(**params)
            score = self.eval_funct(m)
            max_cond = score > self.best_score and self.goal == 'maximize'
            min_cond = score < self.best_score and self.goal == 'minimize'
            if max_cond or min_cond:
                self.best_score = score
                self.best_params = params

            # Print iteration info to console
            self.print_iteration_info(i, params, score)
            # Print iteration info to file
            with open(self.file_name, 'a') as f:
                self.print_iteration_info(i, params, score, file=f)

    def print_iteration_info(self, i, params, score, file=None):
            print('[---------- Iteration %d ----------]' % i, file=file)
            print(params, file=file)
            print('Score: %.4f' % score, file=file)
            print('Best parameters (Score=%.4f):' % self.best_score, file=file)
            print(self.best_params, file=file)

    def select_params(self):
        return {
            l: random.choice(p_list)
            for l, p_list in self.params_lists.items()
        }
