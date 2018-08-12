import random
import numpy as np


class ParamsRandomSearch:

    def __init__(self, model_class, params_lists, eval_funct, n_runs=1000):
        self.model_class = model_class
        self.params_lists = params_lists
        self.eval_funct = eval_funct
        self.n_runs = n_runs
        self.goal = 'minimize'

    def run(self):
        best_score = 0.0 if self.goal == 'maximize' else np.inf
        best_params = {}
        for i in range(self.n_runs):
            params = self.select_params()
            print('[---------- Iteration %d ----------]' % i)
            print(params)
            m = self.model_class(**params)
            score = self.eval_funct(m)
            print('Score: %.4f' % score)
            max_cond = score > best_score and self.goal == 'maximize'
            min_cond = score < best_score and self.goal == 'minimize'
            if max_cond or min_cond:
                best_score = score
                best_params = params
            print('Best parameters so far (Score=%.4f):' % best_score)
            print(best_params)

    def select_params(self):
        return {
            l: random.choice(p_list)
            for l, p_list in self.params_lists.items()
        }
