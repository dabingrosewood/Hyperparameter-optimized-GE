from fitness.supervised_learning.supervised_learning import supervised_learning

from algorithm.parameters import params
from utilities.fitness.error_metric import rmse
from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants
import sys
import numpy as np
np.seterr(all="raise")


class simple_regression(supervised_learning):
    """Fitness function for regression. We just slightly specialise the
    function for supervised_learning."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = rmse

        self.maximise = params['ERROR_METRIC'].maximise

        self.max = params['MAX_FITNESS_INVOKE']
        #==rmse().maximize

    def evaluate(self, ind, **kwargs):

        dist = kwargs.get('dist', 'training')

        self.max -= 1
        # print(self.max)
        if self.max <= 0:
            sys.exit('limited number of invoking the fitness func is used up')

        if dist == "training":
            # Set training datasets.
            x = self.training_in
            y = self.training_exp

        elif dist == "test":
            # Set test datasets.
            x = self.test_in
            y = self.test_exp

        else:
            raise ValueError("Unknown dist: " + dist)

        if params['OPTIMIZE_CONSTANTS']:
            # if we are training, then optimize the constants by
            # gradient descent and save the resulting phenotype
            # string as ind.phenotype_with_c0123 (eg x[0] +
            # c[0] * x[1]**c[1]) and values for constants as
            # ind.opt_consts (eg (0.5, 0.7). Later, when testing,
            # use the saved string and constants to evaluate.
            if dist == "training":
                return optimize_constants(x, y, ind)

            else:
                # this string has been created during training
                phen = ind.phenotype_consec_consts
                c = ind.opt_consts
                # phen will refer to x (ie test_in), and possibly to c
                yhat = eval(phen)
                assert np.isrealobj(yhat)

                # let's always call the error function with the
                # true values first, the estimate second
                return params['ERROR_METRIC'](y, yhat)

        else:
            # phenotype won't refer to C
            yhat = eval(ind.phenotype)
            assert np.isrealobj(yhat)

            # let's always call the error function with the true
            # values first, the estimate second
            return params['ERROR_METRIC'](y, yhat)
