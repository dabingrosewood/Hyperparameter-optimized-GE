from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import sys


class test_fitness(base_ff):

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set target string.
        # self.target = params['TARGET']
        self.max=params['MAX_FITNESS_INVOKE']

    def evaluate(self, ind, **kwargs):
        fitness=0.1



        self.max-=1
        # print('yuxia=',self.max)
        if self.max<=0:
            sys.exit('limited number of invoking the fitness func is used up')
        return fitness
