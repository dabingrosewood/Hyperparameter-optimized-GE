from fitness.base_ff_classes.base_ff import base_ff
from algorithm.parameters import params
import sys

class santa(base_ff):
    """
    this is just for testing
    still a fake func
    """

    maximise = True  # True as it ever was.

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.max = params['MAX_FITNESS_INVOKE']

    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        self.max -= 1
        # print(self.max)
        if self.max <= 0:
            sys.exit('limited number of invoking the fitness func is used up')


        return 0.1
