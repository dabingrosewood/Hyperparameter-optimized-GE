from fitness.base_ff_classes.base_ff import base_ff
from algorithm.parameters import params
import sys

class pymax(base_ff):
    """
    Py-max is a max-style problem where the goal is to generate a function
    which outputs a large number. In the standard GP Max [Gathercole and
    Ross] problem this function can only use the constant (0.5) and functions
    (+, *). The Py-max problem allows more programming: numerical expressions,
    assignment statements and loops. See pymax.pybnf.

    Chris Gathercole and Peter Ross. An adverse interaction between crossover
    and restricted tree depth in genetic programming. In John R. Koza,
    David E. Goldberg, David B. Fogel, and Rick L. Riolo, editors, Genetic
    Programming 1996: Proceedings of the First Annual Conference.
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

        p, d = ind.phenotype, {}

        # Exec the phenotype.
        exec(p, d)

        # Get the output
        s = d['XXX_output_XXX']  # this is the program's output: a number.

        return s
