#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats,print_best_ever,store_best_ever
from algorithm.parameters import params, set_params
import sys


def mane():
    """ Run program """

    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)

    # for irace, just print the fitness
    # print_best_ever()

    #for mlp store the best fitness
    store_best_ever()



if __name__ == "__main__":
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself

    mane()
