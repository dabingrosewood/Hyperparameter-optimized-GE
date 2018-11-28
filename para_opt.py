#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
@email: wangronin@gmail.com
        h.wang@liacs.leidenuniv.nl
"""

from pdb import set_trace

import numpy as np
import re,os,traceback
from datetime import datetime
from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

np.random.seed(666)

filename='../parameters/mlp_test/regression.txt'
dim = 2
n_step = 50
n_init_sample = 10 * dim
eval_type = 'dict' # control the type of parameters for evaluation: dict | list


def obj_func(x):

    file_object=open(filename)
    file_out=open('../parameters/tmp_para','w+')

    for line in file_object:
        # add more parameters here
        if re.search('CROSSOVER_PROBABILITY',line):
            file_out.writelines(re.sub(line, ('CROSSOVER_PROBABILITY:\t' + str(x['CROSSOVER_PROBABILITY'])+'\n'), line))

        elif re.search('MAX_GENOME_LENGTH', line):
            file_out.writelines(
                re.sub(line, ('MAX_GENOME_LENGTH:\t' + str(x['MAX_GENOME_LENGTH']) + '\n'), line))

        elif re.search('MAX_INIT_TREE_DEPTH', line):
            file_out.writelines(
                re.sub(line, ('MAX_INIT_TREE_DEPTH:\t' + str(x['MAX_INIT_TREE_DEPTH']) + '\n'), line))

        elif re.search('MAX_TREE_DEPTH', line):
            file_out.writelines(
                re.sub(line, ('MAX_TREE_DEPTH:\t\t' + str(x['MAX_TREE_DEPTH']) + '\n'), line))

        elif re.search('TOURNAMENT_SIZE', line):
            file_out.writelines(
                re.sub(line, ('TOURNAMENT_SIZE:\t' + str(x['TOURNAMENT_SIZE']) + '\n'), line))

        else:
            file_out.writelines(line)



    file_out.close()
    file_object.close()

    #building the command
    # print('-'*40+'*'*10+'-'*40+'\n')
    # print('parameters with \n',open('../parameters/tmp_para','r').read())
    cmd='python3 ponyge.py --parameters tmp_para'

    tmp=0
    err=0
    for i in range(5):
        try:
            p = os.popen(cmd)
            std_out = p.read()
            file_record = open('tmp_fitness', 'r').read()
            print('current parameter has the fitness of ',file_record)
            tmp+=float(file_record)
        except:
            # record the error and corresponed parameter set.
            log_name = 'logfile_for_mlp/log_' + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
            file_error = open(log_name, 'w+')
            para = open('../parameters/tmp_para', 'r').read()
            file_error.write(para + '\n')
            file_error.write('\n' + '****' * 50, '\n')
            file_error.write(str(traceback.format_exc()))
            file_error.close()
            err+=1



    f=tmp/(5-err)
    return f


CROSSOVER_PROBABILITY = ContinuousSpace([0,1],'CROSSOVER_PROBABILITY')
MAX_GENOME_LENGTH = OrdinalSpace([100,1000],'MAX_GENOME_LENGTH')
MAX_INIT_TREE_DEPTH = OrdinalSpace([5,15],'MAX_INIT_TREE_DEPTH')
MAX_TREE_DEPTH = OrdinalSpace([10,100],'MAX_TREE_DEPTH')
TOURNAMENT_SIZE = OrdinalSpace([1,50],'TOURNAMENT_SIZE')
N = NominalSpace(['OK'], 'N')

search_space = CROSSOVER_PROBABILITY + MAX_GENOME_LENGTH + MAX_INIT_TREE_DEPTH + MAX_TREE_DEPTH + TOURNAMENT_SIZE + N
# search_space = CROSSOVER_PROBABILITY + TOURNAMENT_SIZE + N


model = RandomForest(levels=search_space.levels)

opt = BO(search_space, obj_func, model, max_iter=n_step,
         n_init_sample=n_init_sample,
         n_point=1,        # number of the candidate solution proposed in each iteration
         n_job=1,          # number of processes for the parallel execution
         minimize=True,
         eval_type=eval_type, # use this parameter to control the type of evaluation
         verbose=True,     # turn this off, if you prefer no output
         optimizer='MIES')

xopt, fitness, stop_dict = opt.run()

print('xopt: {}'.format(xopt))
print('fopt: {}'.format(fitness))
print('stop criteria: {}'.format(stop_dict))
