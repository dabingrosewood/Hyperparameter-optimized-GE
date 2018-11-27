import os,traceback
import numpy as np
import re
#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
import datetime



filename='../parameters/mlp_test/regression.txt'
# filename='../parameters/mlp_test/string_match.txt'

# The "black-box" objective function
def obj_func(x):
    test_var=x['CROSSOVER_PROBABILITY']
    # success

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

    #build the command
    print('-'*40+'*'*10+'-'*40+'\n')
    print('parameters with \n',open('../parameters/tmp_para','r').read())
    cmd='python3 ponyge.py --parameters tmp_para'


    try:
        p=os.popen(cmd)
        std_out = p.read()

        file_record=open('tmp_fitness','r').read()
        print('current parameter has the fitness of ',file_record)

        return float(file_record)
    except:
        #record the error and corresponed parameter set.
        log_name='logfile_for_mlp/log_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        file_error=open(log_name,'w+')
        para = open('../parameters/tmp_para', 'r').read()
        file_error.write(para+'\n')
        file_error.write('\n'+'****'*50,'\n')
        file_error.write(str(traceback.format_exc()))
        file_error.close()
        return 999


# First we need to define the Search Space
# the search space consists of two continues variable
# one ordinal (integer) variable
# and one categorical.
#here we defined two variables at once using the same lower and upper bounds.
#One with label C_0, and the other with label C_1

CROSSOVER_PROBABILITY = ContinuousSpace([0,1],'CROSSOVER_PROBABILITY')
MAX_GENOME_LENGTH = OrdinalSpace([100,1000],'MAX_GENOME_LENGTH')
MAX_INIT_TREE_DEPTH = OrdinalSpace([1,10],'MAX_INIT_TREE_DEPTH')
MAX_TREE_DEPTH = OrdinalSpace([10,100],'MAX_TREE_DEPTH')
TOURNAMENT_SIZE = OrdinalSpace([1,20],'TOURNAMENT_SIZE')
N = NominalSpace(['OK'], 'N')

#the search space is simply the product of the above variables
# search_space = CROSSOVER_PROBABILITY * N
search_space = CROSSOVER_PROBABILITY * MAX_GENOME_LENGTH * MAX_INIT_TREE_DEPTH * MAX_TREE_DEPTH * TOURNAMENT_SIZE * N




#next we define the surrogate model and the optimizer.
model = RandomForest(levels=search_space.levels)
opt = mipego(search_space, obj_func, model,
                 minimize=True,     #the problem is a minimization problem.
                 max_eval=10,      #we evaluate maximum 500 times
                 max_iter=10,      #we have max 500 iterations
                 infill='EI',       #Expected improvement as criteria
                 n_init_sample=10,  #We start with 10 initial samples
                 n_point=1,         #We evaluate every iteration 1 time
                 n_job=1,           #  with 1 process (job).
                 optimizer='MIES',  #We use the MIES internal optimizer.
                 verbose=True, random_seed=None)


#and we run the optimization.
incumbent, stop_dict = opt.run()






