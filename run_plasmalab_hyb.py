import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal
import time

from utils.general_utils import loadpklz, savepklz, evaluate_single_state, temp_seed

import models

# model = 'DetectingPedestrian'
# model = 'Merging'
# model = 'Slplatoon'
# model = 'HighwayEnvNEWScen'
# model = 'HighwayEnvNewNewScen'
# model = 'test_model'
# model = 'MergeEnvScen'
# model = 'IntersectionEnvScen'
# model = 'TestModel'
model = 'TestModelHybrid'
num_modes = 1
dimension = 10


if model == 'TestModelHybrid':
    nimc = models.__dict__[model](nc=num_modes, d=dimension)
else:
    nimc = models.__dict__[model]()
# nimc = models.__dict__[model]()
# budget = int(sys.argv[1])

T = nimc.k
dim = nimc.Theta[1].shape[0]
port = 9100
plasmalab_root = '/home/carla/plasmalab-1.4.4/'

out_file_name = 'output_plasmalab.dat'
out_file = open(out_file_name, 'w')
out_file.write('experiment information: ')
out_file.write('model: ' + model + '\n')
out_file.write('number of modes: ' + str(num_modes) + '\n')
out_file.write('dimension: ' + str(dimension) + '\n')


def get_initial_state(seed):
    with temp_seed(np.abs(seed) % (2**32)):
        rnd_vector = np.random.rand(nimc.Theta[1].shape[0] + 1)
        state0_ =  int(rnd_vector[0] * len(nimc.Theta[0]))
        state0 = rnd_vector[0]
        # state0 = state0_
        state1 = rnd_vector[1:]\
              * (nimc.Theta[1][:,1] - nimc.Theta[1][:,0])\
              + nimc.Theta[1][:,0]

    state_ = [state0_]
    state_ = state_ + state1.tolist()

    state = [state0]
    state = state + state1.tolist()

    return state_

if __name__ == '__main__':
    # time_s = time.time()
    # budget = int(budget / 16)
    #epsilon = 0.01
    delta = 0.01
    tmp_model_name = 'model_%d'%port
    tmp_spec_name = 'spec_%d'%port
    with open(tmp_model_name, 'w') as f:
        f.write('%d %d %d'%(dim, T+1, port))
    with open(tmp_spec_name, 'w') as f:
        f.write('F<=1000 (T<=%d & US>0)'%T)

    num_exp = 10
    budgets = [10]
    budgets = budgets + (20 * np.arange(1, 6)).tolist()
    budgets = budgets + (200 * np.arange(1, 5)).tolist()
    budgets = budgets + (1000 * np.arange(1, 51)).tolist()
    
    out_file.write('number of experiments: ' + str(num_exp) + '\n')
    out_file.write('budgets: ' + str(budgets) + '\n')
    out_file.write('-------------------------------------------------------\n')

    total_num_queries = []
    total_running_times = []
    total_mean_optimal_values = []
    total_std_optimal_values = []
    
    for budget in budgets:
        print("--------------------------------------------------------------------------------")
        num_queries_for_each_budget = []
        results_for_each_budget = []
        running_times = []
        
        for e in range(num_exp):
            print("-------------------------------")        
            time_s = time.time()
            #delta = 2 / np.exp((budget*0.8)*2*(epsilon**2))
            epsilon = np.sqrt(np.log(2/delta)/np.log(np.e)/2/(budget*0.8))
            print(epsilon, delta, budget)
            # The os.setsid() is passed in the argument preexec_fn so
            # it's run after the fork() and before  exec() to run the shell.
            # _simulator = subprocess.Popen('python3 simulator.py --model %s --port %d --save data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.pklz'%(model, port, model, epsilon, delta, budget), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
            _simulator = subprocess.Popen('python3 simulator.py --model %s --port %d --save data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.pklz'%(model, port, model, epsilon, delta, budget), shell=True, preexec_fn=os.setsid)
            output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r '+tmp_spec_name+':bltl -a smartsampling -A"Maximum"=True -A"Epsilon"=%lf -A"Delta"=%lf -A"Budget"=%d'%(epsilon, delta, budget), universal_newlines=True, shell=True)
            os.killpg(os.getpgid(_simulator.pid), signal.SIGUSR1)  # Send the signal to all the process groups
            time.sleep(3)
            os.killpg(os.getpgid(_simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
            os.killpg(os.getpgid(_simulator.pid), signal.SIGKILL)  # Send the signal to all the process groups
            with open('data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.txt'%(model, epsilon, delta, budget), 'w') as f:
                f.write(output)

            with open('data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.txt'%(model, epsilon, delta, budget), 'r') as f:
                output = f.readlines()

            # print(''.join(output[-6:]))
            # Strips the newline character
            output = [line.strip() for line in output]
            seeds = output[1:-6]
            num_query = len(seeds)
            print('PlasmaLab finished (%d queries).'%num_query)
            final_iter = [int(line.split(' ')[3]) for line in seeds[-budget+10::]]
            final_iter = list(set(final_iter))
            original_result = float(output[-2].split('|')[2])
            # print(original_results)
            result_map = loadpklz('data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.pklz'%(model, epsilon, delta, budget))
            final_states = [get_initial_state(seed) for seed in final_iter]
            tmp_results = []
            for seed in final_iter:
                result = result_map[seed]
                result = sum(result) / len(result)
                tmp_results.append(result)
            initial_states = final_states[np.argmax(tmp_results)]
            np.random.seed(1024)
            print('Running final MC eval.')
            result = evaluate_single_state(nimc, initial_states, nimc.k, mult=200)
            # result = np.max(tmp_results)
            # print(tmp_results)
            print({'initial_state':initial_states, 'result':result, 'num_query':num_query, 'original_result':original_result})
            time_e = time.time()
            print('Running time:', time_e-time_s)
            
            num_queries_for_each_budget.append(num_query)
            results_for_each_budget.append(result)
            running_times.append(time_e-time_s)
        
        total_running_times.append(np.mean(np.array(running_times)))
        total_num_queries.append(int(np.mean(np.array(num_queries_for_each_budget))))
        total_mean_optimal_values.append(np.mean(np.array(results_for_each_budget)))
        total_std_optimal_values.append(np.std(np.array(results_for_each_budget)))
        
        out_file.write('budget: ' + str(int(np.mean(np.array(num_queries_for_each_budget)))) + '\n')
        out_file.write('result: ' + str(np.mean(np.array(results_for_each_budget))) + '\n')
        out_file.write('-------------------------------------------------------\n')
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("number of queries: ", total_num_queries)
    print("final results mean: ", total_mean_optimal_values)
    print("final results std: ", total_std_optimal_values)
    out_file.write('budgets: ' + str(budgets) + '\n')
    out_file.write('number of queries: ' + str(total_num_queries) + '\n')
    out_file.write('results mean: ' + str(total_mean_optimal_values) + '\n')
    out_file.write('results std: ' + str(total_std_optimal_values) + '\n')
        