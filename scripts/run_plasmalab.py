import sys

sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal
import time

from utils.general_utils import loadpklz, savepklz, evaluate_single_state, temp_seed

import models

model = 'BrokenLidar'
# model = 'Slplatoon'
# model = 'Roundabout'

num_modes = 1
dimension = 1

if model == 'TestModelHybrid':
    nimc = models.__dict__[model](nc=num_modes, d=dimension)
else:
    nimc = models.__dict__[model]()

T = nimc.k
dim = nimc.Theta[1].shape[0]
port = 9100
plasmalab_root = '/home/carla/plasmalab-1.4.4/'

out_file_name = 'output_plasmalab_BrokenLidar.dat'
out_file = open(out_file_name, 'w')
out_file.write('experiment information: ')
out_file.write('model: ' + model + '\n')
out_file.write('number of modes: ' + str(num_modes) + '\n')
out_file.write('dimension: ' + str(dimension) + '\n')


def get_initial_state(seed):
    with temp_seed(np.abs(seed) % (2 ** 32)):
        rnd_vector = np.random.rand(nimc.Theta[1].shape[0] + 1)
        state0 = float(int(rnd_vector[0] * len(nimc.Theta[0])))
        # state0 = rnd_vector[0]
        state1 = rnd_vector[1:] \
                 * (nimc.Theta[1][:, 1] - nimc.Theta[1][:, 0]) \
                 + nimc.Theta[1][:, 0]

    state = [state0]
    state = state + state1.tolist()

    return state


if __name__ == '__main__':
    # epsilon = 0.01
    delta = 0.01
    tmp_model_name = 'model_%d' % port
    tmp_spec_name = 'spec_%d' % port
    with open(tmp_model_name, 'w') as f:
        f.write('%d %d %d' % (dim, T, port))
    with open(tmp_spec_name, 'w') as f:
        f.write('F<=1000 (T<=%d & US>0)' % T)

    num_exp = 10
    budgets = [22500, 25000, 27500, 30000]

    out_file.write('number of experiments: ' + str(num_exp) + '\n')
    out_file.write('budgets: ' + str(budgets) + '\n')
    out_file.write('-------------------------------------------------------\n')
    print(budgets)
    num_queries_for_all_budgets = []
    mean_results_for_all_budgets = []
    std_results_for_all_budgets = []
    running_time_for_all_budgets = []

    for budget in budgets:
        print("--------------------------------------------------------------------------------")
        num_queries_for_each_budget = []
        results_for_each_budget = []
        running_time_for_each_budget = []

        for e in range(num_exp):
            print("-------------------------------")
            time_s = time.time()
            # delta = 2 / np.exp((budget*0.8)*2*(epsilon**2))
            epsilon = np.sqrt(np.log(2 / delta) / np.log(np.e) / 2 / (budget * 0.8))
            print(epsilon, delta, budget)
            # The os.setsid() is passed in the argument preexec_fn so
            # it's run after the fork() and before  exec() to run the shell.
            # _simulator = subprocess.Popen('python3 simulator.py --model %s --port %d --save data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.pklz'%(model, port, model, epsilon, delta, budget), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
            _simulator = subprocess.Popen(
                'python3 simulator.py --model %s --port %d --save data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.pklz' % (
                model, port, model, epsilon, delta, budget), shell=True, preexec_fn=os.setsid)
            output = subprocess.check_output(
                plasmalab_root + '/plasmacli.sh launch -m ' + tmp_model_name + ':PythonSimulatorBridge -r ' + tmp_spec_name + ':bltl -a smartsampling -A"Maximum"=True -A"Epsilon"=%lf -A"Delta"=%lf -A"Budget"=%d' % (
                epsilon, delta, budget), universal_newlines=True, shell=True)
            os.killpg(os.getpgid(_simulator.pid), signal.SIGUSR1)  # Send the signal to all the process groups
            time.sleep(3)
            os.killpg(os.getpgid(_simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
            os.killpg(os.getpgid(_simulator.pid), signal.SIGKILL)  # Send the signal to all the process groups
            with open('data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.txt' % (model, epsilon, delta, budget), 'w') as f:
                f.write(output)

            with open('data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.txt' % (model, epsilon, delta, budget), 'r') as f:
                output = f.readlines()

            # print(''.join(output[-6:]))
            # Strips the newline character
            output = [line.strip() for line in output]
            seeds = output[1:-6]
            num_query = len(seeds)
            print('PlasmaLab finished (%d queries).' % num_query)
            final_iter = [int(line.split(' ')[3]) for line in seeds[-budget + 10::]]
            final_iter = list(set(final_iter))
            original_result = float(output[-2].split('|')[2])
            # print(original_results)
            result_map = loadpklz(
                'data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d.pklz' % (model, epsilon, delta, budget))
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
            print({'initial_state': initial_states, 'result': result, 'num_query': num_query,
                   'original_result': original_result})
            time_e = time.time()
            print('Running time:', time_e - time_s)

            num_queries_for_each_budget.append(num_query)
            results_for_each_budget.append(result)
            running_time_for_each_budget.append(time_e - time_s)

        num_queries_for_all_budgets.append(int(np.mean(np.array(num_queries_for_each_budget))))
        mean_results_for_all_budgets.append(np.mean(np.array(results_for_each_budget)))
        std_results_for_all_budgets.append(np.std(np.array(results_for_each_budget)))
        running_time_for_all_budgets.append(np.mean(np.array(running_time_for_each_budget)))
        out_file.write('budget: ' + str(int(np.mean(np.array(num_queries_for_each_budget)))) + '\n')
        out_file.write('mean result: ' + str(np.mean(np.array(results_for_each_budget))) + '\n')
        out_file.write('std result: ' + str(np.std(np.array(results_for_each_budget))) + '\n')
        out_file.write('running times: ' + str(np.mean(np.array(running_time_for_each_budget))) + '\n')
        out_file.write('-------------------------------------------------------\n')

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("number of queries: ", num_queries_for_all_budgets)
    print("final mean results: ", mean_results_for_all_budgets)
    print("final std results: ", std_results_for_all_budgets)
    print("running times: ", running_time_for_all_budgets)
    out_file.write('number of queries: ' + str(num_queries_for_all_budgets) + '\n')
    out_file.write('mean results: ' + str(mean_results_for_all_budgets) + '\n')
    out_file.write('std results: ' + str(std_results_for_all_budgets) + '\n')
    out_file.write('running times: ' + str(running_time_for_all_budgets) + '\n')
