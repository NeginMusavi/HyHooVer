import sys
sys.path.append('..')
import argparse
import numpy as np
import time
import hyhoover
import random

import models

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', metavar='MODEL',
                        default='Synthetic',
                        help='models available: ' +
                             ' | '.join(model_names) +
                             ' (default: Synthetic)')
    parser.add_argument('--args', nargs='+', type=int,
                        help='<Optional> this can be used to pass special arguments to the model (for instance for '
                             'Synthetic example I use args to pass number of modes and the dimension of the states).')
    parser.add_argument('--nRuns', type=int, default=1, help='number of repetitions. (default: 1)')
    parser.add_argument('--budget', type=int, default=int(100),
                        help='sampling budget for total number of simulations (including final evaluation). (default: '
                             '1000)')
    parser.add_argument('--rho_max', type=float, default=0.95, help='smoothness parameter. (default: 0.95)')
    parser.add_argument('--sigma', type=float,
                        help='<Optional> sigma parameter used in UCB. If not specified, it will be sqrt('
                             '0.5*0.5/batch_size).')
    parser.add_argument('--nHOOs', type=int, default=1, help='number of HyHOO instances to use. (default: 1)')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size parameter. (default: 10)')
    parser.add_argument('--output', type=str, default='output_synthetic.dat',
                        help='file name to save the results. (default: ./output_synthetic.dat)')
    parser.add_argument('--seed', type=int, default=1024, help='random seed for reproducibility. (default: 1024)')
    parser.add_argument('--eval_mult', type=int, default=100, help='sampling budget for final evaluation by '
                                                                   'Monte_carlo simulations. (default: 100)')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.set_defaults(eval=False)
    parser.add_argument('--init', nargs='+', type=float,
                        help='<Optional> this can be used to evaluate a specific initial state or parameter.')
    parser.add_argument('--rnd_sampling', dest='rnd_sampling', action='store_true', help='<Optional> this can be used '
                                                                                         'to specify whether to '
                                                                                         'sample from the geometric '
                                                                                         'center of a region or '
                                                                                         'sample randomly. (default: '
                                                                                         'False)')
    parser.set_defaults(rnd_sampling=False)
    args = parser.parse_args()

    out_file = open(args.output, 'w')
    out_file.write('experiment information: ')
    out_file.write('model: ' + args.model + '\n')
    out_file.write('special args: ' + str(args.args) + '\n')
    out_file.write('number of experiments: ' + str(args.nRuns) + '\n')
    out_file.write('total sampling budget: ' + str(args.budget) + '\n')
    out_file.write('evaluation budget: ' + str(args.eval_mult) + '\n')
    out_file.write('number of HOO instances: ' + str(args.nHOOs) + '\n')
    out_file.write('batch size: ' + str(args.batch_size) + '\n')
    out_file.write('random sampling: ' + str(args.rnd_sampling) + '\n')
    out_file.write('rho max: ' + str(args.rho_max) + '\n')
    out_file.write('seed: ' + str(args.seed) + '\n')
    out_file.write('-------------------------------------------------------\n')

    num_exp = args.nRuns

    # Calculate parameter sigma for UCB.
    # sqrt(0.5*0.5/args.batch_size) is a valid parameter for any model.
    # The user can also pass smaller sigma parameters to encourage the
    # algorithm to explore deeper in the tree.
    if args.sigma is None:
        sigma = np.sqrt(0.5 * 0.5 / args.batch_size)
    else:
        sigma = args.sigma

    rho_max = args.rho_max

    total_budgets = [2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
                     200000, 300000, 400000, 500000, 600000, 700000]  # 10 modes + 10D

    total_num_queries = []
    total_num_nodes = []
    total_running_times = []
    total_memory_usages = []
    total_mean_optimal_values = []
    total_std_optimal_values = []
    total_optimal_xs = []
    total_depths = []

    fixed_seed = args.seed

    s_ = 0
    for b in total_budgets:
        s_ = s_ + 1

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("budget:", b)

        running_times_ = []
        memory_usages_ = []
        depths_ = []
        optimal_xs_ = []
        optimal_values_ = []
        num_nodes_ = []
        n_queries_ = []
        n_nodes_ = []

        budget_for_each_HOO = b / args.args[0] / args.nHOOs - args.eval_mult
        print(budget_for_each_HOO)
        n_steps = 1

        fixed_seed = args.seed

        for _n in range(num_exp):

            print("-------------------------------------------------------")
            print("experiment:", (_n + 1))
            out_file.write('-------------------------------------------------------\n')
            out_file.write('experiment: ' + str(_n + 1) + '\n')
            out_file.write('-------------------------------------------------------\n')

            start_time = time.time()

            # set random seed for reproducibility
            random.seed(fixed_seed + _n + s_)
            np.random.seed(fixed_seed + _n + s_)

            overall_best_mean_values_after_comparison = [None] * n_steps
            overall_best_cells_after_comparison = [None] * n_steps
            overall_best_points_after_comparison = [None] * n_steps
            depth_after_comparison = [None] * n_steps
            max_values = [-1000] * n_steps
            for c in range(args.args[0]):

                print("--------------------------------------")
                print("-----category number:-----", str(c + 1))
                print("--------------------------------------")
                out_file.write('-----------------------------\n')
                out_file.write('-----category number:-----' + str(c + 1) + '\n')
                out_file.write('-----------------------------\n')

                if args.model == 'Synthetic_baseline':
                    if args.args is None:
                        raise ValueError('Please specify the s parameter using --args')
                    model = models.__dict__[args.model](mode_index=c, nc=args.args[0], d=args.args[1])
                if args.model == 'CircularEnvScen_baseline':
                    if args.args is None:
                        raise ValueError('Please specify the s parameter using --args')
                    model = models.__dict__[args.model](category_index=c)

                simc = model
                try:
                    # call hoover.estimate_max_probability with the model and parameters
                    memory_usage, overall_best_cells, overall_best_points, depth, overall_best_mean_values, nq = \
                        hyhoover.estimate_max_probability(simc, args.nHOOs, rho_max, sigma, budget_for_each_HOO,
                                                        args.batch_size, args.rnd_sampling,
                                                        out_file,
                                                        eval_mult=args.eval_mult)
                except AttributeError as e:
                    print(e)
                    continue

                for b in range(min(len(max_values), len(overall_best_mean_values))):
                    if overall_best_mean_values[b] >= max_values[b]:
                        max_values[b] = overall_best_mean_values[b]
                        overall_best_cells_after_comparison[b] = overall_best_cells[b]
                        overall_best_points_after_comparison[b] = overall_best_points[b]
                        depth_after_comparison[b] = depth[b]

            overall_best_mean_values_after_comparison = max_values

            end_time = time.time()
            running_time = end_time - start_time

            running_times_.append(running_time)
            memory_usages_.append(memory_usage / 1024.0 / 1024.0)
            optimal_values_.append(np.array(overall_best_mean_values_after_comparison))
            optimal_xs_.append(overall_best_points_after_comparison)
            depths_.append(depth_after_comparison)
            n_queries_.append(nq[0])
            n_nodes_.append(nq[0])

        running_times = np.mean(np.array(running_times_))
        memory_usages = np.mean(np.array(memory_usages_))
        depths = np.mean(np.array(depths_))
        optimal_xs = optimal_xs_
        mean_optimal_values = np.mean(np.array(optimal_values_))
        std_optimal_values = np.std(np.array(optimal_values_))
        n_queries = np.mean(np.array(n_queries_))
        n_nodes = np.mean(np.array(n_nodes_))

        out_file.write('num_queries: ' + str(n_queries) + '\nnum_nodes: ' + str(n_nodes) + '\nrunning_times: ' + str(
            running_times) + '\nmemory_usages: ' + str(
            memory_usages) + '\noptimal_values_mean: ' + str(mean_optimal_values) + '\noptimal_values_std: ' + str(
            std_optimal_values) + '\noptimal_xs: ' + str(
            optimal_xs) + '\ndepths: ' + str(depths))
        out_file.write('-------------------------------------------------------\n')

        total_num_queries.append((n_queries + args.eval_mult) * args.nHOOs * args.args[0])
        total_num_nodes.append(n_nodes)
        total_mean_optimal_values.append(mean_optimal_values)
        total_std_optimal_values.append(std_optimal_values)
        total_optimal_xs.append(optimal_xs)
        total_running_times.append(running_times)
        total_memory_usages.append(memory_usages)
        total_depths.append(depths)

    print('===================================================================')
    print('===================================================================')
    print('Final Results:')
    print('budget: ' + str(total_budgets))
    print('running time (s):' + str(total_running_times))
    print('memory usage (MB):' + str(total_memory_usages))
    print('n_nodes: ' + str(total_num_nodes))
    print('actual n_queries: ' + str(total_num_queries))
    print('optimal_values_mean:', str(total_mean_optimal_values))
    print('optimal_values_std:', str(total_std_optimal_values))
    print('optimal_xs: ' + str(total_optimal_xs))
    print('depth: ' + str(total_depths))

    out_file.write('=======================================================\n')
    out_file.write('=======================================================\n')
    out_file.write('Final Results: ' + '\n')
    out_file.write('=======================================================\n')
    out_file.write(
        'budgets: ' + str(total_budgets) + '\nnum_queries: ' + str(total_num_queries) + '\nnum_nodes: ' + str(
            total_num_nodes) + '\nrunning_times: ' + str(total_running_times) + '\nmemory_usages: ' + str(
            total_memory_usages) + '\noptimal_values_mean: ' + str(
            total_mean_optimal_values) + '\noptimal_values_std: ' + str(
            total_std_optimal_values) + '\noptimal_xs: ' + str(
            total_optimal_xs) + '\ndepths: ' + str(total_depths))
    out_file.close()