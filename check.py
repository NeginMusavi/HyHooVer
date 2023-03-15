# This code is to run HyHooVer algorithm on scenarios coded in python.
# It can run the algorithm for a given "sampling budget".
# It will the result of verification once the budget is exhausted.

import argparse
import numpy as np
import time
import hyhoover
import random
import models

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

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

    if args.model == 'Synthetic':
        if args.args is None:
            raise ValueError('Please specify the s parameter using --args')
        model = models.__dict__[args.model](nc=args.args[0], d=args.args[1])
    else:
        model = models.__dict__[args.model]()

    if args.eval:
        value = 0
        for _ in range(args.eval_mult):
            reward = model(np.array(args.init), model.k)
            value = value + reward
        value = value / args.eval_mult
        print('prob: %.5f' % value)
        exit()

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

    fixed_seed = args.seed

    running_times = []
    memory_usages = []
    depths = []
    optimal_xs = []
    optimal_values = []
    n_nodes = []
    n_queries = []

    budget_for_each_HOO = (args.budget - args.eval_mult * args.nHOOs) / args.nHOOs

    for _n in range(num_exp):

        print("-----------------------------------------------------------------")
        print("experiment:", (_n + 1))
        out_file.write('-------------------------------------------------------\n')
        out_file.write('experiment: ' + str(_n + 1) + '\n')
        out_file.write('-------------------------------------------------------\n')

        start_time = time.time()

        # set random seed for reproducibility
        random.seed(fixed_seed + _n)
        np.random.seed(fixed_seed + _n)
        SiMC = model

        try:
            # call hoover.estimate_max_probability with the model and parameters for each experiment
            memory_usage, overall_best_cells, overall_best_points, depths_in_tree, overall_best_mean_values, nq = \
                hyhoover.estimate_max_probability(SiMC, args.nHOOs, rho_max, sigma, budget_for_each_HOO,
                                                  args.batch_size, args.rnd_sampling, out_file,
                                                  eval_mult=args.eval_mult)
        except AttributeError as e:
            print(e)
            continue

        end_time = time.time()
        running_time = end_time - start_time

        print("Done!")
        print("Best Result:", overall_best_mean_values)

        running_times.append(running_time)
        memory_usages.append(memory_usage / 1024.0 / 1024.0)
        optimal_values.append(np.array(overall_best_mean_values))
        optimal_xs.append(overall_best_points)
        depths.append(depths_in_tree)

        # Get the real number of queries from the model object.
        # It may be a little different from the budget.
        # n_queries = SiMC.cnt_queries
        n_queries.append((nq[0][0] + args.eval_mult) * args.nHOOs)
        n_nodes.append(nq[0][0])

    print('===================================================================')
    print('===================================================================')
    print('Final Results:')
    print('===================================================================')
    print('===================================================================')
    print('sampling budget: ' + str(args.budget))
    print('running time (s): %.2f +/- %.3f' % (np.mean(np.array(running_times)), np.std(np.array(running_times))))
    print('memory usage (MB): %.2f +/- %.3f' % (np.mean(np.array(memory_usages)), np.std(np.array(memory_usages))))
    print('depth in tree: %.2f +/- %.3f' % (np.mean(np.array(depths)), np.std(np.array(depths))))
    print('number of nodes: ' + str(n_nodes))
    print('number of queries: ' + str(n_queries))
    print('optimal values: %.2f +/- %.3f' % (np.mean(np.array(optimal_values)), np.std(np.array(optimal_values))))
    print('optimal xs: ' + str(optimal_xs))

    out_file.write('=======================================================\n')
    out_file.write('=======================================================\n')
    out_file.write('Final Results: ' + '\n')
    out_file.write('=======================================================\n')
    out_file.write('=======================================================\n')
    out_file.write('num_queries: ' + str(n_queries) + '\nnum_nodes: ' + str(n_nodes) + '\nrunning time (s): %.2f +/- '
                                                                                       '%.3f' % (np.mean(np.array(
        running_times)), np.std(np.array(running_times))) + '\nmemory usage (MB): %.2f +/- %.3f' % (
                   np.mean(np.array(memory_usages)),
                   np.std(np.array(memory_usages))) + '\noptimal values: %.2f +/- %.3f' % (
                   np.mean(np.array(optimal_values)), np.std(np.array(optimal_values))) + '\noptimal_xs: ' + str(
        optimal_xs) + '\ndepth in tree: %.2f +/- %.3f' % (np.mean(np.array(depths)), np.std(np.array(depths))))

    out_file.close()



