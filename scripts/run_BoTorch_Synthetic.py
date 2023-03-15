import numpy as np
from ax import optimize
import math
import time

dim = 10
modes = 10

parameters_ = [
            {
                "name": "x1",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x3",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x4",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x5",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x6",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x7",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x8",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x9",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x10",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x11",
                "type": "choice",
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            },
        ]


def reward(parameterization):

    x = np.array([parameterization.get(f"x{i+1}") for i in range(10 + 1)])

    x_0 = x[10]
    if x_0 == 0:
        a = 0
    elif x_0 != 0:
        a = 0.5

    g = math.sin(12 * x[0]) * math.sin(27 * x[0]) / 2 + 0.5

    for i in range(len(x) - 2):
        g = g - x[i + 1] ** 2

    y = g + np.random.normal(0, 0.1, 1)[0] - a

    return -1 * y


if __name__ == "__main__":

    num_exp = 10
    budgets = [10, 150, 400, 650, 900, 1150]
    eval_mult = 100

    out_file = open("output_botorch_synthetic", 'w')
    out_file.write('experiment information: ')
    out_file.write('number of experiments: ' + str(num_exp) + '\n')
    out_file.write('total budget: ' + str(budgets) + '\n')
    out_file.write('dimension: ' + str(dim) + '\n')
    out_file.write('modes: ' + str(modes) + '\n')
    out_file.write('----------------------------------------------------------------------------\n')

    total_best_parameters = []
    total_best_values_mean_ = []
    total_best_values_mean = []
    total_best_values_std_ = []
    total_best_values_std = []
    total_running_time = []
    for b_ in budgets:
        out_file.write('----------------------------------------------------------------------------\n')
        out_file.write('budget: ' + str(budgets) + '\n')
        best_parameters = []
        best_values__ = []
        best_values = []
        running_time = []
        for k in range(num_exp):
            out_file.write('------------------------\n')
            out_file.write('experiment: ' + str(k) + '\n')
            start_time = time.time()
            best_parameters_, best_values_, experiment, model = optimize(
                parameters=parameters_,
                evaluation_function=reward,
                minimize=True,
                total_trials=b_,
            )
            end_time = time.time()
            running_time_ = end_time - start_time
            out_file.write('BoTorch_estimation_of_best_parameters: ' + str(best_parameters_) + '\n')
            out_file.write('BoTorch_estimation_of_best_values: ' + str(-1 * best_values_[0]['objective']) + '\n')
            out_file.write('running_time: ' + str(running_time_) + '\n')
            best_parameters.append(best_parameters_)

            m_ = 0
            for t in range(eval_mult):
                m_ = m_ + reward(best_parameters_)
            m_ = m_ / eval_mult

            best_values__.append(-1 * best_values_[0]['objective'])
            best_values.append(-1 * m_)
            running_time.append(running_time_)

        out_file.write('------------------------\n')
        out_file.write('------------------------\n')
        out_file.write('BoTorch_estimation_of_best_parameters: ' + str(best_parameters) + '\n')
        out_file.write('BoTorch_estimation_of_best_values_mean: ' + str(np.mean(np.array(best_values__))) + '\n')
        out_file.write('actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_mean: ' + str(
            np.mean(np.array(best_values))) + '\n')
        out_file.write('BoTorch_estimation_of_best_values_std: ' + str(np.std(np.array(best_values__))) + '\n')
        out_file.write('actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_std: ' + str(
            np.std(np.array(best_values))) + '\n')
        out_file.write('running_time_mean: ' + str(np.mean(np.array(running_time))) + '\n')

        total_best_parameters.append(best_parameters)
        total_best_values_mean_.append(np.mean(np.array(best_values__)))
        total_best_values_mean.append(np.mean(np.array(best_values)))
        total_best_values_std_.append(np.std(np.array(best_values__)))
        total_best_values_std.append(np.std(np.array(best_values)))
        total_running_time.append(np.mean(np.array(running_time)))

        print("---------------------------------------------------")
        print('BoTorch_estimation_of_best_values_mean: ', np.mean(np.array(best_values__)))
        print('actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_mean: ', np.mean(np.array(best_values)))
        print('BoTorch_estimation_of_best_values_std: ', np.std(np.array(best_values__)))
        print('actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_std: ', np.std(np.array(best_values)))
        print('running_time_mean: ', np.mean(np.array(running_time)))

    out_file.write('----------------------------------------------------------------------------------\n')
    out_file.write('----------------------------------------------------------------------------------\n')
    out_file.write('total budget: ' + str(budgets) + '\n')
    out_file.write('total_BoTorch_estimation_of_best_parameters: ' + str(total_best_parameters) + '\n')
    out_file.write('total_BoTorch_estimation_of_best_values_mean: ' + str(total_best_values_mean_) + '\n')
    out_file.write('total_actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_mean: ' + str(
        total_best_values_mean) + '\n')
    out_file.write('total_BoTorch_estimation_of_best_values_std: ' + str(total_best_values_std_) + '\n')
    out_file.write('total_actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_std: ' + str(
        total_best_values_std) + '\n')
    out_file.write('total_running_time_mean: ' + str(total_running_time) + '\n')

    print("Done!")
    print("---------------------------------------------------")
    print('total_BoTorch_estimation_of_best_values_mean: ', total_best_values_mean_)
    print('total_actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_mean: ', total_best_values_mean)
    print('total_BoTorch_estimation_of_best_values_std: ', total_best_values_std_)
    print('total_actual_values_evaluated_at_BoTorch_estimation_of_best_parameters_std: ', total_best_values_std)
    print('total_running_time_mean: ', total_running_time)



