
import numpy as np
from ax import optimize
import math
import time

dim = 4
modes = 1

s = 1e-3
time_hor = 10

car_pos_range = [55, 100]
car_v_range = [10, 20]
ped_pos_range = [3, 7]
ped_v_range = [1, 2]

parameters_ = [
            {
                "name": "x1",
                "type": "range",
                "bounds": car_pos_range,
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": car_v_range,
            },
            {
                "name": "x3",
                "type": "range",
                "bounds": ped_pos_range,
            },
            {
                "name": "x4",
                "type": "range",
                "bounds": ped_v_range,
            },
            {
                "name": "x5",
                "type": "choice",
                "values": [0],
            },
        ]


# this is an unsafe lidar model
def lidar_prob(theta, r):
    theta_broken = 0.08
    r_max = 500
    prob = (1 - np.exp(-1.0 * (theta - theta_broken) ** 2 / s)) * ((r - r_max) ** 2 / (r_max ** 2))
    return prob


def reward(parameterization):

    x = np.array([parameterization.get(f"x{i + 1}") for i in range(4 + 1)])
    state = x[:4]

    time_step = 0.25  # 0.25 s
    brake_acc = 8  # m/s^2
    v_error = 0.

    unsafe = 0.0
    for t in range(time_hor):

        if state[0] > 0:
            theta = np.arctan(state[2] / state[0])
            r = np.sqrt(state[2] ** 2 + state[0] ** 2)
            prob = min(max(lidar_prob(theta, r), 0), 1)
        else:
            prob = 0

        if np.random.rand() < prob:
            brake_distance = state[1] ** 2 / (2 * brake_acc)

            time_to_zero = 100000000 if brake_distance < state[0] else (state[1] - np.sqrt(
                state[1] ** 2 - 2 * brake_acc * state[0])) / brake_acc

            state[2] -= state[3] * time_to_zero
            state[0] = 0.
            state[1] = 0.

        state[0] -= (state[1] + np.random.randn() * v_error * 0.1) * time_step
        state[2] -= (state[3] + np.random.randn() * v_error) * time_step

        if -5 <= state[0] <= 0 and -1 <= state[2] <= 1:
            unsafe = 1.0

    return -1 * unsafe


if __name__ == "__main__":

    num_exp = 10
    budgets = [10, 50, 100, 200, 300, 400, 500]
    eval_mult = 100

    out_file = open("output_botorch_broken_lidar", 'w')
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



