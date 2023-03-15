import numpy as np
import sys

sys.path.append('..')
from SiMC import SiMC

# ----------------------------------------------------------------------------------------------------------------------
s = 5e-2  # this parameter controls the sharpness of lidar detection probability around its optimum
time_hor = 10


def lidar_detection_prob(theta, r):
    theta_broken = 0.08
    r_max = 500
    prob = (1 - np.exp(-1.0 * (theta - theta_broken) ** 2 / s)) * ((r - r_max) ** 2 / (r_max ** 2))
    return prob


def reward(init):
    state = np.array(init[1:])

    time_step = 0.25  # 0.25 s
    brake_acc = 8  # m/s^2
    v_error = 0.

    unsafe = 0.0
    for t in range(time_hor):
        if state[0] > 0:
            theta = np.arctan(state[2] / state[0])
            r = np.sqrt(state[2] ** 2 + state[0] ** 2)
            prob = min(max(lidar_detection_prob(theta, r), 0), 1)
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

    return unsafe


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ['BrokenLidar']


class BrokenLidar(SiMC):
    def __init__(self, k=0):
        super(BrokenLidar, self).__init__()

        modes = [0]
        car_pos_range = [55, 100]
        car_v_range = [10, 20]
        ped_pos_range = [3, 7]
        ped_v_range = [1, 2]
        search_space = [modes, car_pos_range, car_v_range, ped_pos_range, ped_v_range]

        self.set_Theta(search_space)
        self.set_k(k)

    def is_unsafe(self, init):
        return reward(init)
