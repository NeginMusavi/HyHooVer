import gym

# import sys
# sys.path.insert(1, '/home/carla/workspace/highway-env')
import highway_env
import numpy as np
import sys

sys.path.append('..')
from SiMC import SiMC

# ----------------------------------------------------------------------------------------------------------------------
time_horizon = 10
env = gym.make('circular-v0')
env.reset()


def set_initial_state(env, x_0, x_1, x_2, x_3, x_4):
    for k in range(len(env.road.vehicles)):
        env.road.vehicles[k].position = env.road.vehicles[k].position + np.random.normal(0, 0.1, 1)[0]

    env.road.vehicles[0].plan_route_to(x_0[0])
    env.road.vehicles[1].plan_route_to(x_0[1])
    env.road.vehicles[2].plan_route_to(x_0[2])

    env.road.vehicles[0].speed = x_1
    if x_0[0] == "nxr" and x_0[2] == "nxr":
        env.road.vehicles[0].target_speed = x_2 + 0.5
    else:
        env.road.vehicles[0].target_speed = x_2

    env.road.vehicles[2].speed = x_3
    env.road.vehicles[2].target_speed = x_4


def reward(init):
    unsafe = False
    env.reset()

    x_0, x_1, x_2, x_3, x_4 = init
    set_initial_state(env, x_0, x_1, x_2, x_3, x_4)

    # start simulation
    # ---------------------------------------------
    for t in range(time_horizon):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, info = env.step(action)
        # env.render()
        if env.road.vehicles[0].crashed:
            unsafe = True
            break
    return unsafe


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ['Roundabout_']


class Roundabout_(SiMC):
    def __init__(self, k=0):
        super(Roundabout_, self).__init__()

        modes = [("nxr", "exr", "exr"), ("exr", "exr", "exr"), ("exr", "exr", "nxr"),
                 ("exr", "nxr", "exr"), ("exr", "nxr", "nxr"), ("exr", "wxr", "exr"), ("exr", "wxr", "nxr"),
                 ("exr", "sxr", "exr"), ("exr", "sxr", "nxr"), ("exr", "exr", "wxr"), ("exr", "exr", "sxr"),
                 ("exr", "wxr", "wxr"), ("exr", "wxr", "sxr"), ("exr", "sxr", "wxr"), ("exr", "sxr", "sxr")]
        green_car_init_speed = [5.01, 5.11]
        green_car_target_speed = [9.31, 9.41]
        blue_car_init_speed = [6.8, 6.99]  # the blue car which is initially located at the most left part of screen
        blue_car_target_speed = [8.99, 9.09]  # the blue car which is initially located at the most left part of screen
        self.set_Theta(
            [modes, green_car_init_speed, green_car_target_speed, blue_car_init_speed, blue_car_target_speed])

        self.set_k(k)

    def is_unsafe(self, init):
        return reward(init)
