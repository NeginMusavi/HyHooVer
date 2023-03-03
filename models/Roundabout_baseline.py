import gym

# import sys
# sys.path.insert(1, '/home/carla/workspace/highway-env')


import highway_env
import numpy as np
import sys

sys.path.append('..')
from NiMC import NiMC

# matplotlib inline

# other parameters
# ----------------------------
time_horizon = 10

# create environment
# ---------------------------------------------
env = gym.make('circular-v0')

# reset environment
# ---------------------------------------------
env.reset()


# pprint.pprint(env.config)


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


def simulate(init):
    unsafe = False
    env.reset()

    x_0, x_1, x_2, x_3, x_4 = init
    set_initial_state(env, x_0, x_1, x_2, x_3, x_4)

    # start simulation
    # ---------------------------------------------
    for t in range(time_horizon):
        action = env.action_type.actions_indexes["IDLE"]
        # obs, reward, done, info = env.step(action)
        obs, reward, done, truncated, info = env.step(action)
        # env.render()
        if env.road.vehicles[0].crashed:
            unsafe = True
            break
    return unsafe


__all__ = ['Roundabout_baseline']


class Roundabout_baseline(NiMC):
    def __init__(self, category_index, k=0):
        super(Roundabout_baseline, self).__init__()


        category_list = [("nxr", "exr", "exr"), ("exr", "exr", "exr"), ("exr", "exr", "nxr"),
                         ("exr", "nxr", "exr"), ("exr", "nxr", "nxr"), ("exr", "wxr", "exr"), ("exr", "wxr", "nxr"),
                         ("exr", "sxr", "exr"), ("exr", "sxr", "nxr"), ("exr", "exr", "wxr"), ("exr", "exr", "sxr"),
                         ("exr", "wxr", "wxr"), ("exr", "wxr", "sxr"), ("exr", "sxr", "wxr"), ("exr", "sxr", "sxr")]

        category = category_list[category_index]

        categories = []
        categories.append(category)

        search_space = []
        search_space.append(categories)
        search_space.append([5.05, 5.21])
        search_space.append([9.35, 9.41])
        search_space.append([6.79, 6.95])
        search_space.append([8.89, 9.05])

        self.set_Theta(search_space)
        self.set_k(k)

    def is_unsafe(self, state):
        # print("I am here")
        return simulate(state)

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state