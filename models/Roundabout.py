import gym

# import sys
# sys.path.insert(1, '/home/carla/workspace/highway-env')


import highway_env
import numpy as np
import sys

sys.path.append('..')
from NiMC import NiMC


# other parameters
# ----------------------------
time_horizon = 10

# create environment
# ---------------------------------------------
env = gym.make('circular-v0')

# reset environment
# ---------------------------------------------
obs, info = env.reset()


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
    obs, info = env.reset()

    x_0, x_1, x_2, x_3, x_4 = init
    set_initial_state(env, x_0, x_1, x_2, x_3, x_4)

    # start simulation
    # ---------------------------------------------
    for t in range(time_horizon):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        # env.render()
        if env.road.vehicles[0].crashed:
            unsafe = True
            break
    return unsafe


__all__ = ['Roundabout']


class Roundabout(NiMC):
    def __init__(self, k=0):
        super(Roundabout, self).__init__()

        self.set_Theta([[("nxr", "exr", "exr"), ("exr", "exr", "exr"), ("exr", "exr", "nxr"),
                         ("exr", "nxr", "exr"), ("exr", "nxr", "nxr"), ("exr", "wxr", "exr"), ("exr", "wxr", "nxr"), 
                         ("exr", "sxr", "exr"), ("exr", "sxr", "nxr"), ("exr", "exr", "wxr"), ("exr", "exr", "sxr"), 
                         ("exr", "wxr", "wxr"), ("exr", "wxr", "sxr"), ("exr", "sxr", "wxr"), ("exr", "sxr", "sxr")], 
                         [5.05, 5.21], [9.35, 9.41], [6.79, 6.95], [8.89, 9.05]])

        
        self.set_k(k)

    def is_unsafe(self, state):
        # print("I am here")
        return simulate(state)

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state