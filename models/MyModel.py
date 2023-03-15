import numpy as np
import sys
sys.path.append('..')
from SiMC import SiMC


def reward(init):

    # init belongs to set Theta/B
    x_0 = init[0]  # mode
    x = init[1:]  # state

    # define reward or noisy observation which depends on x_0 and x
    y = 0

    return y

# ---------------------------------------------------------------------------------------------------------------------
__all__ = ['MyModel']
class MyModel(SiMC):
    def __init__(self, k=0):
        super(MyModel, self).__init__()

        # define list of modes
        modes = [0, 1, 2]
        # define intervals to which parameters/initial states belong to
        parameter1 = [-5, 1]
        parameter2 = [0, 1]
        # create search-space/state-space
        search_space = [modes, parameter1, parameter2]

        # define set Theta/B
        self.set_Theta(search_space)
        self.set_k(k)

    def is_unsafe(self, state):
        return reward(state)
