import numpy as np
import math
import sys

sys.path.append('..')
from SiMC import SiMC


# -----------------------------------------------------------------------------
def reward(init):
    # init belongs to set Theta or B
    x_0 = init[0]  # mode
    x = init[1:]  # state

    if x_0 == 0:
        a = 0
    elif x_0 != 0:
        a = 0.5
    g = math.sin(12 * x[0]) * math.sin(27 * x[0]) / 2 + 0.5
    for i in range(len(x) - 1):
        g = g - x[i + 1] ** 2
    y = g + np.random.normal(0, 0.1, 1) - a  # noisy observation

    return y


# -----------------------------------------------------------------------------
__all__ = ['Synthetic']


class Synthetic(SiMC):
    def __init__(self, nc, d, k=0):
        super(Synthetic, self).__init__()

        # define the search-space
        modes = []
        for i in range(nc):
            modes.append(i)
        search_space = [modes]
        for i in range(d):
            search_space.append([0, 1])

        self.set_Theta(search_space)
        self.set_k(k)

    def is_unsafe(self, init: {SiMC}):
        return reward(init)

# -----------------------------------------------------------------------------
