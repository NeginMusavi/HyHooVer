import numpy as np
import math
import sys
sys.path.append('..')
from NiMC import NiMC


def simulate(init):

    x_0 = init[0]
    x = init[1:]

    if x_0 == 0:
        m = 0
    elif x_0 != 0:
        m = 0.5

    f = math.sin(12*x[0]) * math.sin(27*x[0]) / 2 + 0.5

    g = f

    for i in range(len(x)-1):
        g = g - x[i+1] ** 2

    g = g + np.random.normal(0, 0.1, 1) - m
    return g


__all__ = ['Synthetic']


class Synthetic(NiMC):
    def __init__(self, nc, d, k=0):
        super(Synthetic, self).__init__()
        categories = []
        for i in range(nc):
            categories.append(i)

        search_space = []
        search_space.append(categories)
        for i in range(d):
            search_space.append([0, 1])

        self.set_Theta(search_space)

        self.set_k(k)

    def is_unsafe(self, state):
        return simulate(state)

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return
