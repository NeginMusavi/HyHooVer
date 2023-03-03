import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import llist_generator
from llist import dllist, dllistnode
from NiMC import NiMC

time_hor = 11

prob_close = np.array([0.7,0.15,0.15]) # prob for brake/cruise/speedup when close
prob_fine = np.array([0.15,0.7,0.15])
prob_far = np.array([0.15,0.15,0.7])

v_brake = 1
v_cruise = 4
v_speedup = 7
v_error = 0.1
velocities = [v_brake, v_cruise, v_speedup]

threshold_close = 3
threshold_far = 5

unsafe_rule_close = 0.1

num_cars = 4
initial_separation_between_cars = 20
range_of_initial_set_per_car = 5


def simulate(init):
    # print(init)
    state = np.array(init[1:])

    unsafe = 0.0
    for t in range(time_hor):
        state_current = dllist(state)
        state_old = state_current
        state_current = dllist()

        # update the state
        for car in llist_generator(state_old):
            ita = np.random.randn() * v_error
            #print(ita)
            if car is state_old.first:
                # first car, always cruise
                position = car.value + v_cruise + ita
                state_current.append(position)
                # step_desc += 'first'
            else:
                # not the first car
                distance = car.prev.value - car.value
                p = prob_close if distance < threshold_close else prob_fine if distance < threshold_far else prob_far

                v = np.random.choice(velocities, p = p)
                position = car.value + v + ita
                state_current.append(position)
        state = list(state_current)

        for i in range(1, len(state)):
            if state[i-1] - state[i] < unsafe_rule_close:
                unsafe = 1.0
                return unsafe      
        
    return unsafe

__all__ = ['Slplatoon']


class Slplatoon(NiMC):
    def __init__(self, k=0):
        super(Slplatoon, self).__init__()
        
        state_start = np.array(list(range(num_cars)))[::-1] * initial_separation_between_cars
        state_range = np.array([range_of_initial_set_per_car, ] * num_cars)
        Theta_ = np.stack([state_start, state_start + state_range]).T       

        categories = [0]
        Theta = []
        Theta.append(categories)
        Theta__ = Theta_.tolist()
        for i in range(len(Theta__)):
            Theta.append(Theta__[i])

        print(Theta)

        self.set_Theta(Theta)
        self.set_k(k)

    def is_unsafe(self, state):
        return simulate(state)

    def transition(self, state):
        # assert len(state) == num_cars
        assert len(state) == self.Theta[1].shape[0]
        return state
