import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import map_to_cube, map_to_bounds, map_cell_to_bounds, map_int_to_discrete_part_in_search_space
import time

# -----------------------------------------------------------------------------
class MFSiMC(object):
    """
        This class is a wrapper for SiMC in order to use MFHOO.
        This class normalizes the domain such that length of each dim after normalizing is 1.
    """

    def __init__(self, SiMC, batch_size, rand_sampling, eval_mult):
        self.SiMC = SiMC
        self.Theta_index_non_const_dims = np.where(SiMC.Theta[1][:, 1] - SiMC.Theta[1][:, 0] > 0)[0]

        def reward_function(initial_state, fidel):
            initial_state = self.get_full_state(initial_state)
            reward = SiMC.simulate(initial_state, fidel)
            return reward

        self.reward_function = reward_function

        self.domain_bounds = SiMC.Theta[1][self.Theta_index_non_const_dims, :]
        self.domain_dim = len(self.Theta_index_non_const_dims)
        self.discrete_set = SiMC.Theta[0]
        self.num_trees = len(SiMC.Theta[0])

        if hasattr(SiMC, 'fidel_cost_function'):
            fidel_cost_function = SiMC.fidel_cost_function
        else:
            fidel_cost_function = lambda z: 1

        self.fidel_cost_function = fidel_cost_function
        self.fidel_bounds = np.array([(1, SiMC.k)])
        self.fidel_dim = 1
        self.opt_fidel_cost = self.cost_single(1)
        self.max_iteration = 200
        self.batch_size = batch_size
        self.rand_sampling = rand_sampling
        self.eval_mult = eval_mult

    # ------------------------------------------------------------------------------

    def get_full_state(self, state):
        _state = np.array(state[1:][0]).reshape(-1)
        state_ = self.SiMC.Theta[1][:, 0].copy()
        if len(_state) == len(self.Theta_index_non_const_dims):
            state_[self.Theta_index_non_const_dims] = _state
        elif len(_state) == len(state_):
            state_[:] = _state
        else:
            raise ValueError('Wrong size')

        full_state = list(state_)
        full_state.insert(0, state[0])
        return full_state

    # -----------------------------------------------------------------------------

    def cost_single_average(self, Z):
        """ Evaluates cost at a single point. """
        t1 = time.time()
        d = self.domain_dim
        # #-------------draw random sample (not midpoint)----------------
        # rnd = random.random()
        # X = np.array([rnd] * d)
        # #--------------------------------------------------------------
        X = np.array([0.5] * d)
        self.eval_at_fidel_single_point_normalised_average(Z, X, self.max_iteration)
        t2 = time.time()
        return t2-t1

    # -----------------------------------------------------------------------------

    def cost_single(self, Z):
        """ Evaluates cost at a single point. """
        return self.eval_fidel_cost_single_point_normalised(Z)

    # -----------------------------------------------------------------------------

    def eval_at_fidel_single_point(self, Z, X):
        """ Evaluates X at the given Z at a single point. """
        Z = np.array(Z).reshape((1, self.fidel_dim))
        X_ = X[1]
        X_ = np.array(X_).reshape((1, self.domain_dim))
        X = (X[0], X_)
        return float(self.reward_function(X, Z))

    # -----------------------------------------------------------------------------

    def eval_fidel_cost_single_point(self, Z):
        """ Evaluates the cost function at a single point. """
        return float(self.fidel_cost_function(Z))

    # -----------------------------------------------------------------------------

    def eval_at_fidel_single_point_normalised(self, Z, X):
        """ Evaluates X at the given Z at a single point using normalised coordinates. """
        Z, X = self.get_unnormalised_coords(Z, X)
        return self.eval_at_fidel_single_point(Z, X)

    # -----------------------------------------------------------------------------

    def eval_at_fidel_single_point_normalised_average(self, Z, X, max_iteration):
        """ Evaluates X at the given Z at a single point using normalised coordinates. """
        Z, X = self.get_unnormalised_coords(Z, X)

        mean_value = 0

        for i in range(max_iteration):
            value = self.eval_at_fidel_single_point(Z, X)
            mean_value = mean_value + value

        mean_value = mean_value/max_iteration

        return mean_value

    # -----------------------------------------------------------------------------

    def eval_fidel_cost_single_point_normalised(self, Z):
        """ Evaluates the cost function at a single point using normalised coordinates. """
        Z, _ = self.get_unnormalised_coords(Z, None)
        return self.eval_fidel_cost_single_point(Z)

    # -----------------------------------------------------------------------------

    def get_normalised_coords(self, Z, X):
        """ Maps points in the original space to the cube. """
        ret_Z = None if Z is None else map_to_cube(Z, self.fidel_bounds)
        ret_X = None if X is None else map_to_cube(X, self.domain_bounds)
        return ret_Z, ret_X

    # -----------------------------------------------------------------------------

    def get_unnormalised_coords(self, Z, X):
        """ Maps points in the cube to the original space. """
        if X is not None:
            X_ = X[1]
            ret_X_ = map_to_bounds(X_, self.domain_bounds)
            ret_discrete = map_int_to_discrete_part_in_search_space(X[0], self.discrete_set)
            ret_X = (ret_discrete, ret_X_)
        else:
            ret_X = X
        ret_Z = None if Z is None else map_to_bounds(Z, self.fidel_bounds)
        return ret_Z, ret_X

    # -----------------------------------------------------------------------------

    def get_unnormalised_cell(self, X):
        """ Maps points in the cube to the original space. """
        X_ = X[1]
        ret_Cell_ = None if X_ is None else map_cell_to_bounds(X_, self.domain_bounds)
        ret_discrete = None if X is None else map_int_to_discrete_part_in_search_space(X[0], self.discrete_set)
        ret_Cell = (ret_discrete, ret_Cell_)
        return ret_Cell

    # -----------------------------------------------------------------------------
