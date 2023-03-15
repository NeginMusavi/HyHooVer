import numpy as np

# -----------------------------------------------------------------------------
class SiMC(object):
    """Set-initialized Markov Chains."""

    def __init__(self):
        super(SiMC, self).__init__()
        # increases by 1 every time the simulate function gets called
        self.cnt_queries = 0

    def is_unsafe(self, state):
        raise NotImplementedError('The new model file\
         has to implement the is_unsafe() function.')

    def set_Theta(self, Theta):
        Theta = [Theta[0], np.array(Theta[1:]).astype('float')]  # [{"a", "b", "c"}, [l1,u1],[l2,u2],[l3,u3],...]
        self.Theta = Theta

    def set_k(self, k):
        self.k = k

    def simulate(self, initial_state, k=None):
        if not (hasattr(self, 'Theta') and hasattr(self, 'k')):
            raise NotImplementedError('The new model file\
             has to specify Theta and k by calling set_Theta() and set_k()')

        # increases by 1 every time the simulate function gets called
        self.cnt_queries += 1

        if k is None:
            k = self.k

        state = initial_state
        unsafe = self.is_unsafe(state)
        reward = unsafe
        return reward

    def __call__(self, initial_state, k=None):
        return self.simulate(initial_state, k)
