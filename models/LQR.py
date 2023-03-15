import numpy as np
import scipy as sp
from scipy import linalg
import sys

sys.path.append('..')
from SiMC import SiMC

# ----------------------------------------------------------------------------------------------------------------------
time_hor = 100

A = np.array([[1.5, 0], [0, -2.0]])
B = 0.5 * np.array([1, 1]).reshape((2, 1))
Q = 0.01 * np.array([[0.01, -0.5], [-0.5, 200]])
R = 0.01 * np.array([1]).reshape([1, 1])

mean = [1.0] * A.shape[0]
cov = 0.01 * np.eye(A.shape[0])
cov = cov.tolist()
# np.random.seed(1024)
x0_list = np.random.multivariate_normal(mean, cov, 500)
x0_list_ = np.array(x0_list)

p_star = sp.linalg.solve_discrete_are(A, B, Q, R)
w_star = np.linalg.inv(B.T @ p_star @ B + R) @ (B.T @ p_star @ A)

c = 0
for i in range(len(x0_list)):
    x0 = x0_list[i].reshape(A.shape[0], 1)
    c = c + x0.T @ p_star @ x0
c_star = np.linalg.norm(c / len(x0_list))

print("W* computed via ARE: ", w_star)
print("c* computed via ARE: ", c_star)


# ----------------------------------------------------------------------------------------------------------------------
def lyapunov_equation(W):
    P = sp.linalg.solve_discrete_lyapunov((A - B @ W).T, Q + W.T @ R @ W, method=None)
    return P


def cost_lqr(W_):
    x0_ = np.random.multivariate_normal(mean, cov, 1)
    x0 = x0_.reshape(A.shape[0], 1)
    c = 0
    x = x0
    for k in range(time_hor):
        u = - W_ @ x
        c = c + x.T @ Q @ x + u.T @ R @ u
        x = A @ x + B @ u
    return c


def check_lqr_stability(w):
    stable = True
    Aw = A - B @ w
    eig = np.linalg.eigvals(Aw)
    rho = np.max(abs(eig))
    return rho


def reward(init):
    x0 = init[0]
    x = init[1:]

    w_ = np.array([0.0, 0.0]).reshape((1, 2))
    w_[0, 0] = x[0]
    w_[0, 1] = x[1]
    rho = check_lqr_stability(w_)
    if rho < 0.95:
        cost = cost_lqr(w_)
    else:
        cost = 100
    y = -1 * cost

    return y


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ['LQR']


class LQR(SiMC):
    def __init__(self, k=0):
        super(LQR, self).__init__()

        modes = [0]
        w1 = [-1, 4]  # 1st components of state-feddback gain array
        w2 = [-4, 1]  # 2nd components of state-feddback gain array
        self.set_Theta([modes, w1, w2])
        self.set_k(k)

    def is_unsafe(self, init):
        return reward(init)
