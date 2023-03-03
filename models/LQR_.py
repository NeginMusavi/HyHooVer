
import pprint
import time
import numpy as np
import scipy as sp
from scipy import linalg

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

# x0_ = np.array([1, 1])
# x0 = x0_.reshape(A.shape[0], 1)
#
# Aw = A - B @ w_star
# eig = np.linalg.eigvals(Aw)
# print(eig)
#
# c_star = 0
# for j in range(100):
#     c = 0
#     x = x0
#     for k in range(time_hor):
#         u = - w_star @ x
#         c = c + x.T @ Q @ x + u.T @ R @ u
#         noise = np.random.multivariate_normal(mean, cov, 1).reshape(A.shape[0], 1)
#         # print(noise)
#         x = A @ x + B @ u + noise
#     c_star = c_star + c
# c_star = c_star / 100

print(c_star)


def lyapunov_equation(W):
    P = sp.linalg.solve_discrete_lyapunov((A - B @ W).T, Q + W.T @ R @ W, method=None)
    return P


def cost_lqr(W_):
    x0_ = np.random.multivariate_normal(mean, cov, 1)
    x0 = x0_.reshape(A.shape[0], 1)
    # x0_ = np.array([1, 1])
    # x0 = x0_.reshape(A.shape[0], 1)
    c = 0
    x = x0
    for k in range(time_hor):
        u = - W_ @ x
        c = c + x.T @ Q @ x + u.T @ R @ u
        x = A @ x + B @ u
        # noise = np.random.multivariate_normal(mean, cov, 1).reshape(A.shape[0], 1)
        # x = A @ x + B @ u + noise
    return c


# def cost_lqr(W_):
#     P_ = lyapunov_equation(W_)
#     x0_ = np.random.multivariate_normal(mean, cov, 1)
#     x0 = x0_.reshape(A.shape[0], 1)
#     c = x0.T @ P_ @ x0
#     return np.linalg.norm(c)


def check_lqr_stability(w):
    stable = True
    Aw = A - B @ w
    eig = np.linalg.eigvals(Aw)
    rho = np.max(abs(eig))
    return rho


def simulate(init):
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

    reward = -1 * cost
    return reward


import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['LQR']


class LQR(NiMC):
    def __init__(self, k=0):
        super(LQR, self).__init__()
        self.set_Theta([[0], [-1, 4], [-4, 1]])
        self.set_k(k)

    def is_unsafe(self, state):
        # print("I am here")
        return simulate(state)

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state