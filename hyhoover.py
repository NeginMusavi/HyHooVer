from MFTreeSearchCV.MFHOO import *
from pympler.asizeof import asizeof
from MFTreeSearchCV.MFSiMC import MFSiMC
# ----------------------------------------------------------------------------------------------------------------------

useHOO = True


# def evaluate_single_state(X, mult):
#     # for j in range(mfobject.domain_dim):
#     #     init_state[0][j] = X[j]
#     init_state = X
#     value = 0
#     for _ in range(mult):
#         reward = mfobject.reward_function(init_state, fidelity)
#         value = value + reward
#     value = value / mult
#     return value

# ----------------------------------------------------------------------------------------------------------------------

def estimate_max_probability(SiMC, num_HOO, rho_max, sigma, budget, batch_size, rand_sampling, output, eval_mult=100,
                             debug=False):
    mfobject = MFSiMC(SiMC, batch_size, rand_sampling, eval_mult)
    MP = MFPOO(mfobject=mfobject, nu_max=1.0, rho_max=rho_max, nHOO=num_HOO, sigma=sigma, C=0.1, mult=0.5, tol=1e-3,
               Randomize=False, Auto=True if not useHOO else False, unit_cost=mfobject.opt_fidel_cost, useHOO=useHOO,
               direct_budget=budget)
    MP.run_all_MFHOO(output)
    bp, bc, bv, bd, bbv, nq = MP.get_point()

    init_state = np.zeros((1, mfobject.domain_dim))
    fidelity = mfobject.fidel_bounds[0][1]

    overall_best_cells = []
    overall_best_points = []
    overall_best_mean_values = []
    overall_best_depths = []

    for b in range(len(bv[0])):
        max = -1e6
        max_index = -1
        for hoo in range(len(bv)):
            if bv[hoo][b] >= max:
                max = bv[hoo][b]
                max_index = hoo
        overall_best_cells.append(bc[max_index][b])
        overall_best_points.append(bp[max_index][b])
        overall_best_mean_values.append(bv[max_index][b])
        overall_best_depths.append(bd[max_index][b])

    memory_usage = asizeof(MP)

    return memory_usage, overall_best_cells, overall_best_points, overall_best_depths, overall_best_mean_values, nq