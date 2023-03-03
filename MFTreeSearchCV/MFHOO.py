# Author: Rajat Sen
# Extended by Negin Musavi to Hybrid Setting

from __future__ import print_function
from __future__ import division

import os
import random
import sys
import time
import numpy as np
import math

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# -----------------------------------------------------------------------------

nu_mult = 1  # multiplier to the nu parameter

# -----------------------------------------------------------------------------


def flip(p):
    return True if random.random() < p else False

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MF_node(object):

    def __init__(self, cell, value, fidel, upp_bound, height, dimension, num_trees, num):
        """This is a node of the MFTREE
        cell: tuple denoting the bounding boxes of the partition
        m_value: mean value of the observations in the cell and its children
        value: value in the cell
        fidelity: the last fidelity that the cell was queried with
        upp_bound: B_{i,t} in the paper
        t_bound: upper bound with the t dependent term
        height: height of the cell (sometimes can be referred to as depth in the tree)
        dimension: the dimension of the parent that was halved in order to obtain this cell
        num: number of queries inside this partition so far
        left,right,parent: pointers to left, right and parent
        """
        self.cell = cell
        self.m_value = value
        self.value = value
        self.fidelity = fidel
        self.upp_bound = upp_bound
        self.height = height
        self.dimension = dimension
        self.num = num
        self.t_bound = upp_bound

        self.parent = None
        self.children = []
        self.num_trees = num_trees

    # -----------------------------------------------------------------------------

    def __cmp__(self, other):
        return cmp(other.t_bound, self.t_bound)

    # -----------------------------------------------------------------------------


class MF_tree(object):
    """
    MF_tree class that maintains the multi-fidelity tree
    nu: nu parameter in the paper
    rho: rho parameter in the paper
    sigma: noise variance, ususally a hyperparameter for the whole process
    C: parameter for the bias function as defined in the paper
    root: can initialize a root node, when this parameter is supplied by a MF_node object instance
    """

    # -----------------------------------------------------------------------------

    def __init__(self, nu, rho, sigma, C, root=None):
        self.nu = nu
        self.rho = rho
        self.sigma = sigma
        self.root = root
        self.C = C
        self.root = root
        self.mheight = 0
        self.maxi = float(-sys.maxsize - 1)
        self.current_best = root
        self.current_best_cell = {}

    # -----------------------------------------------------------------------------

    def insert_node(self, current, child):
        """ insert a node in the tree in the appropriate position """
        child.height = current.height + 1
        current.children.append(child)
        index = len(current.children)
        current.children[index-1].parent = current
        return current.children[index-1]

    # -----------------------------------------------------------------------------

    def update_parents(self, node, val):
        """
        update the upperbound and mean value of a parent node, once a new child is inserted in its child tree. This process proceeds recursively up the tree
        """
        if node.parent is None:
            return
        else:
            parent = node.parent
            parent.m_value = (parent.num * parent.m_value + val) / (1.0 + parent.num)
            parent.num = parent.num + 1.0
            parent.upp_bound = parent.m_value + 2 * ((self.rho) ** (parent.height)) * self.nu
            self.update_parents(parent, val)

    # -----------------------------------------------------------------------------

    def update_tbounds(self, root, t):
        """ updating the tbounds of every node recursively """
        if root is None:
            return
        for c in range(len(root.children)):
            self.update_tbounds(root.children[c], t)
        root.t_bound = root.upp_bound + np.sqrt(2 * (self.sigma ** 2) * np.log(t) / root.num)
        maxi = None
        if len(root.children) > 0:
            maxi = root.children[0].t_bound
        for c in range(len(root.children)-1):
            if maxi:
                if maxi < root.children[c + 1].t_bound:
                    maxi = root.children[c + 1].t_bound
            else:
                maxi = root.children[c + 1].t_bound
        if maxi:
            root.t_bound = min(root.t_bound, maxi)

    # -----------------------------------------------------------------------------

    def get_next_node(self, root):
        """
        getting the next node to be queried or broken, see the algorithm in the paper
        """
        if root is None:
            print('Could not find next node. Check Tree.')
        if not root.children:
            return root
        if root.children:
            bounds = []
            for c in range(len(root.children)):
                bounds.append(root.children[c].t_bound)
            bnd = np.array(bounds)
            index = np.random.choice(np.where(bnd == bnd.max())[0])
            return self.get_next_node(root.children[index])

    # -----------------------------------------------------------------------------

    def get_current_best(self, root):
        """
        get current best cell from the tree
        """
        if root is None:
            return
        if not root.children:
            val = root.m_value - self.nu * ((self.rho) ** (root.height))
            if self.maxi < val:
                self.maxi = val
                cell_ = list(root.cell[1])
                self.current_best_cell = root.parent.cell
                # print(root.cell)
                # print(root.parent.cell)
                # # -------------draw random sample (not midpoint)----------------
                # rnd = random.random()
                # self.current_best = np.array([(rnd * (s[1] - s[0]) + s[0]) for s in cell])
                # # --------------------------------------------------------------
                self.current_best = (root.cell[0], np.array([(s[0] + s[1]) / 2.0 for s in cell_]))
            return
        if root.children:
            for c in range(len(root.children)):
                self.get_current_best(root.children[c])

    # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MFHOO(object):
    """
    MFHOO algorithm, given a fixed nu and rho
    mfobject: multi-fidelity noisy function object
    nu: nu parameter
    rho: rho parameter
    budget: total budget provided either in units or time in seconds
    sigma: noise parameter
    C: bias function parameter
    tol: default parameter to decide whether a new fidelity query is required for a cell
    Randomize: True implies that the leaf is split on a randomly chosen dimension, False means the scheme in DIRECT algorithm is used. We recommend using False.
    Auto: Select C automatically, which is recommended for real data experiments
    CAPITAL: 'Time' mean time in seconds is used as cost unit, while 'Actual' means unit cost used in synthetic experiments
    debug: If true then more messages are printed
    """

    # -----------------------------------------------------------------------------

    def __init__(self, mfobject, nu, rho, budget, sigma, C, stored_best_points, stored_best_cells, stored_best_values, stored_best_heights, stored_num_queries, stored_t_bounds, tol=1e-3, \
                 Randomize=False, Auto=True, value_dict={}, \
                 CAPITAL='Time', debug='True', comp_value_dict={}, useHOO=False):
        self.num_query = 0
        self.mfobject = mfobject
        self.nu = nu
        self.rho = rho
        self.budget = int(budget)
        self.C = 1 * C
        self.t = 0
        self.sigma = sigma
        self.tol = tol
        self.Randomize = Randomize
        self.cost = 0
        self.cflag = False
        self.value_dict = value_dict
        self.comp_value_dict = comp_value_dict
        self.CAPITAL = CAPITAL
        self.debug = debug
        self.useHOO = useHOO
        self.stored_best_points = stored_best_points
        self.stored_best_cells = stored_best_cells
        self.stored_best_values = stored_best_values
        self.stored_best_heights = stored_best_heights
        self.stored_num_queries = stored_num_queries
        self.stored_t_bounds = stored_t_bounds
        if Auto:
            z1 = 0.8
            z2 = 0.2
            d = self.mfobject.domain_dim
            # # -------------draw random sample (not midpoint)----------------
            # rnd = random.random()
            # x = np.array([rnd] * d)
            # # --------------------------------------------------------------
            x = [0, np.array([0.5] * d)]
            t1 = time.time()
            if not self.useHOO:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised_average(z1, x, mfobject.max_iteration)
                v2 = self.mfobject.eval_at_fidel_single_point_normalised_average(z2, x, mfobject.max_iteration)
            else:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised_average(z1, x, mfobject.max_iteration) # FIXME: old code use eval_at_fidel_single_point_normalised which only run simulation for one time, here we run mfobject.max_iteration to make it consistent with MFHOO. However, HOO actually doesn't reach here, it doesn't use Auto
                v2 = self.mfobject.eval_at_fidel_single_point_normalised_average(z2, x, mfobject.max_iteration)

            t2 = time.time()
            self.C = 1 * max(np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2), 0.01)
            # self.C = np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2)
            # if self.C == 0:
            #     self.Auto = False
            #     self.nu = self.nu
            # else:
            #     self.Auto = True
            #     self.nu = nu_mult * self.C

            self.nu = nu_mult * self.C
            if self.debug:
                print('Auto Init: ')
                print('C: ' + str(self.C))
                print('nu: ' + str(self.nu))
            c1 = self.mfobject.eval_fidel_cost_single_point_normalised(z1)
            c2 = self.mfobject.eval_fidel_cost_single_point_normalised(z2)
            self.cost = c1 + c2
            if self.CAPITAL == 'Time':
                self.cost = t2 - t1
        d = self.mfobject.domain_dim
        height = 0

        cell = []
        for l in range(self.mfobject.num_trees):
            c = (l, tuple([(0, 1)] * d))
            cell.append(c)
        # cell = tuple(cell)
        cell = tuple(cell)

        if not self.useHOO:
            dimension = 1
            root, cost = self.querie(cell, height, self.rho, self.nu, dimension, option=1)
        else:
            dimension = 0
            root, cost = self.querie(cell, height, self.rho, self.nu, dimension, option=0)

        self.t = self.t + 1 * self.mfobject.batch_size
        self.num_query = self.num_query + 1
        self.Tree = MF_tree(nu, rho, self.sigma, C, root)
        # print(root.cell)
        self.Tree.update_tbounds(self.Tree.root, self.t)
        self.cost = self.cost + cost

    # -----------------------------------------------------------------------------

    def get_value(self, cell, fidel):
        """cell: tuple"""
        mean_value = 0
        m = 0
        cell_ = list(cell[1])
        for i in range(self.mfobject.batch_size):
            if self.mfobject.rand_sampling:
                # -------------draw random sample (not midpoint)----------------
                # random.seed(32) # 14 28 31 32
                rnd = random.random()
                x = (cell[0], np.array([(rnd * (s[1] - s[0]) + s[0]) for s in cell_]))
            else:
                x = (cell[0], np.array([(s[0] + s[1]) / 2.0 for s in cell_]))
            mean_value = mean_value + self.mfobject.eval_at_fidel_single_point_normalised(fidel, x)
        mean_value = mean_value / self.mfobject.batch_size
        return mean_value

    # -----------------------------------------------------------------------------

    def querie(self, cell, height, rho, nu, dimension, option=1):
        diam = nu * (rho ** height)
        if option == 1:
            # if self.C == 0.0:
            #     z = 1
            # else:
            #     z = min(max(1 - diam / self.C, self.tol), 1.0)
            z = min(max(1 - diam / self.C, self.tol), 1.0)
        else:
            z = 1.0
        if False:#cell in self.value_dict: # disable cache for fair comparison. FIXME: a better idea is to change L418-420, check z also when check the cache, add new cell if z is different.
            current = self.value_dict[cell]
            if abs(current.fidelity - z) <= self.tol:
                value = current.value
                cost = 0
            else:
                t1 = time.time()
                value = self.get_value(cell, z)
                t2 = time.time()
                if abs(value - current.value) > self.C * abs(current.fidelity - z):
                    self.cflag = True
                current.value = value
                current.m_value = value
                current.fidelity = z
                self.value_dict[cell] = current
                if self.CAPITAL == 'Time':
                    cost = t2 - t1
                else:
                    cost = self.mfobject.eval_fidel_cost_single_point_normalised(z)
        else:
            t1 = time.time()
            if height == 0:
                value = 0
            else:
                value = self.get_value(cell, z)
            t2 = time.time()
            bhi = 2 * diam + value
            self.value_dict[cell] = MF_node(cell, value, z, bhi, height, dimension, self.mfobject.num_trees, 1)
            if self.CAPITAL == 'Time':
                cost = t2 - t1
            else:
                cost = self.mfobject.eval_fidel_cost_single_point_normalised(z)

        bhi = 2 * diam + value
        current_object = MF_node(cell, value, z, bhi, height, dimension, self.mfobject.num_trees, 1)
        return current_object, cost

    # -----------------------------------------------------------------------------

    def update_comp_value_dict(self, node):
        """
        update the upperbound and mean value of a parent node, once a new child is inserted in its child tree. This process proceeds recursively up the tree
        """
        if node.parent is None:
            return
        else:
            parent = node.parent

            self.comp_value_dict[parent.cell] = parent
            self.update_comp_value_dict(parent)

    # -----------------------------------------------------------------------------

    def split_children(self, current, rho, nu, option=1):
        if current.height == 0:
            children = []
            cost = 0
            for l in range(self.mfobject.num_trees):
                cell = current.cell[l]
                h = current.height + 1
                dimension = l
                child, c = self.querie(cell, h, rho, nu, dimension, option)
                children.append(child)
                cost = cost + c
        else:
            pcell = list(current.cell[1])
            span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
            if self.Randomize:
                dimension = np.random.choice(range(len(pcell)))
            else:
                dimension = np.argmax(span)
            dd = len(pcell)
            if dimension == current.dimension:
                dimension = (current.dimension - 1) % dd
            cost = 0
            h = current.height + 1
            l = np.linspace(pcell[dimension][0], pcell[dimension][1], 3)
            children = []
            for i in range(len(l) - 1):
                cell_ = []
                for j in range(len(pcell)):
                    if j != dimension:
                        cell_ = cell_ + [pcell[j]]
                    else:
                        cell_ = cell_ + [(l[i], l[i + 1])]
                cell = (current.cell[0], tuple(cell_))
                child, c = self.querie(cell, h, rho, nu, dimension, option)
                # _t2 = time.time()
                # real_cost = _t2 - _t1
                # import pdb; pdb.set_trace()
                children = children + [child]
                cost = cost + c

        return children, cost

    # -----------------------------------------------------------------------------

    def take_HOO_step(self):
        current = self.Tree.get_next_node(self.Tree.root)
        children, cost = self.split_children(current, self.rho, self.nu, 1 if not self.useHOO else 0)
        # import pdb; pdb.set_trace()
        self.t = self.t + 2 * self.mfobject.batch_size
        self.cost = self.cost + cost
        for c in range(len(children)):
            rnode = self.Tree.insert_node(current, children[c])
            self.Tree.update_parents(rnode, rnode.value)
            self.update_comp_value_dict(rnode)
        self.Tree.update_tbounds(self.Tree.root, self.t)
    # -----------------------------------------------------------------------------

    def run(self, output):
        #import pdb; pdb.set_trace()

        while (self.t + 2 * self.mfobject.batch_size) <= self.budget:
            self.num_query = self.num_query + 2 * self.mfobject.batch_size
            self.take_HOO_step()

        p_, c_ = self.get_point()
        self.store_bests(p_, c_, self.num_query)
        print('max value with ' + str(self.num_query) + ' queries: ', str(self.stored_best_values[-1]))
        output.write('number of queries: ' + str(self.num_query) + ', max value: ' + str(
            self.stored_best_values[-1]) + '\n')

    # -----------------------------------------------------------------------------

    def get_point(self):
        # print(self.Tree.root.cell)
        self.Tree.get_current_best(self.Tree.root)
        return self.Tree.current_best, self.Tree.current_best_cell
    # -----------------------------------------------------------------------------

    def store_bests(self, p_, c_, q_):
        _node = self.comp_value_dict[c_]
        _ncell = self.mfobject.get_unnormalised_cell(_node.cell)
        _, _npoint = self.mfobject.get_unnormalised_coords(None, p_)
        self.stored_best_points.append(_npoint)
        self.stored_best_cells.append(_ncell)
        v_ = self.mfobject.eval_at_fidel_single_point_normalised_average(1, p_, self.mfobject.max_iteration)
        self.stored_best_values.append(v_)
        h_ = _node.height
        self.stored_best_heights.append(h_)
        self.stored_num_queries.append(q_)
        self.stored_t_bounds.append(self.Tree.root.t_bound)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MFPOO(object):
    """
    MFPOO object that spawns multiple MFHOO instances
    """

    # -----------------------------------------------------------------------------

    def __init__(self, mfobject, nu_max, rho_max, nHOO, sigma, C, mult, tol=1e-3, Randomize=False, Auto=True,
                 unit_cost=1.0, CAPITAL='Time', debug='True', useHOO=False, direct_budget=0.2):
        self.number_of_queries = []
        self.mfobject = mfobject
        self.nu_max = nu_max
        self.rho_max = rho_max
        self.nHOO = nHOO
        self.budget = direct_budget
        self.C = 1* C
        self.t = 0
        self.sigma = sigma
        self.tol = tol
        self.Randomize = Randomize
        self.cost = 0
        self.value_dict = {}
        self.comp_value_dict = {}
        self.MH_arr = []
        self.CAPITAL = CAPITAL
        self.debug = debug
        self.useHOO = useHOO
        if useHOO: assert Auto==False

        if Auto:
            if unit_cost is None:
                z1 = 1.0
                if self.debug:
                    print('Setting unit cost automatically as None was supplied')
            else:
                z1 = 0.8
            z2 = 0.2
            d = self.mfobject.domain_dim
            # # -------------draw random sample (not midpoint)----------------
            # rnd = random.random()
            # x = np.array([rnd] * d)
            # # --------------------------------------------------------------
            x = [0, np.array([0.5] * d)]
            t1 = time.time()
            # max_iteration = 50
            if not self.useHOO:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised_average(z1, x, mfobject.max_iteration)
            else:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised([z1], x)
            t3 = time.time()
            if not self.useHOO:
                v2 = self.mfobject.eval_at_fidel_single_point_normalised_average(z2, x, mfobject.max_iteration)
            else:
                v2 = self.mfobject.eval_at_fidel_single_point_normalised([z2], x)
            t2 = time.time()
            self.C = 1 * max(np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2), 0.01)
            # self.C = np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2)
            # if self.C == 0:
            #     self.Auto = False
            #     self.nu_max = self.nu_max
            # else:
            #     self.Auto = True
            #     self.nu_max = nu_mult * self.C
            self.nu_max = nu_mult * self.C
            if unit_cost is None:
                unit_cost = t3 - t1
                if self.debug:
                    print('Unit Cost: ', unit_cost)
            if self.debug:
                print('Auto Init: ')
                print('C: ' + str(self.C))
                print('nu: ' + str(self.nu_max))
            c1 = self.mfobject.eval_fidel_cost_single_point_normalised(z1)
            c2 = self.mfobject.eval_fidel_cost_single_point_normalised(z2)

        # if self.CAPITAL == 'Time':
        #     self.unit_cost = unit_cost
        # else:
        #     self.unit_cost = self.mfobject.eval_fidel_cost_single_point_normalised(1.0)
        if self.debug:
            print('Number of MFHOO Instances: ' + str(self.nHOO))
            print('Budget per MFHOO Instance:' + str(math.floor(self.budget) + self.mfobject.eval_mult))

    # -----------------------------------------------------------------------------

    def run_all_MFHOO(self, output):
        nu = self.nu_max
        self.number_of_queries = []
        for i in range(self.nHOO):
            rho = self.rho_max ** (float(self.nHOO) / (self.nHOO - i))
            MH = MFHOO(mfobject=self.mfobject, nu=nu, rho=rho, budget=self.budget, sigma=self.sigma, C=self.C, stored_best_points=[], stored_best_cells=[], stored_best_values=[], stored_best_heights=[], stored_num_queries=[], stored_t_bounds=[], tol=1e-3,
                       Randomize=False, Auto=True if not self.useHOO else False, value_dict=self.value_dict, CAPITAL=self.CAPITAL, debug=self.debug, comp_value_dict = self. comp_value_dict, useHOO=self.useHOO)
            print('------------------')
            print('Running HOO number: ' + str(i + 1) + ' rho: ' + str(rho) + ' nu: ' + str(nu))
            output.write('------------------' + '\n')
            output.write('Running HOO number ' + str(i + 1) + ', rho ' + str(rho) + ', nu ' + str(nu) + '\n')

            MH.run(output)

            i = i + 1

            self.number_of_queries = self.number_of_queries + [MH.t]
            print('Done!')
            output.write('Done!\n')

            self.cost = self.cost + MH.cost
            if MH.cflag:
                self.C = 1.4 * self.C
                # if self.C == 0:
                #     nu = self.nu_max
                #     MH.Auto = False
                # else:
                #     nu = nu_mult * self.C
                #     self.nu_max = nu_mult * self.C
                #     MH.Auto = True
                nu = nu_mult * self.C
                self.nu_max = nu_mult * self.C
                if self.debug:
                    print('Updating C')
                    print('C: ' + str(self.C))
                    print('nu_max: ' + str(nu))
            self.value_dict = MH.value_dict
            self. comp_value_dict = MH. comp_value_dict
            self.MH_arr = self.MH_arr + [MH]

    # -----------------------------------------------------------------------------
    def get_point(self):
        b_cells = []
        b_points = []
        b_values = []
        b_heights = []
        n_queries = []
        b_B_values = []
        for H in self.MH_arr:
            b_cells.append(H.stored_best_cells)
            b_points.append(H.stored_best_points)
            b_values.append(H.stored_best_values)
            b_heights.append(H.stored_best_heights)
            n_queries.append(H.stored_num_queries)
            b_B_values.append(H.stored_t_bounds)
        return b_points, b_cells, b_values, b_heights, b_B_values, n_queries
