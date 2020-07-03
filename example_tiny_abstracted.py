# Figure 2 network from fosgerau - have value function of single param cost
import numpy as np
import scipy
from main import RecursiveLogitModel, get_value_func_grad, RecursiveLogitDataSet
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix

import os
from os.path import join
import optimisers as op

# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network
# from optimisers import line_search_asrch
# from optimisers.optimisers import Optimiser

subfolder = "ExampleTiny"  # big data from classical v2
folder = join("Datasets", subfolder)
INCIDENCE = "incidence.txt"
TRAVEL_TIME = 'travelTime.txt'
OBSERVATIONS = "observations.txt"
file_incidence = os.path.join(folder, INCIDENCE)
file_travel_time = os.path.join(folder, TRAVEL_TIME)
file_obs = os.path.join(folder, OBSERVATIONS)
row, col, data = np.loadtxt(file_travel_time, unpack=True)
incidence_data = np.ones(len(data))


# _print = print
# def print(*args, show=False, **kwargs):
#     if show:
#         _print(*args, **kwargs)


def load_csv_to_sparse(fname, dtype=None, delim=" ", matrix_cast=None,
                       square_matrix=True):
    if matrix_cast is None:
        matrix_cast = csr_matrix
    row, col, data = np.loadtxt(fname, delimiter=delim, unpack=True, dtype=dtype)
    # print(row, col, data)
    # convert row and col to integers for coo_matrix
    # note we need this for float inputs since row cols still need to be ints to index
    rows_integer = row.astype(int)
    cols_integer = col.astype(int)
    if 0 not in rows_integer and 0 not in cols_integer:
        rows_integer = rows_integer - 1  # convert to zero based indexing if needed
        cols_integer = cols_integer - 1


    mat = matrix_cast(coo_matrix((data, (rows_integer, cols_integer)), dtype=dtype))
    if mat.shape[0] == mat.shape[1]-1 and square_matrix:
        # this means we have 1 less row than columns from our input data
        # i.e. missing the final k==d row with no successors
        ncols = np.shape(mat)[1]
        sparse_zeros = csr_matrix((1, ncols))
        mat = scipy.sparse.vstack((mat, sparse_zeros))
    return mat


travel_times_mat = load_csv_to_sparse(file_travel_time)
incidence_mat = load_csv_to_sparse(file_incidence, dtype='int')

# Get observations matrix - note: observation matrix is in sparse format, but is of the form
#   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
obs_mat = load_csv_to_sparse(file_obs, dtype='int',square_matrix=False)

network_data_struct = RecursiveLogitDataSet(travel_times=travel_times_mat, incidence_matrix=incidence_mat,
                                            turn_angles=None)
optimiser = op.LineSearchOptimiser(op.OptimType.LINE_SEARCH, op.OptimHessianType.BFGS,
                         vec_length=1, max_iter=4) # TODO check these parameters & defaults

model = RecursiveLogitModel(network_data_struct, optimiser, user_obs_mat = obs_mat)
np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 100

# from optimisers import log_likelihood
beta = np.array(-1.5)# default value, 1d for now
# log_likelihood(beta, data_struct, obs_mat)
#
beta_vec = beta
data = network_data_struct
obs = obs_mat



# temp func
# def log_likelihood(beta_vec, data:DataSet, obs, mu=1):





log_like_out, grad_out = model.get_log_likelihood()

print("Target:\nLL =0.6931471805599454\ngrad=[0.]")
print("Got:")
print("Got LL = ", log_like_out)
print("got grad_cumulative = ", grad_out)


#  Assume line search method for now
# TODO make nice. This is a sketch of line_search_iterate()

# perhaps beta should be an arg to log like - assume it is always changing
value, grad = model.get_log_likelihood()
INITIAL_STEP_LENGTH = 1.0
NEGATIVE_CURVATURE_PARAMETER = 0.0
SUFFICIENT_DECREASE_PARAMETER = 0.0001
CURVATURE_CONDITION_PARAMETER = 0.9
X_TOLERANCE = 2.2e-16
MINIMUM_STEP_LENGTH = 0
MAXIMUM_STEP_LENGTH = 1000
hessian = np.identity(data.n_dims)
p = np.linalg.solve(hessian, -grad)

if np.dot(p, grad) > 0:
    p = -p




def line_arc(step, ds):
    return (step * ds, ds)  # TODO note odd function form


import functools

# TODO this is very weird because we partial to fix a vlaue for ds
#   but then ds is returned so we actually use it even though it's hidden
arc = functools.partial(line_arc, ds=p)
# arc = lambda step: line_arc(step, p)
# sHOULD HAVE OPTIMISED CONSTANT NAMEDTUPLE
OPTIMIZE_CONSTANT_MAX_FEV = 10
stp = INITIAL_STEP_LENGTH
x = model.get_beta_vec()
optim_func = model.get_log_like_new_beta
#
#
# # TODO increment function evals

def hessian_bfgs(step, delta_grad, hessian):
    # Note delta_grad is yk in tien mai
    step_norm = linalg.norm(step, ord=2)
    grad_norm = linalg.norm(delta_grad, ord=2)
    temp = np.dot(step, delta_grad)

    # TODO query this lines correctness, seems odd
    if temp > np.sqrt(op.constants.EPSILON_DOUBLE_PRECISION) * step_norm * grad_norm:
        step_trans = np.transpose(step)
        h_new = (
                (delta_grad @ np.transpose(delta_grad)) / temp
                - ((hessian @ step) @ (step_trans @ hessian)) / (step_trans @ hessian @ step)
                + hessian
        )
        return h_new, True
    else:
        return hessian, False


def update_hessian_approx(model:RecursiveLogitModel, step, delta_grad, hessian):
    # TODO this shouldn't have model as an arg, it should be anchored somewhere where
    op_settings = model.optimiser
    method = op_settings.hessian_type
    # relevant pieces can be gotten with self.
    if method == op.OptimHessianType.BFGS:
        hessian, ok_flag = hessian_bfgs(step, delta_grad, hessian)
        return hessian, ok_flag
    else:
        raise NotImplementedError("Only BFGS hessian implemented")


print("testing p:")
print(x, value, grad, stp)
x, val_new, grad_new, stp, info, n_func_evals = op.line_search.line_search_asrch(
    optim_func, x, value, grad, arc, stp,
    maxfev=OPTIMIZE_CONSTANT_MAX_FEV)

if val_new <= value:
    # TODO need to collect these things into a struct
    #   think there already is an outline of one
    step = p * stp
    delta_grad = grad_new - grad
    value = val_new
    x = x
    grad = grad_new
    hessian, ok = update_hessian_approx(model, step, delta_grad, hessian)  # TODO write this
    print(hessian)
    Rv1 = True
else:
    Rv1 = False

model.hessian = hessian = np.identity(data.n_dims)
print(optimiser.line_search_iteration(model))
