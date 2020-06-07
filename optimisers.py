from enum import Enum, auto

from main import DataSet

import numpy as np


class OptimType(Enum):
    LINE_SEARCH = auto()
    TRUST_REGION = auto()

class OptimHessianType(Enum):
    BFGS = auto()


class Optimiser(object):
    """Global wrapper object around all optim algs, delegate to sub classes for individual"""
    def __init__(self, method=OptimType.LINE_SEARCH, hessian_type = OptimHessianType.BFGS,
                 vec_length=1,
                 max_iter = 4):
        self.method = method
        self.hessian = hessian_type
        self.n = vec_length
        self.max_iter = max_iter

        self.n_func_evals = 0
        self.tol = 1e-6


# def log_likelihood(beta_vec, data:DataSet, obs, mu=1):
#     N = data.n_dims
#
#     grad = np.zeros(N)
#     for n in range(np.shape(obs)[0]):
#         dest = obs[n,0]
#         orig = obs[n,1]
#         # print("Dest = ", dest, "Orig =", orig)
#         instant_value_function = np.sum(beta_vec * data.attrs)
#         print("prod")
#         beta_vec * data.attrs
#         print(sum)
#         print(instant_value_function)



        # Get M and U



