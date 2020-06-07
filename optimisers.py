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


def log_likelihood(beta_vec, data:DataSet, data_obs):
    N = data.n_dims

    grad = np.zeros(N)


