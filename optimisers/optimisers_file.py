import functools
from enum import Enum, auto

import numpy as np
from numpy import linalg

from .hessian_approx import update_hessian_approx, OptimHessianType
from .line_search import line_search_asrch




class OptimType(Enum):
    LINE_SEARCH = auto()
    TRUST_REGION = auto()


class Optimiser(object):
    """Global wrapper object around all optim algs, delegate to sub classes for individual.
    Need to be clear about which properties are effectively read only and which store state.
    Ideally this is a generic optim alg that doesn't know anything about RecursiveLogitModel
    TODO currently is intrinsically dependent"""
    METHOD_FLAG = None
    def __init__(self, hessian_type=OptimHessianType.BFGS,
                 vec_length=1,
                 max_iter = 4):
        self.hessian_type = hessian_type
        self.n = vec_length
        self.max_iter = max_iter

        self.iter_count = 0
        self.n_func_evals = 0
        self.tol = 1e-6

        #
        self.function_value = None # Set and updated before each line search
        self.beta_vec = None

        self.step = 0 # defined in subclasses
        self.current_value = None
        self.beta_vec = None
        self.grad = np.array([0])

    def check_stopping_criteria(self):
        is_stopping = True
        if self.iter_count > self.max_iter:
            stop_type = "max iteration cap"
            is_successful = False
        elif linalg.norm(self.step) <self.tol:
            stop_type = "gradient"
            is_successful = True
        elif self.compute_relative_gradient()< self.tol:
            stop_type = "relative gradient"
            is_successful = True
        else:
            is_stopping = False
            stop_type = None
            is_successful = False

        return is_stopping, stop_type, is_successful


    def compute_relative_gradient(self, typf=1.0):
        """% Compute norm of relative gradient"""
        val = self.current_value
        grad = self.grad
        #typf = 1.0 # some parameter? (input in tien code
        typxi = 1.0 # fixed in tien code
        gmax = 0.0

        for i in range(len(grad)): # tODO check what this concept is
            gmax = max(gmax, abs(grad[i] * max(self.beta_vec[i], typxi)) / max(abs(val), typf))
        print("Loop gmax = ", gmax)
        tmp_beta_max = np.maximum(self.beta_vec, typxi)
        gmax_nice = np.abs(grad * tmp_beta_max / max(abs(val), typf)).max()
        print("vectorised gmax = ", gmax_nice)

        return gmax

    def set_beta_vec(self, beta_vec):
        self.beta_vec = beta_vec

    def set_current_value(self, value):
        self.current_value = value




class LineSearchOptimiser(Optimiser):
    INITIAL_STEP_LENGTH = 1.0
    NEGATIVE_CURVATURE_PARAMETER = 0.0
    SUFFICIENT_DECREASE_PARAMETER = 0.0001
    CURVATURE_CONDITION_PARAMETER = 0.9
    X_TOLERANCE = 2.2e-16
    MINIMUM_STEP_LENGTH = 0
    MAXIMUM_STEP_LENGTH = 1000

    METHOD_FLAG = OptimType.LINE_SEARCH

    def __init__(self,hessian_type=OptimHessianType.BFGS,
                 vec_length=1, max_iter=4, ):
        super().__init__(hessian_type, vec_length, max_iter)
        #TODO adjust fields?


    def line_search_iteration(self, model, verbose=True):
        """ TODO note there is som first time initialisation that need to be removed"""
        self.n_func_evals += 1
        hessian_old = model.hessian
        value_in, grad = model.get_log_likelihood()

        p = np.linalg.solve(hessian_old, -grad)

        if np.dot(p, grad) > 0:
            p = -p

        def line_arc(step, ds):
            return (step * ds, ds)  # TODO note odd function form

        arc = functools.partial(line_arc, ds=p)
        stp = self.INITIAL_STEP_LENGTH
        x = model.get_beta_vec()
        optim_func = model.get_log_like_new_beta

        OPTIMIZE_CONSTANT_MAX_FEV = 10 #TODO sort out how function evals are tracked
        x, val_new, grad_new, stp, info, n_func_evals = line_search_asrch(
            optim_func, x, value_in, grad, arc, stp,
            maxfev=OPTIMIZE_CONSTANT_MAX_FEV)

        if val_new <= value_in:
            # TODO need to collect these things into a struct
            #   think there already is an outline of one
            self.step = p * stp
            self.delta_grad = grad_new - grad
            self.delta_value = val_new - value_in
            self.value = val_new
            self.beta_vec = x
            self.grad = grad_new
            hessian, ok = update_hessian_approx(model, self.step, self.delta_grad, hessian_old)  #
            # TODO write
            # this
            print(hessian)
            out_flag = True
        else:
            out_flag = False
            hessian = hessian_old
        log = self._line_search_iteration_log(model)  # TODO return this
        if verbose:
            print(log)
        return out_flag, hessian, log

    def _line_search_iteration_log(self, model):  # TODO fix hacky argument
        out = f"[Iteration]: {self.n_func_evals}\n"
        out += f"\tLL = {model.get_log_likelihood()}\n"
        beta = u"\u03B2"
        out += f"\t {beta} = " + str(model.get_beta_vec()) + "\n"
        out += f"Norm of step: {linalg.norm(self.step)}\n"
        # out += f"radius: \n" # a trust region thing
        out += f"Norm of grad: {linalg.norm(self.grad)}\n"
        out += f"Norm of relative grad: {self.compute_relative_gradient()} \n"
        out += f"Number of function evals: {self.n_func_evals}"

        return out


class TrustRegionOptimiser(Optimiser):
    METHOD_FLAG = method = OptimType.TRUST_REGION

    def __init__(self, hessian_type=OptimHessianType.BFGS,
                 vec_length=1, max_iter=4, ):
        super().__init__(hessian_type, vec_length, max_iter)
        raise NotImplementedError()
