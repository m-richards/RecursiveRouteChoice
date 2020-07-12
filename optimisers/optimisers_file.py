import functools
import abc
from enum import Enum, auto

import numpy as np
from numpy import linalg

from .hessian_approx import update_hessian_approx, OptimHessianType
from .line_search import line_search_asrch



class OptimType(Enum):
    LINE_SEARCH = auto()
    TRUST_REGION = auto()


class Optimiser(abc.ABC):
    """Global wrapper object around all optim algs, delegate to sub classes for individual.
    Need to be clear about which properties are effectively read only and which store state.
    Ideally this is a generic optim alg that doesn't know anything about RecursiveLogitModel
    TODO currently is intrinsically dependent"""
    METHOD_FLAG = None
    NUMERICAL_ERROR_THRESH = -1e-3
    RESIDUAL = 1e-3
    LL_ERROR_VALUE = 99999
    OPTIMIZE_CONSTANT_MAX_FEV = 10

    def __init__(self, hessian_type=OptimHessianType.BFGS, max_iter=4):
        self.hessian_type = hessian_type
        # self.n = vec_length
        self.max_iter = max_iter

        self.iter_count = 0
        self.n_func_evals = 0
        self.tol = 1e-6

        #
        self.function_value = None  # Set and updated before each line search
        self.beta_vec = None

        self.step = 0  # defined in subclasses
        self.current_value = None
        self.beta_vec = None
        self.grad = None

    def check_stopping_criteria(self):
        is_stopping = True
        if self.iter_count > self.max_iter:
            stop_type = "max iteration cap"
            is_successful = False
        elif linalg.norm(self.grad) < self.tol:
            stop_type = "gradient"
            is_successful = True
        elif linalg.norm(self.step) < self.tol * self.tol:
            stop_type = "step too small"
            is_successful = False
        elif self.compute_relative_gradient() < self.tol:
            stop_type = "relative gradient"
            is_successful = True
        else:
            is_stopping = False
            stop_type = None
            is_successful = False

        return is_stopping, stop_type, is_successful

    def compute_relative_gradient(self, typf=1.0):
        """% Compute norm of relative gradient"""
        # TODO review mathematics behind this
        val = self.current_value
        grad = self.grad
        typxi = 1.0  # fixed in tien code
        tmp_beta_max = np.maximum(self.beta_vec, typxi)
        gmax = np.abs(grad * tmp_beta_max / max(abs(val), typf)).max()

        return gmax

    def set_beta_vec(self, beta_vec):
        self.beta_vec = beta_vec

    def set_current_value(self, value):
        self.current_value = value

    def get_iteration_log(self, model):
        raise NotImplementedError()

    def iterate_step(self, model, verbose=False, output_file=None):
        raise NotImplementedError()


class LineSearchOptimiser(Optimiser):
    INITIAL_STEP_LENGTH = 1.0
    NEGATIVE_CURVATURE_PARAMETER = 0.0
    SUFFICIENT_DECREASE_PARAMETER = 0.0001
    CURVATURE_CONDITION_PARAMETER = 0.9
    X_TOLERANCE = 2.2e-16
    MINIMUM_STEP_LENGTH = 0
    MAXIMUM_STEP_LENGTH = 1000

    METHOD_FLAG = OptimType.LINE_SEARCH

    def __init__(self, hessian_type=OptimHessianType.BFGS, max_iter=4):
        super().__init__(hessian_type, max_iter)

    def iterate_step(self, model, verbose=True, output_file=None):
        """ Performs a single step of the line search iteration,
            evaluating the value function and taking a step based upon the gradient"""
        self.iter_count += 1
        hessian_old = model.hessian
        value_current, grad = model.get_log_likelihood()
        p = np.linalg.solve(hessian_old, -grad)

        if np.dot(p, grad) > 0:
            p = -p

        def line_arc(step, ds):
            return step * ds, ds  # TODO note odd function form

        arc = functools.partial(line_arc, ds=p)
        stp = self.INITIAL_STEP_LENGTH # for each iteration we start from a fixed step length
        x = model.get_beta_vec()
        # Nested function, still same method
        def compute_new_log_like(new_beta_vec):
            """Note function not method: 'lambda' passed to optim alg, lets us update the
            n_func_evals with each call"""
            self.n_func_evals += 1
            model.update_beta_vec(new_beta_vec)
            return model.get_log_likelihood()

        optim_func = compute_new_log_like
        x, val_new, grad_new, stp, info, n_func_evals = line_search_asrch(
            optim_func, x, value_current, grad, arc, stp,
            maxfev=self.OPTIMIZE_CONSTANT_MAX_FEV)

        if val_new <= value_current:
            # TODO need to collect these things into a struct
            #   think there already is an outline of one
            self.step = p * stp
            self.delta_grad = grad_new - grad
            self.delta_value = val_new - value_current
            self.value = val_new
            self.beta_vec = x
            self.grad = grad_new
            hessian, ok = update_hessian_approx(model, self.step, self.delta_grad, hessian_old)  #
            out_flag = True
        else:
            out_flag = False
            hessian = hessian_old
        log = self.get_iteration_log(model)  
        if verbose:
            print(log, file=output_file)
        # TODO this is super sneaky and non obvious, copying an anti pattern
        model.hessian = hessian
        return out_flag, hessian, log

    def get_iteration_log(self, model):  # TODO fix hacky argument
        out = f"[Iteration]: {self.iter_count}\n"
        val, grad = model.get_log_likelihood()
        if self.grad is None:
            self.grad = grad
        out += f"\tLL = {val}, grad = {grad}\n"
        beta = u"\u03B2"
        out += f"\t{beta} = " + str(model.get_beta_vec()) + "\n"
        out += f"\tNorm of step: {linalg.norm(self.step)}\n"
        # out += f"radius: \n" # a trust region thing
        out += f"\tNorm of grad: {linalg.norm(self.grad)}\n"
        out += f"\tNorm of relative grad: {self.compute_relative_gradient()} \n"
        out += f"\tNumber of function evals: {model.n_log_like_calls_non_redundant}"

        return out


class TrustRegionOptimiser(Optimiser):
    METHOD_FLAG = method = OptimType.TRUST_REGION

    def __init__(self, hessian_type=OptimHessianType.BFGS,
                 vec_length=1, max_iter=4, ):
        super().__init__(hessian_type, max_iter)
        raise NotImplementedError()
