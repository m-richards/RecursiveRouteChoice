import functools
import abc
from enum import Enum, auto

import numpy as np
from numpy import linalg
from scipy.optimize._numdiff import approx_derivative

from .extra_optim import OptimFunctionState
from .hessian_approx import update_hessian_approx, OptimHessianType
from .line_search import line_search_asrch
from scipy.optimize import minimize
# from scipy import optimize

OPTIMIZE_CONSTANT_MAX_FEV = 10


class OptimType(Enum):
    LINE_SEARCH = auto()
    TRUST_REGION = auto()


class OptimiserBase(abc.ABC):
    """Base class for both custom and scipy to inherit from"""
    METHOD_FLAG = None
    NUMERICAL_ERROR_THRESH = -1e-3
    RESIDUAL = 1e-3
    LL_ERROR_VALUE = 99999

    def __init__(self):
        self.iter_count = 0

    def get_iteration_log(self, optim_vals: OptimFunctionState):
        out = f"[Iteration]: {self.iter_count}\n"
        val, grad = optim_vals.val_grad_function()

        out += f"\tLL = {val}, grad = {grad}\n"
        beta = u"\u03B2"
        out += f"\t{beta} = " + str(optim_vals.beta_vec) + "\n"
        out += f"\tNorm of grad: {linalg.norm(grad)}\n"
        rel_grad = self._compute_relative_gradient(val, grad, optim_vals.beta_vec)
        out += f"\tNorm of relative grad: {rel_grad} \n"
        out += f"\tNumber of function evals: {optim_vals.function_evals_count()}\n"
        # out += f"\tNumber of function evals: {model.n_log_like_calls_non_redundant}"
        return out

    @staticmethod
    def _compute_relative_gradient(fun_val, fun_grad, current_beta, typf=1.0):
        """% Compute norm of relative gradient.
                """
        # TODO review mathematics behind this

        # typf = 1.0 # some parameter? (input in tien code
        typxi = 1.0  # fixed in tien code
        # gmax = 0.0
        tmp_beta_max = np.maximum(current_beta, typxi)
        gmax = np.abs(fun_grad * tmp_beta_max / max(abs(fun_val), typf)).max()

        return gmax


SCIPY_METHODS_ZEROTH_ORDER = ('nelder-mead', 'powell', 'cobyla')
SCIPY_METHODS_FIRST_ORDER = ('cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc',
                             'slsqp', 'dogleg', 'trust-ncg', 'trust-krylov',
                             'trust-exact', 'trust-constr', '_custom')
SCIPY_METHODS_SECOND_ORDER = ('newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov',
                              'trust-exact', 'trust-constr', '_custom')
SCIPY_METHODS = (SCIPY_METHODS_ZEROTH_ORDER + SCIPY_METHODS_FIRST_ORDER
                 + SCIPY_METHODS_SECOND_ORDER)


class ScipyOptimiser(OptimiserBase):
    """Wrapper around scipy.optimize.minimize to conform to the format we
    require."""
    METHOD_FLAG = "scipy-master"

    def __init__(self, method: str, options=None, fd_options=None):

        super().__init__()
        method = method.lower()
        if method not in SCIPY_METHODS:
            raise ValueError("Method type must be a scipy.optimize"
                             " method type, one of {}".format(SCIPY_METHODS))
        self.method = method
        self.options = options
        self.fd_options = fd_options

        self.hessian_type = None  # TODO fix this, here because optimStateFunc expects it

    def solve(self, optim_function_state: OptimFunctionState, verbose=True,
              output_file=None):
        """Solve for optimal beta which minimises the negative log likelihood.
        Analogous to iterate step method on custom alg, except that these methods don't
        require step by step intervention."""

        def fun_wrapper(x):
            return optim_function_state.val_grad_function(x)[0]
        grad_wrapper = None
        hess_wrapper = None
        fd_options = self.fd_options
        options = self.options
        if options is None:
            options = {}

        if self.method not in SCIPY_METHODS_ZEROTH_ORDER:
            def grad_wrapper(x):
                return optim_function_state.val_grad_function(x)[1]

        if self.method in SCIPY_METHODS_SECOND_ORDER:
            if fd_options is None:
                fd_options = {}
            if 'method' not in fd_options:
                fd_options['method'] = '2-point'  # need to default to something

            def hess_wrapper(x):
                return approx_derivative(grad_wrapper, x, **fd_options)
        x0 = optim_function_state.beta_vec

        if verbose:
            self.iter_count = 0
            options['disp'] = True

            def cb(x):
                print(self.get_iteration_log(optim_function_state), file=output_file)
                self.iter_count += 1
        else:
            cb = None
        # print("options are", options)
        # print("method is", self.method)
        # if self.method in ['l-bfgs-b', 'tnc', 'slsqp', 'powell', 'trust-constr']:
        #     print("enforcing negative beta")
        #     bound = optimize.Bounds(-np.inf, 0.0, keep_feasible=False)
        # else:
        bound = None

        return minimize(fun_wrapper, x0, method=self.method, bounds=bound,
                        jac=grad_wrapper, hess=hess_wrapper, options=options,
                        callback=cb)


class CustomOptimiserBase(OptimiserBase):
    """Global wrapper object around all optim algs, delegate to sub classes for individual.
    Need to be clear about which properties are effectively read only and which store state.

    Reviewed perspective after implementing Scipy support is that this should be allowed to know
    details of recursive logit - and it itself may wrap a generic algorithm.
    """

    def __init__(self, hessian_type=OptimHessianType.BFGS, max_iter=4):
        super().__init__()
        self.hessian_type = hessian_type
        # self.n = vec_length
        self.max_iter = max_iter

        self.n_func_evals = 0
        self.tol = 1e-6

        #
        self.function_value = None  # Set and updated before each line search
        self.beta_vec = None

        self.step = 0  # defined in subclasses
        self.current_value = None
        self.beta_vec = None
        self.grad = None

        self.delta_grad = None
        self.delta_value = None

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
        elif self.compute_relative_gradient_non_static() < self.tol:
            stop_type = "relative gradient"
            is_successful = True
        else:
            is_stopping = False
            stop_type = None
            is_successful = False

        return is_stopping, stop_type, is_successful

    def compute_relative_gradient_non_static(self, typf=1.0):
        """% Compute norm of relative gradient.
        Note this exists as we need this information in the stopping criteria,
        where getting the fields with explicit references would be messy
        """
        return self._compute_relative_gradient(self.current_value, self.grad,
                                               self.beta_vec)

    def set_beta_vec(self, beta_vec):
        self.beta_vec = beta_vec

    def set_current_value(self, value):
        self.current_value = value

    def iterate_step(self, model, verbose=False, output_file=None, debug_counter=None):
        raise NotImplementedError()


class LineSearchOptimiser(CustomOptimiserBase):
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

    # TODO what if optimvals is part of state on both, rather than an argument
    def iterate_step(self, optim_vals: OptimFunctionState, verbose=True, output_file=None,
                     debug_counter=None):
        """ Performs a single step of the line search iteration,
            evaluating the value function and taking a step based upon the gradient"""
        self.iter_count += 1
        hessian_old = optim_vals.hessian
        value_old, grad = optim_vals.value, optim_vals.grad
        p = np.linalg.solve(hessian_old, -grad)

        if np.dot(p, grad) > 0:
            p = -p

        def line_arc(step, ds):
            return step * ds, ds  # TODO note odd function form

        arc = functools.partial(line_arc, ds=p)
        stp = self.INITIAL_STEP_LENGTH  # for each iteration we start from a fixed step length
        x = optim_vals.beta_vec

        def compute_log_like_callback(new_beta_vec):
            """Note function not method: 'lambda' passed to optim alg, lets us update the
            n_func_evals with each call"""
            self.n_func_evals += 1
            return optim_vals.val_grad_function(new_beta_vec)

        optim_func = compute_log_like_callback
        x, val_new, grad_new, stp, info, n_func_evals = line_search_asrch(
            optim_func, x, value_old, grad, arc, stp,
            maxfev=OPTIMIZE_CONSTANT_MAX_FEV)

        if val_new <= value_old:
            # TODO need to collect these things into a struct?
            self.step = p * stp
            self.delta_grad = grad_new - grad
            self.delta_value = val_new - value_old
            self.current_value = val_new
            self.beta_vec = x
            self.grad = grad_new
            hessian, ok = update_hessian_approx(optim_vals.hessian_approx_type,
                                                self.step, self.delta_grad,
                                                hessian_old)
            # update hessian stored to new value
            optim_vals.hessian = hessian
            out_flag = True
        else:
            out_flag = False
            hessian = hessian_old
        log = self.get_iteration_log(optim_vals)
        if verbose:
            print(log, file=output_file)
        return out_flag, hessian, log

    def get_iteration_log(self, optim_vals: OptimFunctionState):
        out = super().get_iteration_log(optim_vals)
        val, grad = optim_vals.val_grad_function()
        if self.grad is None:
            self.grad = grad
        out += f"\tNorm of step: {linalg.norm(self.step)}\n"

        return out


class TrustRegionOptimiser(CustomOptimiserBase):
    METHOD_FLAG = method = OptimType.TRUST_REGION

    def __init__(self, hessian_type=OptimHessianType.BFGS,
                 max_iter=4, ):
        super().__init__(hessian_type, max_iter)
        raise NotImplementedError()
