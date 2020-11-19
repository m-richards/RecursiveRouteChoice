import numpy as np


class OptimFunctionState(object):
    """Data Class to store elements of log likelihood state & grad which are relevant both
    in the computation of the log likelihood and also the global optimisation algorithm

    "Bridge" between RL and the more general optimisation code
    """

    def __init__(self, value: float, grad: np.array, hessian: np.array,
                 hessian_approx_type,  # tODO type hint OptimHessianType
                 val_and_grad_evaluation_function, beta_vec, function_evals_stat=None):
        """
        :param value:
        :type value:
        :param grad:
        :type grad:
        :param hessian:
        :type hessian:
        :param val_and_grad_evaluation_function: function which takes in x_vec and returns the
            value and gradient at that particular x
        :type val_and_grad_evaluation_function:
        :param beta_vec:
        :type beta_vec:
        """
        self.value = value
        self.grad = grad
        self.hessian = hessian
        self._val_grad_function = val_and_grad_evaluation_function  # function of beta vec
        self.beta_vec = beta_vec
        self.hessian_approx_type = hessian_approx_type
        if function_evals_stat is None:
            def null_func():
                return None
            function_evals_stat = null_func
        self.function_evals_count = function_evals_stat

    def val_grad_function(self, beta_vec=None):
        if beta_vec is not None:
            self.beta_vec = beta_vec
        return self._val_grad_function(self.beta_vec)
