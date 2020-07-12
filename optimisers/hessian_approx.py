from enum import Enum, auto

import numpy as np
from scipy import linalg

from .constants import EPSILON_DOUBLE_PRECISION


class OptimHessianType(Enum):
    BFGS = auto()
    # TODO add more


def _hessian_bfgs(step, delta_grad, hessian):
    # Note delta_grad is yk in tien mai
    # print("hessian in vals")
    # print(f"step = {step}, del_grad = {delta_grad}, hess = \n{hessian}")
    step_norm = linalg.norm(step, ord=2)
    grad_norm = linalg.norm(delta_grad, ord=2)
    temp = np.dot(step, delta_grad)

    # TODO query this lines correctness, seems odd
    if temp > np.sqrt(EPSILON_DOUBLE_PRECISION) * step_norm * grad_norm:
        # TODO probably should propagate this up the chain
        step = np.atleast_2d(step)
        delta_grad = np.atleast_2d(delta_grad)  # make sure transpose actually does transpose so we
        # get matrices and not scalars (4x1)x(1x4) vs (1x4)x(4x1)
        if step.shape[1] > step.shape[0]:
            step = np.transpose(step)
        if delta_grad.shape[1] > delta_grad.shape[0]:
            delta_grad = np.transpose(delta_grad)
        delta_grad = np.atleast_2d(delta_grad)
        # print("step shape", step.shape)
        # print("delta shape", delta_grad.shape, np.transpose(delta_grad).shape)
        step_trans = np.transpose(step)
        h_new = (
                (delta_grad @ np.transpose(delta_grad)) / temp
                - ((hessian @ step) @ (step_trans @ hessian)) / (step_trans @ hessian @ step)
                + hessian
        )
        # print("!!!!!")
        # print((delta_grad @ np.transpose(delta_grad)) / temp)
        # print(- ((hessian @ step) @ (step_trans @ hessian)) / (step_trans @ hessian @ step))


        return h_new, True
    else:
        return hessian, False


def update_hessian_approx(approx_method, step, delta_grad, hessian):
    if approx_method == OptimHessianType.BFGS:
        hessian, ok_flag = _hessian_bfgs(step, delta_grad, hessian)
        return hessian, ok_flag
    else:
        raise NotImplementedError("Only BFGS hessian implemented")
