from enum import Enum, auto

import numpy as np
from scipy import linalg


from .constants import EPSILON_DOUBLE_PRECISION

class OptimHessianType(Enum):
    BFGS = auto()
    # TODO add more


def _hessian_bfgs(step, delta_grad, hessian):
    # Note delta_grad is yk in tien mai
    step_norm = linalg.norm(step, ord=2)
    grad_norm = linalg.norm(delta_grad, ord=2)
    temp = np.dot(step, delta_grad)

    # TODO query this lines correctness, seems odd
    if temp > np.sqrt(EPSILON_DOUBLE_PRECISION) * step_norm * grad_norm:
        step_trans = np.transpose(step)
        h_new = (
                (delta_grad @ np.transpose(delta_grad)) / temp
                - ((hessian @ step) @ (step_trans @ hessian)) / (step_trans @ hessian @ step)
                + hessian
        )
        return h_new, True
    else:
        return hessian, False


def update_hessian_approx(model, step, delta_grad, hessian):
    # TODO this shouldn't have model as an arg, it should be anchored somewhere where
    op_settings = model.optimiser
    method = op_settings.hessian_type
    # relevant pieces can be gotten with self.
    if method == OptimHessianType.BFGS:
        hessian, ok_flag = _hessian_bfgs(step, delta_grad, hessian)
        return hessian, ok_flag
    else:
        raise NotImplementedError("Only BFGS hessian implemented")