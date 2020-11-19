"""
Very similar to the recursive logit in recursive_route_choice.py
Instead using the sherman morrison update formula to save a substantial number of
matrix inverse calculations.

Note that empirical tests suggest forming the matrix inverse explicitly is much faster,
but may be less numericall stable (relative to LU). In either case these are likely dense if
A is sparse, so Krylov subspace methods may have to be used for large instances

"""

import logging

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve

from scipy.sparse.linalg import splu  # noqa: F401  - this is a lie, it is used

from .optimisers import OptimiserBase
from .recursive_route_choice import (ModelDataStruct,
                                     RecursiveLogitModelEstimation, RecursiveLogitModelPrediction,
                                     RecursiveLogitModel, _to_dense_if_sparse)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ALLOW_POSITIVE_VALUE_FUNCTIONS = True


class DenseLUObj(object):
    """Wrapper class around lu_factor and lu_solve to allow consistent api usage between scipy
    dense and SuperLU sparse decomps - i.e having a solve method"""
    def __init__(self, mat):
        lu, pivot = lu_factor(mat)
        self.lu = lu
        self.pivot = pivot

    def solve(self, b, trans=0, overwrite_b=False, check_finite=True):
        """Solves Ax=b via back substitution, where A is the matrix the class was initialised
        with. For options documentation see scipy.linalg.lu_solve"""
        return lu_solve((self.lu, self.pivot), b, trans, overwrite_b, check_finite)


class RecursiveLogitModelEstimationSM(RecursiveLogitModelEstimation):

    def __init__(self, data_struct: ModelDataStruct,
                 optimiser: OptimiserBase, observations_record,
                 initial_beta=-1.5, mu=1, sort_obs=True):
        # call base class init
        RecursiveLogitModel.__init__(self, data_struct, initial_beta, mu)
        self._init_estimation_body(data_struct, optimiser, observations_record, sort_obs)

        # unfortuantely SuperLU requires a dense rhs, which is inefficient. this is why it is
        # dense in both cases. # TODO
        n = self.get_exponential_utility_matrix().shape[1]
        # used in the following update, initialise once first - see method for explanation of
        # being always dense
        std_rhs = np.zeros((n, 1))
        std_rhs[-1, 0] = 1
        self.std_rhs = std_rhs  # we reuse this right hand side vector a lot so saved
        # matrix system setup for I-M without a changing sink applied
        self.update_base_matrix_system()

        self._init_post_init()

    def update_base_matrix_system(self):
        m_mat = self.get_exponential_utility_matrix()
        n = m_mat.shape[1]
        if self.is_network_data_sparse:
            a_base = sparse.identity(n) - m_mat
            lu_obj = sparse.linalg.splu(a_base.tocsc())
            # std_rhs = sparse.lil_matrix((n, 1))
        else:
            a_base = np.identity(n) - m_mat
            lu_obj = DenseLUObj(a_base)
        # unfortunately SuperLU requires a dense rhs, which is inefficient. this is why it is
        # dense in both cases. That's why it is set outside fixed dense # TODO

        self.a_base_solver = lu_obj  # this is updated when beta changes
        # we need this system solution a lot using sherman morrison so save
        self.a_base = a_base
        # reuse this system solve lots too
        self.z_base = lu_obj.solve(self.std_rhs).reshape(n, 1)

    def update_beta_vec(self, new_beta_vec, from_init=False) -> bool:
        out = super(RecursiveLogitModelEstimationSM, self).update_beta_vec(new_beta_vec)
        if from_init is False:
            self.update_base_matrix_system()
        return out

    def compute_value_function(self, m_tilde, data_is_sparse=None) -> bool:
        # TODO refactor the checks out of this so this can be overloaded at minimal burden
        """
        Solves the linear system :math:`z = Mz+b` and stores the output for future use.
        Returns a boolean indicating if solving the linear system was successful or not.

        Parameters
        ----------
        m_tilde : :py:class:`scipy.sparse.csr_matrix`
            The matrix M modified to reflect the current location of the sink destination state.
        data_is_sparse : bool, optional
            flag to indicate if the data - in this case m_tilde is sparse. If supplied, we don't
            need to check this manually.
        Returns
        -------
        error_flag : bool
            Returns True if an error was encountered, else false if execution finished
            successfully. Errors are due to the linear system having no solution, high residual
            or having negative solution, such that the value functions have no real solution.
        """
        if data_is_sparse is None:
            data_is_sparse = sparse.issparse(m_tilde)
        a_mat, z_vec, rhs = self._compute_exp_value_function(m_tilde, return_pieces=True,
                                                             data_is_sparse=data_is_sparse)
        return self._value_function_checks(a_mat, z_vec, rhs)

    def _compute_exp_value_function(self, m_tilde, data_is_sparse, return_pieces=False):
        """Compute the exponential value functions exp(V(s)) = z_s using the
            Sherman-Morrison update"""

        # m_mat = self.get_exponential_utility_matrix()
        ncols = m_tilde.shape[1]
        if data_is_sparse:
            iden = sparse.identity(ncols)
        else:
            iden = np.identity(ncols)

        a_tilde = iden - m_tilde
        # a_orig = iden - m_mat
        # a_tilde = a_orig + u_vec.dot(v_vec)  # TODO if we only need this for the norm,
        # is there a better way?, some sort of local update for norm?

        rhs = self.std_rhs
        # only change the final column in outer prod, include transpose explicitly
        v_vec = rhs.T
        # u_vec is th final col of  (I-Mtilde) - (I-M) = M - Mtilde.
        # but M is all zero in the final col
        u_vec = - m_tilde[:, -1]

        z = self.z_base
        # Again ideally would use sparse vec but SuperLU doesn't support
        y = self.a_base_solver.solve(u_vec.A)

        z_tilde = z - v_vec.dot(z) / (1 + v_vec.dot(y)) * y
        if return_pieces:
            return a_tilde, z_tilde, rhs
        else:
            return z_tilde


class RecursiveLogitModelPredictionSM(RecursiveLogitModelPrediction):
    """This class is a bit messy because theres a bit of an inheritance problem.
    A potential fix is to roll the two Prediction and Estimation classes together,
    but we might want to have them in separate instances.
    Totally could have this and methods in common subclass of RecursiveLogitModel,
    but currently don't have time to look at MRO to get the subclasses for prediction and
    estimation behaving.

    """

    def __init__(self, data_struct: ModelDataStruct, initial_beta=-1.5, mu=1.0):
        super(RecursiveLogitModelPredictionSM, self).__init__(data_struct, initial_beta, mu)
        # content is duplicated from Estimation init, should update that first then this
        # unfortuantely SuperLU requires a dense rhs, which is inefficient. this is why it is
        # dense in both cases. # TODO
        n = self.get_exponential_utility_matrix().shape[1]
        # used in the following update, initialise once first - see method for explanation of
        # being always dense
        std_rhs = np.zeros((n, 1))
        std_rhs[-1, 0] = 1
        self.std_rhs = std_rhs  # we reuse this right hand side vector a lot so saved
        # matrix system setup for I-M without a changing sink applied
        self.update_base_matrix_system()

    def update_base_matrix_system(self):
        """Copy of method from Estimation. Should be implemented in a common parent, or as an
        interface/ subclass, rather than duplication"""
        m_mat = self.get_exponential_utility_matrix()
        n = m_mat.shape[1]
        if self.is_network_data_sparse:
            a_base = sparse.identity(n) - m_mat
            lu_obj = sparse.linalg.splu(a_base.tocsc())
            # std_rhs = sparse.lil_matrix((n, 1))
        else:
            a_base = np.identity(n) - m_mat
            lu_obj = DenseLUObj(a_base)
        # unfortunately SuperLU requires a dense rhs, which is inefficient. this is why it is
        # dense in both cases. That's why it is set outside fixed dense # TODO

        self.a_base_solver = lu_obj  # this is updated when beta changes
        # we need this system solution a lot using sherman morrison so save
        self.a_base = a_base
        # reuse this system solve lots too
        self.z_base = lu_obj.solve(self.std_rhs).reshape(n, 1)

    def _compute_exp_value_function(self, m_tilde, data_is_sparse, return_pieces=False):
        """Compute the exponential value functions exp(V(s)) = z_s using the
            Sherman-Morrison update. Copy from estimation code"""

        # m_mat = self.get_exponential_utility_matrix()
        ncols = m_tilde.shape[1]
        if data_is_sparse:
            iden = sparse.identity(ncols)
        else:
            iden = np.identity(ncols)

        a_tilde = iden - m_tilde
        # a_orig = iden - m_mat
        # a_tilde = a_orig + u_vec.dot(v_vec)  # TODO if we only need this for the norm,
        # is there a better way?, some sort of local update for norm?

        rhs = self.std_rhs
        # only change the final column in outer prod, include transpose explicitly
        v_vec = rhs.T
        # u_vec is th final col of  (I-Mtilde) - (I-M) = M - Mtilde.
        # but M is all zero in the final col
        u_vec = - m_tilde[:, -1]

        z = self.z_base
        # Again ideally would use sparse vec but SuperLU doesn't support
        y = self.a_base_solver.solve(_to_dense_if_sparse(u_vec))

        z_tilde = z - v_vec.dot(z) / (1 + v_vec.dot(y)) * y
        if return_pieces:
            return a_tilde, z_tilde, rhs
        else:
            return z_tilde
