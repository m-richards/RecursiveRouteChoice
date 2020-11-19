import abc
import warnings
from typing import List, Union
import logging

import numpy as np
from scipy import linalg, sparse
# from scipy.sparse import coo_matrix
from scipy.sparse import linalg as splinalg

import awkward1 as ak

# from debug_helpers import print_sparse, print_data_struct
from .optimisers.extra_optim import OptimFunctionState
from .optimisers.optimisers_file import CustomOptimiserBase, OptimType, ScipyOptimiser, \
    OptimiserBase

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ALLOW_POSITIVE_VALUE_FUNCTIONS = True


def _to_dense_if_sparse(mat, is_sparse=None):
    """ convert to dense if sparse. Optional flag if this information is known a priori -
        should be faster to recieve a bool directly than checking with isinstance"""
    is_sparse = is_sparse if is_sparse is not None else sparse.issparse(mat)
    if is_sparse:
        return mat.toarray()
    else:
        return mat


def _zero_pad_mat(mat, top=False, left=False, bottom=False, right=False):
    """Abstracted since this will get done a fair bit and this is convenient way of doing it but
    perhaps not the fastest"""
    # pad_width =((int(top), int(bottom)), (int(left), int(right)))
    # return np.pad(arr, pad_width=pad_width, mode='constant',
    #               constant_values=0)
    if sparse.issparse(mat):

        if right:
            m, n = mat.shape

            mat = sparse.hstack([mat, sparse.dok_matrix((m, 1))])
        if bottom:
            m, n = mat.shape
            mat = sparse.vstack([mat, sparse.dok_matrix((1, n))])
        if left:
            m, n = mat.shape
            mat = sparse.hstack([sparse.dok_matrix((m, 1)), mat])
        if top:
            m, n = mat.shape
            mat = sparse.vstack([sparse.dok_matrix((1, n), mat, )])
        return mat.todok()  # don't want to stay as coo # TODO do explicit casts where we need them
    else:
        if right:
            m, n = mat.shape
            mat = np.c_[mat, np.zeros((m, 1))]
        if bottom:
            m, n = mat.shape
            mat = np.r_[mat, np.zeros((1, n))]
        if left:
            m, n = mat.shape
            mat = np.c_[np.zeros((m, 1)), mat]
        if top:
            m, n = mat.shape
            mat = np.r_[np.zeros((1, n)), mat]
        return mat


class ModelDataStruct(object):
    """Generic struct which stores all the arc attributes together in a convenient manner.
    Additionally, if it hasn't already been done, the input data is padded with an additional
    row/ col to bottom right which will have the destination dummy arc mapped to.
    This is perhaps not particularly, clear but it is done here to avoid having to
    resize all the dependent quantities later

    """

    def __init__(self, data_attribute_list: List[sparse.dok_matrix],
                 incidence_matrix: sparse.dok_matrix, data_array_names_debug=None,
                 resize=True):
        r"""
        Instantiate a ModelDataStruct instance.

        Parameters
        ----------
        data_attribute_list : list of :py:class:`scipy.sparse.dok_matrix` or list of array like
            List of all network attribute matrices
        incidence_matrix : :py:class:`scipy.sparse.dok_matrix` or array like
            Network incidence matrix
        data_array_names_debug : list of str, optional
            List of plaintext descriptors for the network attributes, used for debug printing
        resize : bool
            Whether to resize matrix to include zero column for dest. Should be True, unless you
            know what you are doing.

        """
        if sparse.issparse(data_attribute_list[0]):
            self.is_data_format_sparse = True
        else:
            self.is_data_format_sparse = False
        # data list should be all sparse or all dense (there's probably a nicer way to
        # write this (ex-NOR)
        if (all([not sparse.issparse(i) for i in data_attribute_list])
                or all([sparse.issparse(i) for i in data_attribute_list])) is False:
            warnings.warn("Recieved a mix of sparse and dense data types in "
                          "data_attribute_list. Unexpected "
                          "behaviour may occur. Please specify either sparse matrices or "
                          "numpy arrays")

        # check if the bottom row and right col are empty, if so, we can store the dest in them,
        # if not, we need to append
        if sparse.issparse(incidence_matrix):
            self.is_incidence_mat_sparse = True
            nnz = (incidence_matrix[-1, :].count_nonzero()
                   + incidence_matrix[:, -1].count_nonzero())

        else:
            self.is_incidence_mat_sparse = False
            nnz = (np.count_nonzero(incidence_matrix[-1, :])
                   + np.count_nonzero(incidence_matrix[:, -1]))
        if self.is_data_format_sparse != self.is_incidence_mat_sparse:
            raise ValueError("Recieved one sparse/ dense network attributes and the opposite for "
                             "for incidence matrix. Please specify both as sparse or dense, "
                             "not a mix")

        if nnz > 0 and resize:
            print("Adding an additional row and column to house sink state.")
            incidence_matrix = _zero_pad_mat(incidence_matrix, bottom=True, right=True)
            data_matrix_list_new = []
            for i in data_attribute_list:
                data_matrix_list_new.append(_zero_pad_mat(i, bottom=True, right=True))
            data_attribute_list = data_matrix_list_new
            self.padded = True
        else:
            self.padded = False

        self.incidence_matrix = incidence_matrix.astype(int)

        self.data_array = np.array(data_attribute_list)
        if data_array_names_debug is None:
            data_array_names_debug = (),
        self.data_fields = data_array_names_debug  # convenience for debugging
        self.n_dims = len(self.data_array)


class RecursiveLogitModel(abc.ABC):
    """Abstraction of the linear algebra operations for the recursive logit
    model to solve the matrix system for the value functions.

    Is subclassed to provide functionality for prediction and estimation and
    should not be directly instantiated
    """

    zeros_error_override = None  # Hack attribute to make a test case relevant, should be removed
    # not a global class attribute, but presented here to indicate it is different from other
    # fields
    # TODO this is probably an okay way to implement the non positive value function toggle

    def __init__(self, data_struct: ModelDataStruct, initial_beta=-1.5, mu=1.0, safety_checks=True):
        """
        Initialises a RecursiveLogitModel instance.

        Parameters
        ----------
        data_struct : ModelDataStruct
            The ModelDataStruct corresponding to the network being estimated on
        initial_beta : float or int or list[float] or array_like
            The initial value for the parameter weights of the network attributes
        mu : float
            The scale parameter of the Gumbel random variables being modelled. Generally set
            equal to 1 as it is non-identifiable due to the uncertainty in the parameter weights.
        """

        self.network_data = data_struct  # all network attributes
        self.data_array = data_struct.data_array
        # flags for sparsity, save us rechecking. Realistically, there should only be one
        # flag since there is no use case where one is sparse and one is dense, but a hybrid
        # is current permitted in the tests
        self.is_network_data_sparse = self.network_data.is_data_format_sparse
        self._is_incidence_sparse = self.network_data.is_incidence_mat_sparse
        self.n_dims = len(self.data_array)
        self.mu = mu

        self._short_term_utility = None
        self._exponential_utility_matrix = None
        self._value_functions = None
        self._exp_value_functions = None

        self.flag_log_like_stored = False
        self.flag_exp_val_funcs_error = True

        self.n_log_like_calls = 0
        self.n_log_like_calls_non_redundant = 0

        # allow for specification of vector beta or scalar beta repeated
        if isinstance(initial_beta, (float, int)):
            beta_vec = np.array([initial_beta for _ in range(self.n_dims)])
        elif isinstance(initial_beta, (list, tuple)):
            if len(initial_beta) != self.n_dims:
                raise ValueError(f"beta vector must have length {self.n_dims} for given network "
                                 f"attributes.")
            beta_vec = np.array(initial_beta)
        else:
            beta_vec = initial_beta
        if np.any(beta_vec > 0):
            raise ValueError("Beta vector contains positive terms. Beta must be negative "
                             "so that the short term utilities are negative.")
        if len(beta_vec) != self.n_dims:
            raise ValueError("Beta must have same length as number of network attributes.\n"
                             f"Got len(beta)={len(beta_vec)} and # attrs = {self.n_dims}")
        # setup optimiser initialisation
        self._beta_vec = beta_vec
        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        if safety_checks:
            m_mat = self.get_exponential_utility_matrix()
            if len(m_mat[m_mat > 1e-10]) == 0:  # matrix is all zero
                raise ValueError("Matrix M is consists of only zeros\n  i.e. "
                                 "Some attribute is so large such that exp(-r(s,a;beta))=0."
                                 "Try reducing the corresponding coefficient beta_q.")
            if linalg.norm(_to_dense_if_sparse(m_mat)) < 1e-10:
                logger.warning("\nInitial values of beta are such that the short term utilities "
                               "are approaching zero.\n This means legal solutions of "
                               "value functions may not be possible.\n If optimiser does "
                               "not find a legal step quickly it will terminate in failure.\n"
                               " Try setting initial beta of smaller magnitude.")

    def get_beta_vec(self):
        """
        Get the current value of the network attribute parameters.

        Returns
        -------
        beta_vec : :py:func:`numpy.array` of float
        """
        return self._beta_vec

    def _compute_short_term_utility(self, skip_check=False) -> bool:
        # Data array is a np.ndarray, of either sparse matrices, or 2d nd-arrays.
        # So we check the sparsity of the first element. If the user puts mixed types in here
        # then errors are on them
        if self.is_network_data_sparse:
            self.short_term_utility = np.sum(self.get_beta_vec() * self.data_array, axis=0)
            # note axis=0 means that ndarrays will give a matrix result back, not a scalar
            # which is what we want
        else:
            # (beta * each_data_array_slice).sum(over all slices) is a tensor contraction
            self.short_term_utility = np.tensordot(self.get_beta_vec(), self.data_array, axes=1)

        if skip_check is False:
            if self.is_network_data_sparse:
                nz_rows, nz_cols = self.short_term_utility.nonzero()  # this is dense so cast ok
                condition = self.short_term_utility[nz_rows, nz_cols].toarray()
            else:
                condition = self.short_term_utility
            if np.any(condition > 0):
                # only a debug print since this occurs frequently without error, -> with bad line
                # search steps
                logger.debug("Short term utility contains positive terms, which is illegal. "
                             "Network attributes must be non-negative and beta must be "
                             "negative.")
                return False
        return True

    def get_short_term_utility(self):
        """
        Returns the matrix of short term utilities between all states for the current value of beta.

        In the mathematical notation this is :math:`[r(s,a)]` or :math:`[v(a|k)]` in the
        notation of Fosgerau.

        Returns
        -------
        short_term_utility: :py:class:`scipy.sparse.csr_matrix`
        """
        return self.short_term_utility

    def _compute_exponential_utility_matrix(self):
        """ Compute M_{ka} matrix which is not orig dependent, with empty bottom, right
        row/ col for dest.
        """

        # explicitly do need this copy since we modify m_mat
        m_mat = self.get_short_term_utility().copy()
        # note we currently use incidence matrix here, since this distinguishes the
        # genuine zero arcs from the absent arcs
        # (since data format has zero arcs, should not be a problem with good data)
        nonzero_entries = self.network_data.incidence_matrix.nonzero()
        # note dense cast is fine here, since is the dense cast of a dense selection from sparse
        # (note that sparse matrices don't have exp method defined on them because exp(0)=1
        m_mat[nonzero_entries] = np.exp(
            1 / self.mu * _to_dense_if_sparse(m_mat[nonzero_entries],
                                              self.is_network_data_sparse))
        self._exponential_utility_matrix = m_mat

    def get_exponential_utility_matrix(self):
        """
        Returns the matrix of exponentiated short term utilities between all states for the current
        value of beta.

        In the mathematical notation this is :math:`[M_{s,a}]`.

        Returns
        -------
        short_term_utility: :py:class:`scipy.sparse.csr_matrix`
        """

        return self._exponential_utility_matrix

    # @staticmethod
    def _compute_exp_value_function(self, m_tilde, data_is_sparse, return_pieces=False):
        """ Actual linear system solve without any error checking.
        Here is a static method, subclasses are not"""
        ncols = m_tilde.shape[1]
        if data_is_sparse:
            rhs = sparse.lil_matrix((ncols, 1))  # suppressing needless sparsity warning
            rhs[-1, 0] = 1
            # (I-M)z =b
            a_mat = sparse.identity(ncols) - m_tilde
            z_vec = splinalg.spsolve(a_mat, rhs)
            z_vec = z_vec.reshape(ncols, 1)  # rhs has to be (n,1)

        else:
            rhs = np.zeros((ncols, 1))
            rhs[-1, 0] = 1
            a_mat = np.identity(ncols) - m_tilde
            z_vec = linalg.solve(a_mat, rhs)
            # note shape is natively correct in the dense case, no need to worry
        if return_pieces:
            return a_mat, z_vec, rhs
        else:
            return z_vec

    def compute_value_function(self, m_tilde, data_is_sparse=None) -> bool:
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
            data_is_sparse = sparse.issparse(m_tilde)  # do this once rather than continual query

        a_mat, z_vec, rhs = self._compute_exp_value_function(m_tilde, return_pieces=True,
                                                             data_is_sparse=data_is_sparse)
        return self._value_function_checks(a_mat, z_vec, rhs)

    def _value_function_checks(self, a_mat, z_vec, rhs):
        error_flag = False  # start with no errors

        # if we have z values negative, or near zero, parameters
        # are infeasible since log will be complex or -infty
        if z_vec.min() <= min(-1e-10, OptimiserBase.NUMERICAL_ERROR_THRESH):
            # thresh on this?
            error_flag = True  # TODO abs and stuff once i fix tests
            return error_flag

        # Norm of residual
        # Note: Scipy sparse norm doesn't have a 2 norm so we use this
        # TODO note this is expensive, in practice we may want to switch to sparse frobenius
        #  norm element wise norm which would be convenient for sparse matrices
        # residual - i.e. ill conditioned solution
        # note that z_vec is dense so this should be dense without explicit cast
        elif np.any(~np.isfinite(z_vec)):
            print(f"W: Value function not finite (beta={self._beta_vec})")
            self.flag_exp_val_funcs_error = True
            error_flag = True
            return error_flag

        elif linalg.norm(
                np.array(a_mat @ z_vec - rhs)) > OptimiserBase.RESIDUAL:
            self.flag_exp_val_funcs_error = True
            logger.warning(
                "W: Value function solution is not exact, has residual."
                f" (beta={self._beta_vec})")
            error_flag = True
            # raise ValueError("value function solution does not satisfy system well.")
        else:  # No errors or non terminal errors
            z_vec = np.squeeze(z_vec.T)  # not required but convenient at this point !TODO
            zeroes = z_vec[z_vec == 0]
            # Not considered error, even though degenerate case.
            if len(zeroes) > 0:
                # print("W: Z contains zeros in it, so value functions are undefined
                #       for some nodes")
                val_funcs_tmp = z_vec.copy()
                val_funcs_tmp[val_funcs_tmp == 0] = np.nan
                # with np.errstate(invalid='ignore'):
                # At the moment I would prefer this crashes
                self._value_functions = np.log(val_funcs_tmp)
                error_flag = (True if self.zeros_error_override is None else
                              self.zeros_error_override)
                # TODO this error is uncaught in Tien Mai. I believe it has to be caught. Whilst
                #  if we only use a path based upon the value functions that have legal values,
                #  we are still informing the decision for beta based upon illegal values if we
                #  leave it.
                #   Change was introduced in @42f564e9, results in a number of test cases having
                #       invalid values. Should be reviewed

            else:
                self._value_functions = np.log(z_vec)
            self._exp_value_functions = z_vec  # TODO should this be saved onto OptimStruct?
        return error_flag

    @staticmethod
    def _apply_dest_column(dest_index, m_tilde, i_tilde):
        """Takes in a exponential short term utility matrix and incidence matrix
        with zero padding  bottom right columns and updates them to include
        the correct encoding for the destination column. This is a small task, but is
        used in both prediction and estimation so convenient to have in one spot. It is also
        something where alternative schemes are possible."""

        # Destination enforcing # TODO review these assumptions and nonzero dest util
        #                               note that dest util updates would need to update
        #                               short term util locally as well (just final col).
        #                               Reivew needs to change the inverse ops at end of loop

        # Allow positive value functions:
        if ALLOW_POSITIVE_VALUE_FUNCTIONS:
            m_tilde[dest_index, -1] = 1  # exp(v(a|k)) = 1 when v(a|k) = 0 # try 0.2
            i_tilde[dest_index, -1] = 1
        else:
            # ban positive value funtions
            m_tilde[dest_index, :] = 0.0
            i_tilde[dest_index, :] = 0
            m_tilde[dest_index, -1] = 1  # exp(v(a|k)) = 1 when v(a|k) = 0 # try 0.2
            i_tilde[dest_index, -1] = 1
        return m_tilde, i_tilde
        # TODO check if this can occur inplace without returns

    # TODO review signature after tests are updated
    @staticmethod
    def _revert_dest_column(dest_index, m_tilde, i_tilde, local_exp_util_mat, local_incidence_mat,
                            data_is_sparse, incidence_is_sparse):
        """Inverse method to apply dest col - so that we can undo changes without resetting the
        matrix. Should be such that applying apply then revert is an identity operation."""

        if data_is_sparse is None:
            data_is_sparse = sparse.issparse(m_tilde)
        if incidence_is_sparse is None:
            incidence_is_sparse = sparse.issparse(local_incidence_mat)

        if ALLOW_POSITIVE_VALUE_FUNCTIONS:
            # Allow positive value functions
            m_tilde[dest_index, -1] = 0
            i_tilde[dest_index, -1] = 0
        else:
            # Ban positive value functions code:
            # legacy for old understanding which required negative value functions
            # TODO the conditional to dense casts here are to avoid a bug in scipy with slice
            #  indexing a zero sparse matrix
            # This is patched in @ 353f256, PR #12830, currently merged to master and due to
            # fix in Scipy 1.6, but is also marked with backport tag
            if data_is_sparse and local_exp_util_mat[dest_index, :].count_nonzero() == 0:
                m_tilde[dest_index, :] = 0
            else:  # if this only has zeros in rhs, assignment does nothing
                m_tilde[dest_index, :] = local_exp_util_mat[dest_index, :]
            if incidence_is_sparse and local_incidence_mat[dest_index, :].count_nonzero() == 0:
                i_tilde[dest_index, :] = 0
            else:
                i_tilde[dest_index, :] = local_incidence_mat[dest_index, :]
            # TODO remove this - being extra paranoid since this is a bug with scipy, not my code
            assert np.all(
                _to_dense_if_sparse(m_tilde[dest_index, :], data_is_sparse)
                == _to_dense_if_sparse(local_exp_util_mat[dest_index, :]), incidence_is_sparse)

        return m_tilde, i_tilde


class RecursiveLogitModelPrediction(RecursiveLogitModel):
    """Subclass which generates simulated observations based upon the supplied beta vector.
    Uses same structure as estimator in the hopes I can unify these such that one can use the
    estimator for prediction as well (probably have estimator inherit from this)"""

    def _check_index_valid(self, indices):
        max_index_present = np.max(indices)
        m, n = self.network_data.incidence_matrix.shape
        max_legal_index = m - 1  # zero based
        max_legal_index -= 1  # also subtract padding column from count
        if max_index_present > max_legal_index:
            if max_index_present == max_legal_index + 1:
                raise IndexError("Received observation index "
                                 f"{max_index_present} > {max_legal_index}. "
                                 f"Network data does have valid indices [0, ..., {m-1}] "
                                 f"but the final "
                                 "index is reserved for internal dummy sink state. The "
                                 "dimensions of the original data passed in would have been "
                                 "augmented, or data already had an empty final row and col.")
            raise IndexError("Received observation index "
                             f"{max_index_present} > {max_legal_index}. Can only simulate "
                             "observations from indexes which are in the model.")

    def generate_observations(self, origin_indices, dest_indices, num_obs_per_pair, iter_cap=1000,
                              rng_seed=None,
                              ):
        """

        Parameters
        ----------
        origin_indices : list or array like
            iterable of indices to start paths from
        dest_indices : list or array like
            iterable of indices to end paths at
        num_obs_per_pair : int
            Number of observations to generate for each OD pair
        iter_cap : int
            iteration cap in the case of non convergent value functions.
            Hopefully should not occur but may happen if the value function solutions
            are negative and ill conditioned.
        rng_seed : int or np.random.BitGenerator
             (any legal input to np.random.default_rng() )

        Returns
        -------
        output_path_list : list of list of int
        List of lists containing all observations generated

        """
        self._check_index_valid(dest_indices)
        self._check_index_valid(origin_indices)
        rng = np.random.default_rng(rng_seed)

        # store output as list of lists, using AwkwardArrays might be more efficient,
        # but the output format will not be the memory bottleneck
        output_path_list = []

        local_short_term_util = self.get_short_term_utility().copy()

        local_exp_util_mat = self.get_exponential_utility_matrix()
        local_incidence_mat = self.network_data.incidence_matrix
        m_tilde = local_exp_util_mat.copy()
        i_tilde = local_incidence_mat.copy()

        m, n = m_tilde.shape
        assert m == n  # paranoia

        # destination dummy arc is the final column (which was zero until filled)
        dest_dummy_arc_index = m-1

        for dest in dest_indices:
            self._apply_dest_column(dest, m_tilde, i_tilde)

            z_vec = self._compute_exp_value_function(m_tilde, self.is_network_data_sparse)
            with np.errstate(divide='ignore', invalid='ignore'):
                value_funcs = np.log(z_vec)
            # catch errors manually
            if np.any(~np.isfinite(value_funcs)):  # any infinite (nan/ -inf)
                if np.any(~np.isfinite(z_vec)):
                    msg = "exp(V(s)) contains nan or infinity"
                elif np.any(z_vec <= -1e-10):
                    # print(z_vec)
                    msg = "exp(V(s)) contains negative values (likely |beta| too small)"
                elif np.linalg.norm(z_vec[:-1]) <= 1e-10 and np.allclose(z_vec[-1], 1.0):
                    msg = "M is numerically zero for given beta"
                elif np.any(np.abs(z_vec) < 1e-10):
                    msg = ("V(s) diverges for some s, since z_s contains zeros"
                           " (likely |beta| too large, not always true)")
                else:
                    msg = "Unknown cause"
                raise ValueError("RLPrediction: Parameter Beta is incorrectly determined, "
                                 "or poorly chosen.\n Value functions cannot be solved"
                                 f" [{msg}].")

            elif ALLOW_POSITIVE_VALUE_FUNCTIONS is False and np.any(value_funcs > 0):
                warnings.warn("WARNING: Positive value functions:"
                              f"{value_funcs[value_funcs > 0]}")

            # loop through path starts, with same base value functions
            for orig in origin_indices:
                if orig == dest:  # redundant case
                    continue

                # repeat until we have specified number of obs
                for i in range(num_obs_per_pair):
                    # while we haven't reached the dest

                    current_arc = orig
                    # format is [dest, orig, l1, l2, ..., ln, dest]
                    # why? so we know what the dest is without having to lookup to the
                    # last nonzero (only really a problem for sparse, but still marginally
                    # more efficient. Perhaps could have flags for data formats? TODO
                    current_path = [int(dest)]
                    count = 0
                    while current_arc != dest_dummy_arc_index:  # index of augmented dest arc
                        count += 1
                        # TODO Note the int conversion here is purely for json serialisation compat
                        #   if this is no longer used then we can keep numpy types
                        current_path.append(int(current_arc))
                        if count > iter_cap:

                            raise ValueError(f"Max iterations reached. No path from {orig} to "
                                             f"{dest} was found within cap.")
                        current_incidence_col = i_tilde[current_arc, :]
                        # all arcs current arc connects to (index converts from 2d to 1d)
                        if self._is_incidence_sparse:
                            # we get a matrix return type for 1 row * n cols
                            # from .nonzero() - [0]th component is the row (always zero)
                            # and [1]th component is col,
                            neighbour_arcs = current_incidence_col.nonzero()[1]
                        else:
                            # nd-arrays are smart enough to work out they are 1d
                            neighbour_arcs = current_incidence_col.nonzero()[0]

                        # TODO it could be cheaper to block generate these in larger batches
                        eps = rng.gumbel(loc=-np.euler_gamma, scale=1, size=len(neighbour_arcs))

                        value_functions_observed = (
                                local_short_term_util[current_arc, neighbour_arcs]
                                + value_funcs[neighbour_arcs].T
                                + eps)

                        if np.any(np.isnan(value_functions_observed)):
                            raise ValueError("beta vec is invalid, gives non real solution")

                        next_arc_index = np.argmax(value_functions_observed)
                        next_arc = neighbour_arcs[next_arc_index]
                        current_arc = next_arc
                        # offset stops arc number being recorded as zero

                    if len(current_path) <= 2:
                        continue  # this is worthless information saying we got from O to D in one
                        # step
                    # Note we don't append the final (dummy node) because it changes location
                    output_path_list.append(current_path)

            # Fix the columns we changed for this dest (cheaper than refreshing whole matrix)
            self._revert_dest_column(dest, m_tilde, i_tilde,
                                     local_exp_util_mat, local_incidence_mat,
                                     self.is_network_data_sparse, self._is_incidence_sparse)

        return output_path_list


class RecursiveLogitModelEstimation(RecursiveLogitModelPrediction):
    """Extension of `RecursiveLogitModel` to support Estimation of network attribute parameters.
    """

    # TODO this should subclass prediction so that we can optimise into prediction

    def __init__(self, data_struct: ModelDataStruct,
                 optimiser: OptimiserBase, observations_record,
                 initial_beta=-1.5, mu=1, sort_obs=True):
        """
        Initialise recursive logit model for estimation

        Parameters
        ----------
        data_struct : ModelDataStruct
            containing network attributes of desired network
        optimiser : ScipyOptimiser or LineSearchOptimiser
            The wrapper instance for the desired optimisation routine
        observations_record : ak.highlighlevel.Array or :py:class:`scipy.sparse.spmatrix`
            or list of list

            record of observations to estimate from
        initial_beta : float or list or array like
            initial guessed values of beta to begin optimisation algorithm with. If a scalar,
            beta are uniformly initialised to this value
        mu :
            The scale parameter of the Gumbel random variables being modelled. Generally set
           equal to 1 as it is non-identifiable due to the uncertainty in the parameter weights.
        sort_obs : bool
            flag to sort input observations to allow for efficient caching. Should only be set to
            False if the data is large and already known to be sorted.
        """
        super().__init__(data_struct, initial_beta, mu)
        self._init_estimation_body(data_struct, optimiser, observations_record, sort_obs)
        self._init_post_init()

    def _init_estimation_body(self, data_struct: ModelDataStruct,
                              optimiser: OptimiserBase, observations_record, sort_obs=True):
        """Factor component out for the sake of inheritance"""
        self.optimiser = optimiser  # optimisation alg wrapper class

        beta_vec = super().get_beta_vec()  # orig without optim tie in
        # setup optimiser initialisation
        self.optim_function_state = OptimFunctionState(None, None, np.identity(data_struct.n_dims),
                                                       self.optimiser.hessian_type,
                                                       self.eval_log_like_at_new_beta,
                                                       beta_vec,
                                                       self._get_n_func_evals)
        self.update_beta_vec(beta_vec, from_init=True)  # to this to refresh dependent matrix
        # quantitites

        # book-keeping on observations record
        if sparse.issparse(observations_record):  # TODO perhaps convert to ak format?
            self.is_obs_record_sparse = True
            self.obs_count, _ = observations_record.shape
            self.obs_min_legal_index = 1  # Zero is reserved index for sparsity

        else:
            self.is_obs_record_sparse = False
            observations_record = self._convert_obs_record_format(observations_record)
            # num_obs = len(observations_record) # equivalent but clearer this is ak array
            self.obs_count = ak.num(observations_record, axis=0)

            # self.obs_min_legal_index should be zero, and it is legal to index from zero since
            # it's not being used for sparsity. There are exception cases if data is strange:
            #       i.e. json min index is 1, and hasn't been produced by my code
            #       or converted from a sparse matrix input

            # if observations have the same length always, suspect they've been converted
            # from numpy / scipy with zero padding
            inner_dim_lengths = ak.num(observations_record, axis=1)
            if np.all(inner_dim_lengths == np.max(inner_dim_lengths)):  # all the same shape
                # format must be [d, o, l1, ..., ln, d] and may be zero padded
                # if any end with zero, but don't start with zero then the zero isn't indicating
                # the dest and we must be using padding. Then zero is an illegal index
                if np.any((observations_record[:, -1] == 0) & (observations_record[:, 0] != 0)):
                    self.obs_min_legal_index = 1
                else:
                    self.obs_min_legal_index = ak.min(observations_record, axis=None)

            else:  # just get the minimum index used, which should be zero or 1
                self.obs_min_legal_index = ak.min(observations_record, axis=None)
        if sort_obs:
            self.obs_record = self._sort_obs(observations_record)
        else:
            self.obs_record = observations_record

    def _init_post_init(self):
        # finish initialising
        self.get_log_likelihood()  # need to compute starting LL for optimiser
        optimiser = self.optimiser
        if isinstance(optimiser, CustomOptimiserBase):
            optimiser.set_beta_vec(self._beta_vec)
            optimiser.set_current_value(self.log_like_stored)

        self._path_start_nodes = None
        self._path_finish_nodes = None

    @staticmethod
    def _convert_obs_record_format(observations_record) -> ak.highlevel.Array:
        if isinstance(observations_record, ak.highlevel.Array):
            return observations_record

        elif isinstance(observations_record, list):
            if all(isinstance(i, list) for i in observations_record):
                try:
                    return ak.from_iter(observations_record)
                except Exception as e:  # TODO BAD
                    raise TypeError("Obs format invalid, failed to parse input obs "
                                    "as Awkward Array.") from e
            else:
                raise TypeError("List observation format must contain list of lists, "
                                "not a singleton list")

        # else we blindly try to convert
        try:
            return ak.from_iter(observations_record)
        except Exception as e:  # TODO BAD
            raise TypeError("Obs format invalid, failed to parse input obs as Awkward Array.") \
                from e

    @staticmethod
    def _sort_obs(obs_record):
        """
        Sorts input observations by destination, to allow efficient computation of log likelihood.
        This allows us to cache some value functions and not recomputed them
        Recall that destination is the first index of an observations.

        Parameters
        ----------
        obs_record : ak.highlevel.Array or array like

        Returns
        -------
        sorted_obs_record : observations record sorted by destination.
        """
        if isinstance(obs_record, ak.highlevel.Array):
            # sort by first index
            obs_record = obs_record[ak.argsort(obs_record[:, 0])]
        else:  # assuming sparse, but should work for dense
            index = np.argsort(obs_record[:, 0].toarray().squeeze())  # squeeze is since we have
            # redundant 2d and we end up with all zeros in redundant dim
            obs_record = obs_record[index, :]
        return obs_record

    def _get_n_func_evals(self):
        return self.n_log_like_calls_non_redundant

    def solve_for_optimal_beta(self, verbose=True, output_file=None):
        """
        Executes the optimisation algorithm specified in the init to determine the most likely
        parameter values based upon the input observations

        Parameters
        ----------
        verbose : bool
            Flag for printing of output
        output_file : str or os.path.PathLike, optional
            file to send verbose output to

        Returns
        -------
        beta_vec :  :py:func:`numpy.array`
            The optimal determined vector of parameters
        """
        # if a scipy method then it is self contained and we don't need to query inbetween loops
        if isinstance(self.optimiser, ScipyOptimiser):
            optim_res = self.optimiser.solve(self.optim_function_state, verbose=verbose,
                                             output_file=output_file
                                             )
            if verbose:
                print(optim_res)
            if optim_res.success is False:
                raise ValueError("Scipy alg error flag was raised. Process failed. Details:\n"
                                 f"{optim_res.message} ")
            return optim_res.x
        # otherwise we have Optimiser is a Custom type
        print(self.optimiser.get_iteration_log(self.optim_function_state), file=None)
        n = 0

        while n <= 1000:  # TODO magic constant
            n += 1
            if self.optimiser.METHOD_FLAG == OptimType.LINE_SEARCH:
                ok_flag, hessian, log_msg = self.optimiser.iterate_step(self.optim_function_state,
                                                                        verbose=False,
                                                                        output_file=None,
                                                                        debug_counter=n)
                if ok_flag:
                    print(log_msg)
                else:
                    raise ValueError("Line search error flag was raised. Process failed.")
            else:
                raise NotImplementedError("Only have line search implemented")
            # check stopping condition
            is_stopping, stop_type, is_successful = self.optimiser.check_stopping_criteria()

            if is_stopping:
                print(f"The algorithm stopped due to condition: {stop_type}")
                return self.optim_function_state.beta_vec
        raise ValueError("Optimisation algorithm failed to converge")

    def update_beta_vec(self, new_beta_vec, from_init=False) -> bool:
        """
        Update the interval value for the network parameters beta with the supplied value.


        Parameters
        ----------
        new_beta_vec : :py:func:`numpy.array` of float
        from_init : bool
            flag used in subclasses to avoid messy dependence sequencing issues


        Returns
        -------
        error_flag : bool
            Returns a boolean flag to indicate if updating beta was successful. This fails for an
            ill chosen beta which either is positive and illegal, or is large in magnitude such
            that the short term utility calculation overflows.

            This flag is likely not used by an end user, only the internal code
        """
        # Prevent redundant computation
        if np.all(new_beta_vec == self._beta_vec):
            return True
        self.flag_log_like_stored = False
        self.optim_function_state.beta_vec = new_beta_vec
        self._beta_vec = new_beta_vec

        success_flag = self._compute_short_term_utility()
        if success_flag is False:
            return success_flag
        self._compute_exponential_utility_matrix()
        return True

    def _return_error_log_like(self):
        self.log_like_stored = OptimiserBase.LL_ERROR_VALUE
        self.grad_stored = np.ones(self.n_dims)  # TODO better error gradient?
        self.flag_log_like_stored = True
        # print("Parameters are infeasible.")
        return self.log_like_stored, self.grad_stored

    def get_log_likelihood(self):
        """Compute the log likelihood of the data  and its gradient for the current beta vec

        Main purpose is to update internal state however also returns the negative of these two
        quantities

        Returns
        -------
        log_like_stored:  float
            The current negative log likelihood. Note that it is negative to be consistent
            with a minimisation problem
        self.grad_stored  :py:func:`numpy.array` of float
            The current negative gradient of the log likelihood. Note that it is negative to be
            consistent with a minimisation problem
        """
        self.n_log_like_calls += 1  # TODO remove this counter?
        if self.flag_log_like_stored:
            return self.log_like_stored, self.grad_stored
        self.n_log_like_calls_non_redundant += 1

        obs_record = self.obs_record
        # local references with idomatic names
        n_dims = self.n_dims  # number of attributes in data
        mu = self.mu
        v_mat = self.get_short_term_utility()
        m_mat = self.get_exponential_utility_matrix()
        local_incidence_mat = self.network_data.incidence_matrix
        # local copies which have dest column connections applied to
        m_tilde = m_mat.copy()
        i_tilde = local_incidence_mat.copy()

        m, n = m_tilde.shape
        assert m == n  # paranoia

        mu_log_like_cumulative = 0.0  # weighting of all observations
        mu_ll_grad_cumulative = np.zeros(n_dims)  # gradient combined across all observations

        # iterate through observation number
        dest_index_old = -np.nan
        for n in range(self.obs_count):
            # TODO we should be sorting by dest index to avoid recomputation
            #   if dests are the same we don't need to recompute value functions
            # Should also sort on orig index, because then we could reuse this too, but
            # benefit might not be worth the sort cost
            # a[ak.argsort(a[:,0]) or a[np.argsort(a[:,0])
            # Subtract 1 since we expect dests are 1,2,..., without zero, subtract to align into
            # zero arrays. This is in part left over from sparse array obs format,
            # which should probably be deprecated in favour of more efficient formats only.
            dest_index = obs_record[n, 0] - self.obs_min_legal_index
            orig_index = obs_record[n, 1] - self.obs_min_legal_index

            # actually need to do this unless I cache M and reset at the top of the function call
            # tODO this would be free minor speed I think
            m_tilde, i_tilde = self._apply_dest_column(dest_index, m_tilde, i_tilde)

            # value functions are unchanged so we can reuse them
            if dest_index == dest_index_old:
                value_funcs, exp_val_funcs = self._value_functions, self._exp_value_functions
                # TODO if this holds and the origin is the same as before we can reuse grad V(s_0)
            else:

                # Now get exponentiated value funcs
                error_flag = self.compute_value_function(m_tilde, self.is_network_data_sparse)
                # If we had numerical issues in computing value functions
                if error_flag:  # terminate early with error vals
                    return self._return_error_log_like()

                value_funcs, exp_val_funcs = self._value_functions, self._exp_value_functions

            dest_index_old = dest_index
            # Gradient and log like depend on origin values

            grad_orig = self.get_value_func_grad_orig(orig_index, m_tilde, exp_val_funcs)

            self._compute_obs_path_indices(obs_record[n, :])  # required to compute LL & grad
            # LogLikeGrad in Code doc
            mu_gradient_current_obs = self._compute_current_obs_mu_ll_grad(grad_orig)
            # LogLikeFn in Code Doc - for this n
            mu_log_like_obs = self._compute_current_obs_mu_log_like(v_mat, value_funcs[orig_index])

            # This is the aggregation in Tien Mai's code. Best interpretation is some kind of
            # successive over relaxation/ momentum/ learning rate except that the gradient
            # computation isn't batched, nor are observations shuffled, so it doesn't really make
            # sense
            # mu_log_like_cumulative += (mu_log_like_obs - mu_log_like_cumulative) / (n + 1)
            # mu_ll_grad_cumulative += (mu_gradient_current_obs - mu_ll_grad_cumulative) / (n + 1)

            # compute cumulative totals in the natural, standard mathematical way.
            mu_log_like_cumulative += mu_log_like_obs
            mu_ll_grad_cumulative += mu_gradient_current_obs

            # Put our matrices back untouched:
            m_tilde, i_tilde = self._revert_dest_column(dest_index, m_tilde, i_tilde,
                                                        m_mat, local_incidence_mat,
                                                        self.is_network_data_sparse,
                                                        self._is_incidence_sparse)

        # only apply this rescaling once rather than on all terms inside loops
        self.log_like_stored = -1/mu * mu_log_like_cumulative
        self.grad_stored = -1/mu * mu_ll_grad_cumulative

        self.optim_function_state.value = self.log_like_stored
        self.optim_function_state.grad = self.grad_stored
        self.flag_log_like_stored = True
        return self.log_like_stored, self.grad_stored

    def eval_log_like_at_new_beta(self, beta_vec):
        """update beta vec and compute log likelihood in one step - used for lambdas
        Effectively a bad functools.partial"""
        if np.any(np.isnan(beta_vec)):
            raise ValueError(f"Input beta vector {beta_vec} contains nans. Algorithm can no "
                             f"longer converge.")
        success_flag = self.update_beta_vec(beta_vec)
        if success_flag is False:
            self.log_like_stored = OptimiserBase.LL_ERROR_VALUE
            self.grad_stored = np.ones(self.n_dims)  # TODO better error gradient?
            self.flag_log_like_stored = True
            return self.log_like_stored, self.grad_stored
        x = self.get_log_likelihood()
        if np.isnan(x[0]):
            raise ValueError("unexpected nan")
        return x

    def _compute_obs_path_indices(self, obs_row: Union[sparse.dok_matrix,
                                                       ak.highlevel.Array]):
        """Takes in the current observation sequence.
        Returns the vectors of start positions and end positions in the provided observation.
        This is used in computation of the particular observations log likelihood and its
        gradient. This essentially slices indices from [:-1] and from [1:] with
        some extra index handling included. Cache values since typically called twice in
        sequence. Changed so that this is called once, but second call assumes this function
        has been called and values are set appropriately."""
        # obs_row note this is a shape (1,m) array not (m,)
        # know all zeros are at the end since row is of the form:
        #   [dest, orig, ... , dest, 0 padding since sparse]

        if self.is_obs_record_sparse:
            path_len = obs_row.count_nonzero()
            # Note that this is now dense, so cast isn't so bad (required since subtract not
            # defined for sparse matrices)
            # start at 1 to omit the leading dest node marking
            path = obs_row[0, 1:path_len].toarray().squeeze()

        elif isinstance(obs_row, ak.highlevel.Array):
            path = obs_row[1:]
            path = ak.to_numpy(path)

        else:
            raise ValueError("obs record has unsupported type")

        # if np.any(path != self._prev_path): # infer I was planning caching, unused though #TODO

        min_legal_index = self.obs_min_legal_index
        # in a sequence of obs [o, 1, 2, 3, 4, 5, 6, 7, 8, 9, d]
        # we have the transitions
        #    [(o, 1), (1, 2), (2, 3), (3, 4), ...] or
        # origins [o, 1, 2, 3, 4, ..., 9] and dests [1, 2, 3, ..., 9, d]
        # all nodes i in (i ->j) transitions
        # subtract min_legal to get 0 based indexes if we were read sparse matrices
        path_arc_start_nodes = path[:-1] - min_legal_index
        path_arc_finish_nodes = path[1:] - min_legal_index

        # tODO think there should be an extra -1 (one for zero based, one for pad col) in every
        #  case but would need to review to make sure.
        #  On review, this should be checked globally on obs_record, there is a global max index
        final_index_in_data = np.shape(self.network_data.incidence_matrix)[0] - 1

        if np.any(path_arc_finish_nodes > final_index_in_data):
            raise ValueError("Observation received contains indexes larger than the dimension"
                             "of the network matrix. This means input data is somehow "
                             "inconsistent.")

        self._path_start_nodes = path_arc_start_nodes
        self._path_finish_nodes = path_arc_finish_nodes
        return self._path_start_nodes, self._path_finish_nodes

    def _compute_current_obs_mu_log_like(self, v_mat, value_func_orig):
        r"""
        Compute the log likelihood function for the currently observed path.
        "Current" in this sense stipulates that compute_obs_path_indices has been called prior
        in order to update these indices.
        Log likelihood is given by
        ..math::
            \frac{1}{\sigma}\sum_{n=1}^N\bigg[\sum_{i=0}^{I_n-1}\Big(r(s_i^n, s_^n_{i+1})\Big) -V(
            s_0^n) \Bigg]
        This function computes this inner summation - i.e. for fixed n.

        """
        arc_start_nodes = self._path_start_nodes
        arc_finish_nodes = self._path_finish_nodes

        # index out v[(o,i1), (i1,i2), (i2, i3) ... (in,d)] then sum (all the short term contrib.)
        sum_inst_util = v_mat[arc_start_nodes, arc_finish_nodes].sum()
        log_like_obs = sum_inst_util - value_func_orig  # LogLikeFn in Code Doc
        return log_like_obs

    def _compute_current_obs_mu_ll_grad(self, grad_orig):
        """# LogLikeGrad in Code doc
        Compute the log likelihood function gradient for the currently observed path.
        "Current" in this sense stipulates that compute_obs_path_indices has been called prior
        in order to update these indices."""
        # subtract 1 to get 0 based indexes
        arc_start_nodes = self._path_start_nodes
        arc_finish_nodes = self._path_finish_nodes
        # recall n_dims is number of network attributes
        sum_current_attr = np.zeros(self.n_dims)  # sum of observed attributes
        for attr in range(self.n_dims):  # small number of dims so inexpensive
            sum_current_attr[attr] = self.network_data.data_array[attr][
                arc_start_nodes, arc_finish_nodes].sum()

        return sum_current_attr - grad_orig  # LogLikeGrad in Code doc

    def get_value_func_grad_orig(self, orig_index, m_tilde, exp_val_funcs):
        """Note relies on 'self' for the attribute data.
        Computes the gradient of the value function evaluated at the origin
        Named GradValFnOrig in companion notes, see also GradValFn2 for component part
        - this function applies the 1/z_{orig} * [rest]_{evaluated at orig}
        """
        partial_grad = self._get_value_func_incomplete_grad(m_tilde, exp_val_funcs)
        # with np.errstate(divide='ignore', invalid='ignore'):
        return partial_grad[:, orig_index] / exp_val_funcs[orig_index]

    def _get_value_func_incomplete_grad(self, m_tilde, exp_val_funcs):
        r"""
        Function to compute GradValFn2 from algorithm chapter
        \pderiv{\bm{V}}{\beta_q}
                & =\frac{1}{\bm{z}}\circ
                (I - \bm{M})^{-1}\paren{\bm{M}\circ \chi^q}  \bm{z}

                without the leading 1/z term as a matrix for all q and k.
                We compute this quantity
                since in order to evaluate the gradient at the orig,
                we need to compute the mvp term, but not the elementwise product
                (we only need the row at the orig) so this is slightly more efficient.

        Computes gradient of value function with respect to beta, returns a matrix,
        for each row of V and for each beta
        \equiv \frac{\partial V}{\partial \beta}, which is a matrix.

        We are mainly concerned with the value of \frac{\partial V(k_0^n)}{\partial_\beta}
        as this appears in the gradient of the log likelihood
        Returns a data.n_dims * shape(M)[0] matrix
        - gradient in each x component at every node"""
        # TODO check if dimensions should be transposed
        grad_v = np.zeros((self.n_dims, np.shape(m_tilde)[0]))
        z = exp_val_funcs  # consistency with maths doc

        # low number of dims -> not vectorised for convenience
        # (actually easy to fix now)
        for q in range(self.n_dims):
            chi = self.data_array[q]  # current attribute of data
            # Have experienced numerical instability with spsolve, but believe this is
            # due to matrices with terrible condition numbers in the examples
            # spsolve(A,b) == inv(A)*b
            # Note: A.multiply(B) is A .* B for sparse matrices
            if self.is_network_data_sparse:
                identity = sparse.identity(np.shape(m_tilde)[0])
                grad_v[q, :] = splinalg.spsolve(identity - m_tilde, m_tilde.multiply(chi) @ z)
            else:
                identity = np.identity(np.shape(m_tilde)[0])
                grad_v[q, :] = linalg.solve(identity - m_tilde, m_tilde * chi @ z)

        return grad_v
