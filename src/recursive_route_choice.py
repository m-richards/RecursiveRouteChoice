import abc
import warnings
from typing import List, Union
import scipy
from scipy import linalg
# from scipy.sparse import coo_matrix
from scipy.sparse import linalg as splinalg
from scipy import sparse

import awkward1 as ak

from data_loading import load_standard_path_format_csv
from data_processing import AngleProcessor
# from debug_helpers import print_sparse, print_data_struct
from optimisers.extra_optim import OptimFunctionState
from optimisers.optimisers_file import CustomOptimiserBase, OptimType, ScipyOptimiser, OptimiserBase
import numpy as np

ALLOW_POSITIVE_VALUE_FUNCTIONS = False


def _to_dense_if_sparse(mat):
    if sparse.issparse(mat):
        return mat.toarray()
    else:
        return mat


def _zero_pad_mat(mat, top=False, left=False, bottom=False, right=False):
    """Abstracted since this will get done a fair bit and this is convenient way of doing it but
    perhaps not the fastest"""
    # pad_width =((int(top), int(bottom)), (int(left), int(right)))
    # return np.pad(arr, pad_width=pad_width, mode='constant',
    #               constant_values=0)
    if scipy.sparse.issparse(mat):

        if right:
            m, n = mat.shape

            # print(mat.shape, np.zeros(m).shape)
            # print(mat, type(mat))
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
            # print(mat.shape, np.zeros((m, 1)).shape)
            # print(mat, type(mat))
            mat = np.c_[mat, np.zeros((m, 1))]
        if bottom:
            m, n = mat.shape
            # print(mat.shape, np.zeros((1, n)).shape)
            # print(mat, type(mat))
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

    def __init__(self, data_matrix_list: List[scipy.sparse.dok_matrix],
                 incidence_matrix: scipy.sparse.dok_matrix, data_array_names_debug=None,
                 resize=True):

        # check if the bottom row and right col are empty, if so, we can store the dest in them,
        # if not, we need to append
        if sparse.issparse(incidence_matrix):
            nnz = (incidence_matrix[-1, :].count_nonzero()
                   + incidence_matrix[:, -1].count_nonzero())
        else:
            nnz = (np.count_nonzero(incidence_matrix[-1, :])
                   + np.count_nonzero(incidence_matrix[:, -1]))

        if nnz > 0 and resize:
            print("resizing to include zero pad")
            incidence_matrix = _zero_pad_mat(incidence_matrix, bottom=True, right=True)
            data_matrix_list_new = []
            for i in data_matrix_list:
                data_matrix_list_new.append(_zero_pad_mat(i, bottom=True, right=True))
            data_matrix_list = data_matrix_list_new

        self.incidence_matrix = incidence_matrix

        self.data_array = np.array(data_matrix_list)
        if data_array_names_debug is None:
            data_array_names_debug = (),
        self.data_fields = data_array_names_debug  # convenience for debugging
        self.n_dims = len(self.data_array)


class RecursiveLogitDataStructDeprecated(object):
    """Generic struct which stores all the arc attributes together in a convenient manner.
    Also provides convenience constructors.
    # TODO want to kill this in favour of some kind of preprocessing and then have this totally
    generic
    """

    def __init__(self, travel_times: scipy.sparse.dok_matrix,
                 incidence_matrix: scipy.sparse.dok_matrix, turn_angle_mat=None):
        self.travel_times = travel_times
        self.incidence_matrix = incidence_matrix
        self.turn_angle_mat = turn_angle_mat
        # TODO think perhaps no specific labels should be here, that these should be done by some
        #  other preprocessing step before this class and
        #  this should be a generic data class.
        #  On review, easiest to have this non generic for now, but could easily
        #  move init to classmethods. Wait for real data to see.
        #  Should at least have methods to allow generic initialisation
        self.data_array = np.array([self.travel_times])
        self.data_fields = ['travel_time']  # convenience for debugging
        self.n_dims = len(self.data_array)

        self.has_categorical_turns = False
        self.has_nz_incidence_mat = False
        self.has_turn_angles = False

    def add_second_travel_time_for_testing(self):
        """Want a non 1d test case without dealing with angles"""
        self.data_array = np.array([self.travel_times, self.travel_times])
        self.data_fields = ['travel_time', 'travel_time']
        self.n_dims = len(self.data_array)

    def add_left_turn_incidence_uturn_for_comparison(self, left_turn_thresh=None,
                                                     u_turn_thresh=None):
        """Temporary method for comparison with Tien Mai's code to add data matrices in the same
        order, shouldn't be used generally. Messy combination of
            add_turn_categorical_variables and add_nonzero_arc_incidence methods."""
        nz_arc_incidence = (self.travel_times > 0).astype('int').todok()
        if self.has_categorical_turns:
            return
        self.has_categorical_turns = True
        if self.turn_angle_mat is None:
            raise ValueError("Creating categorical turn matrices failed. Raw turn angles matrix "
                             "must be supplied in the constructor")
        left_turn_dummy = AngleProcessor.get_left_turn_categorical_matrix(self.turn_angle_mat,
                                                                          left_turn_thresh,
                                                                          u_turn_thresh)
        u_turn_dummy = AngleProcessor.get_u_turn_categorical_matrix(self.turn_angle_mat,
                                                                    u_turn_thresh)
        self.data_array = np.concatenate(
            (self.data_array, np.array((left_turn_dummy, nz_arc_incidence, u_turn_dummy)))
        )
        self.n_dims = len(self.data_array)
        self.has_turn_angles = True
        self.has_nz_incidence_mat = True
        self.data_fields.extend(("left_turn_dummy", "nonzero_arc_incidence", "u_turn_dummy"))

    def add_turn_categorical_variables(self, left_turn_thresh=None, u_turn_thresh=None):
        """Uses the turn matrix in the constructor and splits out into categorical matrices
        for uturns and left uturns.

        Note left turn_thresh is negative in radians, with angles lesser to the left,
        and u_turn thresh is positive in radians, less than pi.
        """
        if self.has_categorical_turns:
            return
        self.has_categorical_turns = True
        if self.turn_angle_mat is None:
            raise ValueError("Creating categorical turn matrices failed. Raw turn angles matrix "
                             "must be supplied in the constructor")
        left_turn_dummy = AngleProcessor.get_left_turn_categorical_matrix(self.turn_angle_mat,
                                                                          left_turn_thresh,
                                                                          u_turn_thresh)
        u_turn_dummy = AngleProcessor.get_u_turn_categorical_matrix(self.turn_angle_mat,
                                                                    u_turn_thresh)
        self.data_array = np.concatenate(
            (self.data_array, np.array((left_turn_dummy, u_turn_dummy)))
        )
        self.n_dims = len(self.data_array)
        self.has_turn_angles = True
        self.data_fields.extend(("left_turn_dummy", "u_turn_dummy"))

    def add_nonzero_arc_incidence(self):
        """Adds an incidence matrix which is only 1 if the additional condition that the arc is
            not of length zero is met. This encoded in the "LeftTurn" matrix in Tien's code"""
        nz_arc_incidence = (self.travel_times > 0).astype('int').todok()
        self.data_array = np.concatenate(
            (self.data_array, np.array((nz_arc_incidence,)))
        )
        self.n_dims = len(self.data_array)
        self.has_nz_incidence_mat = True
        self.data_fields.append("nonzero_arc_incidence")

    @classmethod
    def from_directory(cls, path, add_angles=True, angle_type='correct', delim=None,
                       match_tt_shape=False):
        """Creates data set from specified folder, assuming standard file path names.
        Also returns obs mat to keep IO tidy and together"""
        if add_angles and angle_type not in ['correct', 'comparison']:
            raise KeyError("Angle type should be 'correct' or 'comparison'")

        obs_mat, (network_attribs) = load_standard_path_format_csv(path,
                                                                   delim, match_tt_shape,
                                                                   angles_included=add_angles)
        if add_angles:
            incidence_mat, travel_times_mat, turn_angle_mat = network_attribs
            out = RecursiveLogitDataStructDeprecated(travel_times_mat, incidence_mat,
                                                     turn_angle_mat)
            if angle_type == 'correct':
                out.add_turn_categorical_variables()
            else:
                # BROKEN = True
                BROKEN = False
                if BROKEN:
                    out.add_left_turn_incidence_uturn_for_comparison()
                else:
                    out.add_turn_categorical_variables()
                    out.add_nonzero_arc_incidence()  # swap f
        else:
            incidence_mat, travel_times_mat = network_attribs
            out = RecursiveLogitDataStructDeprecated(travel_times_mat, incidence_mat,
                                                     turn_angle_mat=None)
        return out, obs_mat


class RecursiveLogitModel(abc.ABC):
    """Abstraction of the linear algebra type relations on the recursive logit model to solve
    the matrix system and compute log likelihood.

    Doesn't handle optimisation directly (but does compute log likelihood), should be
    passed into optimisation algorithm in a clever way

    """

    def __init__(self, data_struct: ModelDataStruct,
                 initial_beta=-1.5, mu=1, ):
        self.network_data = data_struct  # all network attributes
        self.data_array = data_struct.data_array
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
        else:
            beta_vec = initial_beta
        if np.any(beta_vec > 0):
            raise ValueError("Beta vector contains positive terms. Beta must be negative "
                             "so that the short term utilities are negative.")

        # setup optimiser initialisation
        self._beta_vec = beta_vec
        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        #
        # self.get_log_likelihood()  # need to compute starting LL for optimiser

    def get_beta_vec(self):
        """Getter is purely to imply that beta vec is not a fixed field"""
        return self._beta_vec

    def _compute_short_term_utility(self, skip_check=False):  # TODO review
        self.short_term_utility = np.sum(self.get_beta_vec() * self.data_array, axis=0)
        # note axis=0 means that ndarrays will give a matrix result back, not a scalar
        # which is what we want
        if skip_check is False:
            if sparse.issparse(self.short_term_utility):
                nz_rows, nz_cols = self.short_term_utility.nonzero()  # this is dense so cast ok
                cond = self.short_term_utility[nz_rows, nz_cols].toarray()
            else:
                cond = self.short_term_utility
            if np.any(cond > 0):
                warnings.warn("Short term utility contains positive terms, which is illegal. "
                              "Network attributes must be non-negative and beta must be "
                              "negative.")
                # raise ValueError("Short term utility contains positive terms, which is illegal. "
                #                  "Network attributes must be non-negative and beta must be "
                #                  "negative.")

    def get_short_term_utility(self):
        """Returns v(a|k)  for all (a,k) as 2D array,
        uses current value of beta
        :rtype: np.array<scipy.sparse.csr_matrix>"""

        return self.short_term_utility

    def _compute_exponential_utility_matrix(self):
        """ Compute M_{ka} matrix which is not orig dependent, with empty bottom, right
        row/ col for dest.
        """

        # explicitly do need this copy since we modify m_mat
        m_mat = self.get_short_term_utility().copy()
        # print(type(m_mat))
        # note we currently use incidence matrix here, since this distinguishes the
        # genuine zero arcs from the absent arcs
        # (since data format has zero arcs, should not be a problem with good data)
        nonzero_entries = self.network_data.incidence_matrix.nonzero()
        # note dense cast is fine here, since is the dense cast of a dense selection from sparse
        # (note that sparse matrices don't have exp method defined on them because exp(0)=1
        m_mat[nonzero_entries] = np.exp(
            1 / self.mu * _to_dense_if_sparse(m_mat[nonzero_entries]))
        self._exponential_utility_matrix = m_mat

    def get_exponential_utility_matrix(self):
        """ #
        Returns M_{ka} matrix
        """

        return self._exponential_utility_matrix

    @staticmethod
    def _compute_exp_value_function(m_tilde, return_pieces=False):
        """ Actual linear system solve without any error checking"""
        ncols = m_tilde.shape[1]
        rhs = scipy.sparse.lil_matrix((ncols, 1))  # suppressing needless sparsity warning
        rhs[-1, 0] = 1
        # (I-M)z =b
        a_mat = sparse.identity(ncols) - m_tilde
        z_vec = splinalg.spsolve(a_mat, rhs)  # rhs has to be (n,1)
        z_vec = np.atleast_2d(z_vec).T  # Transpose to have appropriate dims
        if return_pieces:
            return a_mat, z_vec, rhs
        else:
            return z_vec

    def compute_value_function(self, m_tilde):
        """Solves the system Z = Mz+b and stores the output for future use.
        Has rudimentary flagging of errors but doesn't attempt to solve any problems"""
        # print("mmat_int")
        # print_sparse(m_tilde)
        error_flag = False  # start with no errors
        a_mat, z_vec, rhs = self._compute_exp_value_function(m_tilde, return_pieces=True)
        # if we z values negative, or near zero, parameters
        # are infeasible since log will be complex or -infty
        # print("z_pre", z_vec)
        if z_vec.min() <= min(-1e-10, OptimiserBase.NUMERICAL_ERROR_THRESH):
            # thresh on this?
            error_flag = True  # TODO abs and stuff once i fix tests
            return error_flag
            # handling via flag rather than exceptions for efficiency
            # raise ValueError("value function has too small entries")
        # z_vec = abs()

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

        elif linalg.norm(
                np.array(a_mat @ z_vec - rhs)) > OptimiserBase.RESIDUAL:
            self.flag_exp_val_funcs_error = True
            print(f"W: Value function solution is not exact, has residual. (beta={self._beta_vec})")
            # TODO convert from soft warning to legitimate warning. Soft since it happens
            #  relatively frequently
            error_flag = True
            # raise ValueError("value function solution does not satisfy system well.")
        else:  # No errors or non terminal errors
            z_vec = np.squeeze(z_vec.T)  # not required but convenient at this point !TODO
            zeroes = z_vec[z_vec == 0]
            # Not considered error, even though degenerate case.
            if len(zeroes) > 0:
                print("W: Z contains zeros in it, so value functions are undefined for some nodes")
                val_funcs_tmp = z_vec.copy()
                val_funcs_tmp[val_funcs_tmp == 0] = np.nan
                # with np.errstate(invalid='ignore'):
                # At the moment I would prefer this crashes
                self._value_functions = np.log(val_funcs_tmp)
                # error_flag = True # TODO not from tien mai

                # in the cases where nans occur we might not actually need to deal with the numbers
                # that are nan so we don't just end here (this is not good justification TODO)
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
    def _revert_dest_column(dest_index, m_tilde, i_tilde, local_exp_util_mat, local_incidence_mat):
        """Inverse method to apply dest col - so that we can undo changes without resetting the
        matrix. Should be such that applying apply then revert is an identity operation."""

        if ALLOW_POSITIVE_VALUE_FUNCTIONS:
            # Allow positive value functions
            m_tilde[dest_index, -1] = 0
            i_tilde[dest_index, -1] = 0
        else:
            # Ban positive value functions code:
            # legacy for old understanding which required negative value functions
            # TODO the conditional to dense casts here are to avoid a bug in scipy with slice
            #  indexing a zero sparse matrix
            if (sparse.issparse(local_exp_util_mat)
                    and local_exp_util_mat[dest_index, :].count_nonzero() == 0):
                m_tilde[dest_index, :] = 0
            else:  # if this only has zeros in rhs, assignment does nothing
                m_tilde[dest_index, :] = local_exp_util_mat[dest_index, :]
            if (sparse.issparse(local_incidence_mat)
                    and local_incidence_mat[dest_index, :].count_nonzero() == 0):
                i_tilde[dest_index, :] = 0
            else:
                i_tilde[dest_index, :] = local_incidence_mat[dest_index, :]
            # TODO remove this - being extra paranoid since this is a bug with scipy, not my code
            assert np.all(
                _to_dense_if_sparse(m_tilde[dest_index, :])
                == _to_dense_if_sparse(local_exp_util_mat[dest_index, :]))

        return m_tilde, i_tilde


class RecursiveLogitModelEstimation(RecursiveLogitModel):
    """Abstraction of the linear algebra type relations on the recursive logit model to solve
    the matrix system and compute log likelihood.

    This extension has a handle to an optimiser class to enable dynamic updating of beta in a
    nice way. Ideally, this dependency would work neutrally and this wouldn't
     be a subclass. Makes sense for the Model not to be an arg to optimiser because optimiser should
     be able to take in any function and optimise.

     # TODO this should subclass prediction so that we can optimise into prediction



    """

    def __init__(self, data_struct: ModelDataStruct,
                 optimiser: OptimiserBase, observations_record,
                 initial_beta=-1.5, mu=1):

        super().__init__(data_struct, initial_beta, mu)
        self.optimiser = optimiser  # optimisation alg wrapper class
        # TODO this probably shouldn't know about the optimiser
        #    on the basis that if it doesn't then we can use the same base class here fro
        #    estimation. It might be easier for now to make a base class and have a
        #    subclass which gets an optimiser
        self.obs_record = observations_record  # matrix of observed trips

        beta_vec = super().get_beta_vec()  # orig without optim tie in
        # setup optimiser initialisation
        self.optim_function_state = OptimFunctionState(None, None, np.identity(data_struct.n_dims),
                                                       self.optimiser.hessian_type,
                                                       self.eval_log_like_at_new_beta,
                                                       beta_vec,
                                                       self._get_n_func_evals)
        self.update_beta_vec(beta_vec)  # to this to refresh dependent matrix quantitites

        # book-keeping on observations record
        if sparse.issparse(observations_record):  # TODO redact this compatibility
            self.obs_count, _ = observations_record.shape
            self.obs_min_legal_index = 1  # Zero is reserved index for sparsity

        else:  # TODO list of lists support
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
            else:  # just get the minimum index used, which should be zero or 1
                self.obs_min_legal_index = ak.min(observations_record, axis=None)
                #  TODO no unit test written for the zero case

        # finish initialising
        self.get_log_likelihood()  # need to compute starting LL for optimiser
        if isinstance(optimiser, CustomOptimiserBase):
            optimiser.set_beta_vec(beta_vec)
            optimiser.set_current_value(self.log_like_stored)

        self._path_start_nodes = None
        self._path_finish_nodes = None

    def _get_n_func_evals(self):
        return self.n_log_like_calls_non_redundant

    def solve_for_optimal_beta(self, verbose=True, extra_verbose=False, output_file=None):
        """Runs the line search optimisation algorithm until a termination condition is reached.
        Print output"""
        # print out iteration 0 information

        # if a scipy method then it is self contained and we don't need to query inbetween loops
        if isinstance(self.optimiser, ScipyOptimiser):
            optim_res = self.optimiser.solve(self.optim_function_state, verbose=verbose,
                                             output_file=output_file
                                             )
            print(optim_res)
            if optim_res.success is False:
                raise ValueError("Scipy alg error flag was raised. Process failed.")
            return optim_res.x
        # otherwise we have Optimiser is a Custom type
        print(self.optimiser.get_iteration_log(self.optim_function_state), file=None)
        n = 0

        while n <= 1000:
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
        print("Infinite loop happened somehow, shouldn't have happened")

    def update_beta_vec(self, new_beta_vec):
        """Change the current parameter vector beta and update intermediate results which depend
        on this"""
        # Prevent redundant computation
        if np.all(new_beta_vec == self._beta_vec):
            return
        self.flag_log_like_stored = False
        self.optim_function_state.beta_vec = new_beta_vec
        self._beta_vec = new_beta_vec

        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()

    def get_log_likelihood(self):
        """Compute the log likelihood of the data with the current beta vec
                n_obs override is for debug purposes to artificially lower the
                 number of observations"""
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
        for n in range(self.obs_count):
            # TODO we should be sorting by dest index to avoid recomputation
            #   if dests are the same we don't need to recompute value functions
            # Should also sort on orig index, because then we could reuse this too, but
            # benefit might not be worth the sort cost
            # a[ak.argsort(a[:,0]) or a[np.argsort(a[:,0])
            # Subtract 1 since we expect dests are 1,2,..., without zero, subtract to align into
            # zero arrays. This is in part left over from sparse array obs format,
            # which should probably be deprecated in favour of more efficient formats only.
            # print("observation")
            # print(obs_record[n, :])
            dest_index = obs_record[n, 0] - self.obs_min_legal_index
            orig_index = obs_record[n, 1] - self.obs_min_legal_index

            m_tilde, i_tilde = self._apply_dest_column(dest_index, m_tilde, i_tilde)

            # Now get exponentiated value funcs
            error_flag = self.compute_value_function(m_tilde)
            # If we had numerical issues in computing value functions
            if error_flag:  # terminate early with error vals
                self.log_like_stored = OptimiserBase.LL_ERROR_VALUE
                self.grad_stored = np.ones(n_dims)  # TODO better error gradient?
                self.flag_log_like_stored = True
                print("Parameters are infeasible.")
                return self.log_like_stored, self.grad_stored

            value_funcs, exp_val_funcs = self._value_functions, self._exp_value_functions
            # Gradient and log like depend on origin values

            # respective sub function? probably yes
            grad_orig = self.get_value_func_grad_orig(orig_index, m_tilde, exp_val_funcs)

            self._compute_obs_path_indices(obs_record[n, :])  # required to compute LL & grad
            # LogLikeGrad in Code doc
            mu_gradient_current_obs = self._compute_current_obs_mu_ll_grad(grad_orig)
            # LogLikeFn in Code Doc - for this n
            mu_log_like_obs = self._compute_current_obs_mu_log_like(v_mat, value_funcs[orig_index])

            # # Some kind of successive over relaxation/ momentum
            mu_log_like_cumulative += (mu_log_like_obs - mu_log_like_cumulative) / (n + 1)
            # mu_log_like_cumulative += mu_log_like_obs
            # ll_grad_cumulative += mu_gradient_current_obs
            mu_ll_grad_cumulative += (mu_gradient_current_obs - mu_ll_grad_cumulative) / (n + 1)
            # print("current grad weighted", ll_grad_cumulative)

            # Put our matrices back untouched:
            m_tilde, i_tilde = self._revert_dest_column(dest_index, m_tilde, i_tilde,
                                                        m_mat, local_incidence_mat)
            # m_tilde[dest_index, :] = m_mat[dest_index, :]
            # i_tilde[dest_index, :] = local_incidence_mat[dest_index, :]

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
        self.update_beta_vec(beta_vec)
        return self.get_log_likelihood()

    def _compute_obs_path_indices(self, obs_row: Union[scipy.sparse.dok_matrix,
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

        if sparse.issparse(obs_row):
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

        # if np.any(path != self._prev_path): # infer I was planning caching, unused though

        min_legal_index = self.obs_min_legal_index
        # in a sequence of obs [o, 1, 2, 3, 4, 5, 6, 7, 8, 9, d]
        # we have the transitions
        #    [(o, 1), (1, 2), (2, 3), (3, 4), ...] or
        # origins [o, 1, 2, 3, 4, ..., 9] and dests [1, 2, 3, ..., 9, d]
        # all nodes i in (i ->j) transitions
        # subtract min_legal to get 0 based indexes if we were read sparse matrices
        path_arc_start_nodes = path[:-1] - min_legal_index
        path_arc_finish_nodes = path[1:] - min_legal_index

        # Note -1 is because data struct adds a zero column to put the dest into
        # TODO unless user pre-empts this and it doesn't
        final_index_in_data = np.shape(self.network_data.incidence_matrix)[0]-1

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
        # print("incomplete grad func recieved for expV\n", exp_val_funcs)
        grad_v = np.zeros((self.n_dims, np.shape(m_tilde)[0]))
        identity = sparse.identity(np.shape(m_tilde)[0])
        z = exp_val_funcs  # consistency with maths doc

        # low number of dims -> not vectorised for convenience
        # (actually easy to fix now)
        for q in range(self.n_dims):
            chi = self.data_array[q]  # current attribute of data
            # Have experienced numerical instability with spsolve, but believe this is
            # due to matrices with terrible condition numbers in the examples
            # spsolve(A,b) == inv(A)*b
            # Note: A.multiply(B) is A .* B for sparse matrices
            grad_v[q, :] = splinalg.spsolve(identity - m_tilde, m_tilde.multiply(chi) @ z)
            # print(np.linalg.norm((I- m_tilde)*grad_v[q, :] - m_tilde.multiply(chi) * z))

        return grad_v


class RecursiveLogitModelPrediction(RecursiveLogitModel):
    """Subclass which generates simulated observations based upon the supplied beta vector.
    Uses same structure as estimator in the hopes I can unify these such that one can use the
    estimator for prediction as well (probably have estimator inherit from this)"""

    def generate_observations(self, origin_indices, dest_indices, num_obs_per_pair, iter_cap=1000,
                              rng_seed=None,
                              ):
        """

        :param origin_indices: iterable of indices to start paths from
        :type origin_indices: list or iterable
        :param dest_indices: iterable of indices to end paths at
        :type dest_indices: List or iterable
        :param num_obs_per_pair: Number of observations to generate for each OD pair
        :type num_obs_per_pair: int
        :param iter_cap: iteration cap in the case of non convergent value functions.
                        Hopefully should not occur but may happen if the value function solutions
                        are negative but ill conditioned.
        :type iter_cap: int
        :param rng_seed: Seed for numpy generator, or instance of np.random.BitGenerator
        :type rng_seed: int or np.random.BitGenerator  (any legal input to np.random.default_rng())

        :rtype list<list<int>>
        :return List of list of all observations generated
        """
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

        # print("shapes", m_tilde.shape, i_tilde.shape, local_short_term_util.shape,
        #       local_exp_util_mat.shape)
        # print(m_tilde.toarray())

        for dest in dest_indices:
            self._apply_dest_column(dest, m_tilde, i_tilde)

            z_vec = self._compute_exp_value_function(m_tilde)
            # print(z_vec)
            with np.errstate(divide='ignore', invalid='ignore'):
                value_funcs = np.log(z_vec)
            # catch errors manually
            if np.any(~np.isfinite(value_funcs)):  # any infinite (nan/ -inf)
                raise ValueError("Parameter Beta is incorrectly determined, value function has "
                                 "no solution in this case.")

            elif ALLOW_POSITIVE_VALUE_FUNCTIONS is False and np.any(value_funcs > 0):
                warnings.warn("WARNING: Positive value functions:"
                              f"{value_funcs[value_funcs > 0]}")

            # loop through path starts, with same base value functions
            for orig in origin_indices:
                if orig == dest:  # redundant case
                    # print("skipping o==d for o=", orig)
                    continue

                # repeat until we have specified number of obs
                for i in range(num_obs_per_pair):
                    # while we haven't reached the dest

                    current_arc = orig
                    current_path = [orig]
                    # path_string = f"Start: {orig}"
                    count = 0
                    while current_arc != dest_dummy_arc_index:  # index of augmented dest arc
                        count += 1
                        if count > iter_cap:
                            # print("orig, dest =", orig, dest, value_funcs[0])
                            # print("path failed, in progress is:", path_string)

                            raise ValueError(f"Max iterations reached. No path from {orig} to "
                                             f"{dest} was found within cap.")
                        current_incidence_col = i_tilde[current_arc, :]
                        # all arcs current arc connects to (index converts from 2d to 1d)
                        if sparse.issparse(current_incidence_col):
                            # we get a matrix return type for 1 row * n cols
                            # from .nonzero() - [0]th component is the row (always zero)
                            # and [1]th component is col,
                            neighbour_arcs = current_incidence_col.nonzero()[1]
                        else:
                            # nd-arrays are smart enough to work out they are 1d
                            neighbour_arcs = current_incidence_col.nonzero()[0]
                        # print(neighbour_arcs, "curr inc", current_incidence_col)
                        # print('\t', current_incidence_col.nonzero())

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
                        # print(np.max(value_functions_observed), next_arc)
                        # path_string += f" -> {next_arc}"
                        current_arc = next_arc
                        # TODO Note the int conversion here is purely for json serialisation compat
                        #   if this is no longer used then we can keep numpy types
                        current_path.append(int(current_arc))
                        # offset stops arc number being recorded as zero

                    # print(path_string + ": Fin")
                    if len(current_path) <= 2:
                        continue  # this is worthless information saying we got from O to D in one
                        # step
                    output_path_list.append(current_path)

            # Fix the columns we changed for this dest (cheaper than refreshing whole matrix)
            self._revert_dest_column(dest, m_tilde, i_tilde,
                                     local_exp_util_mat, local_incidence_mat)

        return output_path_list
