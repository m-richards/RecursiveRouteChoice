import os
import warnings
from typing import List
import scipy
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse import linalg as splinalg
from scipy import sparse

import awkward1 as ak

from data_loading import load_csv_to_sparse, resize_to_dims, load_standard_path_format_csv
from data_processing import AngleProcessor
from debug_helpers import print_sparse, print_data_struct
from optimisers.extra_optim import OptimFunctionState
from optimisers.optimisers_file import Optimiser, OptimType
import numpy as np

"""
Sketch of what we need

Class to store data
    travel time
    turn stuff
    (link size)

    keep track of optimisation progress - number of function evaluations

    - function to compute likelihood estimates
    - keep track of hyper parameters
     print some feedback on each operation - PrintOut function
        log likelihood
        values of beta
        norm of step
        radius
        norm & relative norm of grad
        number of evaluations

    log results on each iteration

    - optimisation algorithm LINE SEARCH for now?
    - stopping condition

    -


"""


def _to_dense_if_sparse(mat):
    if sparse.issparse(mat):
        return mat.todense()
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
        return mat.todok() # don't want to stay as coo # TODO do explicit casts where we need them
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



# TODO think perhaps no specific labels should be here, that these should be done by some
#  other preprocessing step before this class and
#  this should be a generic data class.
#  On review, easiest to have this non generic for now, but could easily
#  move init to classmethods. Wait for real data to see.
#  Should at least have methods to allow generic initialisation

class RecursiveLogitDataStruct(object):
    """Generic struct which stores all the arc attributes together in a convenient manner.
    Also provides convenience constructors.
    generic
    """

    def __init__(self, data_matrix_list: List[scipy.sparse.dok_matrix],
                 incidence_matrix: scipy.sparse.dok_matrix, data_array_names_debug=None):

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
            out = RecursiveLogitDataStructDeprecated(travel_times_mat, incidence_mat, turn_angle_mat)
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
            out = RecursiveLogitDataStructDeprecated(travel_times_mat, incidence_mat, turn_angle_mat=None)
        return out, obs_mat


class RecursiveLogitModel(object):
    """Abstraction of the linear algebra type relations on the recursive logit model to solve
    the matrix system and compute log likelihood.

    Doesn't handle optimisation directly (but does compute log likelihood), should be
    passed into optimisation algorithm in a clever way

    """

    def __init__(self, data_struct: RecursiveLogitDataStruct,
                 initial_beta=-1.5, mu=1, ):
        self.network_data = data_struct  # all network attributes
        # TODO this probably shouldn't know about the optimiser
        #    on the basis that if it doesn't then we can use the same base class here fro
        #    estimation. It might be easier for now to make a base class and have a
        #    subclass which gets an optimiser
        # self.user_obs_mat = user_obs_mat  # matrix of observed trips
        self.data_array = data_struct.data_array
        self.n_dims = len(self.data_array)
        self.mu = mu

        self._short_term_utility = None
        self._exponential_utility_matrix = None
        self._value_functions = None
        self._exp_value_functions = None

        self.flag_log_like_stored = False
        self.flag_exp_val_funcs_error = True
        self._prev_path = None

        self.n_log_like_calls = 0
        self.n_log_like_calls_non_redundant = 0

        # allow for specification of vector beta or scalar beta repeated
        if isinstance(initial_beta, (float, int)):
            beta_vec = np.array([initial_beta for _ in range(self.n_dims)])
        else:
            beta_vec = initial_beta
        # setup optimiser initialisation
        self._beta_vec = beta_vec
        # TODO these should be in init, but aren't because of bad coupling in estimation subclass
        # self._compute_short_term_utility()
        # self._compute_exponential_utility_matrix()
        #
        # self.get_log_likelihood()  # need to compute starting LL for optimiser


        self._path_start_nodes = None
        self._path_finish_nodes = None

    def get_beta_vec(self):
        """Getter is purely to imply that beta vec is not a fixed field"""
        return self._beta_vec

    def update_beta_vec(self, new_beta_vec):
        """Change the current parameter vector beta and update intermediate results which depend
        on this"""
        # If beta has changed we need to refresh values
        if (new_beta_vec != self._beta_vec).any():
            self.flag_log_like_stored = False

        # TODO delay this from happening until the update is needed - use flag

        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        # self._compute_value_function_matrix()
        # TODO make sure new stuff gets added here

    def _compute_short_term_utility(self):
        print("data dim\n")
        # print_data_struct(self.network_data)

        print("beta", self.get_beta_vec())
        self.short_term_utility = np.sum(self.get_beta_vec() * self.data_array, axis=0)
        # note axis=0 means that ndarrays will give a matrix result back, not a scalar
        # which is what we want
        print(self.data_array)
        print("vmat",)
        print(self.get_beta_vec().shape, self.data_array.shape)
        print(self.short_term_utility)
        print_sparse(self.short_term_utility)
        # print(type(self.short_term_utility))

    def get_short_term_utility(self):
        """Returns v(a|k)  for all (a,k) as 2D array,
        uses current value of beta
        :rtype: np.array<scipy.sparse.csr_matrix>"""

        return self.short_term_utility

    def _compute_exponential_utility_matrix(self):
        """ # TODO can cached this if I deem it handy.
            Returns M_{ka} matrix which is not orig dependent
        """

        # explicitly do need this copy since we modify m_mat
        m_mat = self.get_short_term_utility().copy()
        # note we currently use incidence matrix here, since this distinguishes the
        # genuine zero arcs from the absent arcs
        # (since data format has zero arcs for silly reasons)
        nonzero_entries = self.network_data.incidence_matrix.nonzero()

        m_mat[nonzero_entries] = np.exp(
            1 / self.mu * _to_dense_if_sparse(m_mat[nonzero_entries]))
        self._exponential_utility_matrix = m_mat

    def get_exponential_utility_matrix(self):
        """ # TODO can cached this if I deem it handy.
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
        a_mat = identity(ncols) - m_tilde
        z_vec = splinalg.spsolve(a_mat, rhs)  # rhs has to be (n,1) not (1,n)
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
        # if we have non near zero negative, we have a problem and parameters are infeasible
        # since log will be complex
        # print("z_pre", z_vec)
        if z_vec.min() <= min(-1e-10, Optimiser.NUMERICAL_ERROR_THRESH):
            # thresh on this?
            error_flag = True # TODO abs and stuff once i fix tests
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
        elif linalg.norm(
                np.array(a_mat @ z_vec - rhs)) > Optimiser.RESIDUAL:
            self.flag_exp_val_funcs_error = True
            print("W: Value function solution is not exact, has residual.")
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
                    # in the cases where nans occur we might not actually need to deal with the numbers
                    # that are nan so we don't just end here (this is not good justification TODO)
            else:
                self._value_functions = np.log(z_vec)
            self._exp_value_functions = z_vec  # TODO should this be saved onto OptimStruct?
        return error_flag

    def _compute_obs_path_indices(self, obs_row: scipy.sparse.dok_matrix):
        """Takes in the current iterate row of the observation matrix.
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
            path_len = len(obs_row)
            path = obs_row[1:]
            path = ak.to_numpy(path)

        else:
            raise ValueError("obs record has unsupported type")

        # if np.any(path != self._prev_path): # infer I was planning caching, unused though
            # subtract 1 to get 0 based indexes
        path_arc_start_nodes = path[:-1] - 1  # all nodes i in (i ->j) transitions
        path_arc_finish_nodes = path[1:] - 1  # all nodes j

        final_index_in_data = np.shape(self.network_data.incidence_matrix)[0]

        if np.any(path_arc_finish_nodes > final_index_in_data):
            path_arc_finish_nodes = np.minimum(path_arc_finish_nodes, final_index_in_data)
            warnings.warn("WARN, dodgy bounds indexing hack occur in path tracing,"
                  " changed a node to not exceed maximum", category='error')

        self._path_start_nodes = path_arc_start_nodes
        self._path_finish_nodes = path_arc_finish_nodes
        return self._path_start_nodes, self._path_finish_nodes

    def _compute_obs_log_like(self, v_mat, orig_utility, mu):
        """# LogLikeFn in Code Doc
        Compute the log likelihood function for the currently observed path.
        "Current" in this sense stipulates that compute_obs_path_indices has been called prior
        in order to update these indices."""
        arc_start_nodes = self._path_start_nodes
        arc_finish_nodes = self._path_finish_nodes

        sum_inst_util = v_mat[arc_start_nodes, arc_finish_nodes].sum()
        log_like_orig = -1 * (1 / mu) * orig_utility  # log probability from origin
        log_like_obs = 1 / mu * sum_inst_util + log_like_orig  # LogLikeFn in Code Doc
        return log_like_obs

    def _compute_obs_log_like_grad(self, grad_orig):
        """# LogLikeGrad in Code doc
        Compute the log likelihood function gradient for the currently observed path.
        "Current" in this sense stipulates that compute_obs_path_indices has been called prior
        in order to update these indices."""
        # subtract 1 to get 0 based indexes
        arc_start_nodes = self._path_start_nodes
        arc_finish_nodes = self._path_finish_nodes

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
        """
        Function to compute GradValFn2 from algorithm chapter
        \pderiv{\bm{V}}{\beta_q}
                & =\frac{1}{\bm{z}}\circ
                (I - \bm{M})^{-1}\paren{\bm{M}\circ \chi^q}  \bm{z}

                without the leading 1/z term as a matrix for all q.
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
        I = identity(np.shape(m_tilde)[0])
        z = exp_val_funcs  # consistency with maths doc

        # low number of dims -> not vectorised for convenience
        # (actually easy to fix now)
        for q in range(self.n_dims):
            chi = self.data_array[q]  # current attribute of data
            # Have experienced numerical instability with spsolve, but believe this is
            # due to matrices with terrible condition numbers in the examples
            # spsolve(A,b) == inv(A)*b
            grad_v[q, :] = splinalg.spsolve(I - m_tilde, m_tilde.multiply(chi) * z)
            # print(np.linalg.norm((I- m_tilde)*grad_v[q, :] - m_tilde.multiply(chi) * z))

        return grad_v


class RecursiveLogitModelEstimation(RecursiveLogitModel):
    """Abstraction of the linear algebra type relations on the recursive logit model to solve
    the matrix system and compute log likelihood.

    This extension has a handle to an optimiser class to enable dynamic updating of beta in a
    nice way. Ideally, this dependency would work neutrally and this wouldn't
     be a subclass. Makes sense for the Model not to be an arg to optimiser because optimiser should
     be able to take in any function and optimise.



    """

    def __init__(self, data_struct: RecursiveLogitDataStruct, optimiser: Optimiser, observations_record,
                 initial_beta=-1.5, mu=1):

        super().__init__(data_struct, initial_beta, mu)
        self.optimiser = optimiser  # optimisation alg wrapper class
        # TODO this probably shouldn't know about the optimiser
        #    on the basis that if it doesn't then we can use the same base class here fro
        #    estimation. It might be easier for now to make a base class and have a
        #    subclass which gets an optimiser
        self.obs_record = observations_record  # matrix of observed trips

        beta_vec = super().get_beta_vec() # orig without optim tie in
        # setup optimiser initialisation
        self.optim_function_state = OptimFunctionState(None, None, np.identity(data_struct.n_dims),
                                        self.optimiser.hessian_type,
                                        self.eval_log_like_at_new_beta,
                                        beta_vec,
                                        self._get_n_func_evals)
        self.update_beta_vec(beta_vec)  # to this to refresh dependent matrix quantitites
        self.get_log_likelihood()  # need to compute starting LL for optimiser
        optimiser.set_beta_vec(beta_vec)
        optimiser.set_current_value(self.log_like_stored)

        self._path_start_nodes = None
        self._path_finish_nodes = None

    def _get_n_func_evals(self):
        return self.n_log_like_calls_non_redundant

    def solve_for_optimal_beta(self, output_file=None):
        """Runs the line search optimisation algorithm until a termination condition is reached.
        Print output"""
        # print out iteration 0 information
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
                return
        print("Infinite loop happened somehow, shouldn't have happened")

    def get_beta_vec(self):
        """Getter is purely to imply that beta vec is not a fixed field"""
        return self.optim_function_state.beta_vec

    def update_beta_vec(self, new_beta_vec):
        """Change the current parameter vector beta and update intermediate results which depend
        on this"""
        # If beta has changed we need to refresh values
        if (new_beta_vec != self.optim_function_state.beta_vec).any():
            self.flag_log_like_stored = False
        self.optim_function_state.beta_vec = new_beta_vec

        # self._beta_changed = True
        # TODO delay this from happening until the update is needed - use flag

        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        # self._compute_value_function_matrix()
        # TODO make sure new stuff gets added here

    def get_log_likelihood(self, n_obs_override=None):
        """Compute the log likelihood of the data with the current beta vec
                n_obs override is for debug purposes to artificially lower the number of observations"""
        self.n_log_like_calls += 1
        # TODO reinstate caching, currently have problems because beta can update externally
        # if self.flag_log_like_stored:
        #     return self.log_like_stored, self.grad_stored
        self.n_log_like_calls_non_redundant += 1

        obs_record = self.obs_record
        if sparse.issparse(obs_record): # TODO redact this compatibility
            num_obs, _ = obs_record.shape
        else:
            # num_obs = len(obs_record) # equivalent but clearer this is ak array
            num_obs = ak.num(obs_record, axis=0)
        if n_obs_override is not None:
            num_obs = n_obs_override
        # local references with idomatic names
        n_dims = self.n_dims  # number of attributes in data
        mu = self.mu
        v_mat = self.get_short_term_utility()  # capital u in tien mai's code
        m_mat = self.get_exponential_utility_matrix()

        log_like_cumulative = 0.0  # weighting of all observations
        grad_cumulative = np.zeros(n_dims)  # gradient combined across all observations

        # iterate through observation number
        for n in range(num_obs):
            dest_index = obs_record[n, 0] - 1  # subtract 1 for zero based python
            orig_index = obs_record[n, 1] - 1

            # TODO review this, still not sure it makes mathematical sense
            # Compute modified matrices for current dest
            old_row_shape = m_mat.shape[0]
            last_index_in_rows = old_row_shape - 1

            m_tilde = m_mat[0:last_index_in_rows + 1,
                      0:last_index_in_rows + 1]  # plus 1 for inclusive
            m_tilde[:, last_index_in_rows, ] = m_mat[:, dest_index]

            # Now get exponentiated value funcs
            error_flag = self.compute_value_function(m_tilde)
            # If we had numerical issues in computing value functions
            if error_flag:  # terminate early with error vals
                self.log_like_stored = Optimiser.LL_ERROR_VALUE
                self.grad_stored = np.ones(n_dims)
                self.flag_log_like_stored = True
                print("Parameters are infeasible.")
                return self.log_like_stored, self.grad_stored

            value_funcs, exp_val_funcs = self._value_functions, self._exp_value_functions
            # Gradient and log like depend on origin values
            orig_utility = value_funcs[orig_index]  # TODO should these be nested inside
            # respective sub function? probably yes
            grad_orig = self.get_value_func_grad_orig(orig_index, m_tilde, exp_val_funcs)

            self._compute_obs_path_indices(obs_record[n, :])  # required to compute LL & grad
            # LogLikeGrad in Code doc
            gradient_current_obs = self._compute_obs_log_like_grad(grad_orig)
            # LogLikeFn in Code Doc - for this n
            log_like_obs = self._compute_obs_log_like(v_mat, orig_utility, mu)

            # # Some kind of successive over relaxation/ momentum
            log_like_cumulative += (log_like_obs - log_like_cumulative) / (n + 1)
            # log_like_cumulative += log_like_obs
            # grad_cumulative += gradient_current_obs
            grad_cumulative += (gradient_current_obs - grad_cumulative) / (n + 1)
            # print("current grad weighted", grad_cumulative)

        self.log_like_stored = -log_like_cumulative
        self.grad_stored = -grad_cumulative

        self.optim_function_state.value = self.log_like_stored
        self.optim_function_state.grad = self.grad_stored

        return self.log_like_stored, self.grad_stored

    def eval_log_like_at_new_beta(self, beta_vec):
        """update beta vec and compute log likelihood in one step - used for lambdas
        Effectively a bad functools.partial"""
        self.update_beta_vec(beta_vec)
        return self.get_log_likelihood()


class RecursiveLogitModelPrediction(RecursiveLogitModel):
    """Subclass which generates simulated observations based upon the supplied beta vector.
    Uses same structure as estimator in the hopes I can unify these such that one can use the
    estimator for prediction as well (probably have estimator inherit from this)"""
    def __init__(self, data_struct: RecursiveLogitDataStruct, initial_beta=-1.5,
                 mu=1):
        super().__init__(data_struct, initial_beta, mu)
        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()

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
        # this is never updated beyond initial padding (just need the zero col for dest util
        local_short_term_util = _zero_pad_mat(local_short_term_util, right=True)


        local_exp_util_mat =self.get_exponential_utility_matrix()
        local_incidence_mat =self.network_data.incidence_matrix
        m_tilde = local_exp_util_mat.copy()
        i_tilde = local_incidence_mat.copy()

        first_iter = True
        m, n = m_tilde.shape
        assert m == n  # paranoia

        dest_dummy_arc_index = m  # zero based equivalent of m+1 cols

        m_tilde = _zero_pad_mat(m_tilde, bottom=True, right=True)
        incidence_tilde = _zero_pad_mat(i_tilde, bottom=True, right=True)

        for dest in dest_indices:
            # print("dest = ", dest, "augmented dest col =", m)

            # reset to default vals
            if not first_iter:
                m_tilde[:-1, :-1] = local_exp_util_mat
                m_tilde[-1, :] = 0
                m_tilde[:, -1] = 0

                incidence_tilde[:-1, :-1] = local_incidence_mat
                incidence_tilde[-1, :] = 0
                incidence_tilde[:, -1] = 0

                first_iter = False

            # Destination enforcing # TODO review these assumptions and nonzero dest util
            #                               note that dest util updates would need to update
            #                               short term util locally as well (just final col).
            m_tilde[dest, :] = 0.0
            m_tilde[dest, -1] = 1  # exp(v(a|k)) = 1 when v(a|k) = 0 # try 0.2
            incidence_tilde[dest, :] = 0
            incidence_tilde[dest, -1] = 1

            z_vec = self._compute_exp_value_function(m_tilde)
            value_funcs = np.log(z_vec)
            if np.any(value_funcs > 0):
                warnings.warn(f"WARNING: Positive value functions: {value_funcs[value_funcs > 0]}",
                              category='error')

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
                        current_incidence_col = incidence_tilde[current_arc, :]
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

        return output_path_list





