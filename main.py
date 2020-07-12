import os

import scipy
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse import linalg as splinalg

from data_loading import load_csv_to_sparse, get_left_turn_categorical_matrix, \
    get_uturn_categorical_matrix, resize_to_dims
from debug_helpers import print_sparse
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


class RecursiveLogitDataStruct(object):
    """Generic struct which stores all the arc attributes together in a convenient manner.
    Also provides convenience constructors.
    """

    def __init__(self, travel_times: scipy.sparse.dok_matrix,
                 incidence_matrix: scipy.sparse.dok_matrix, turn_angle_mat=None):
        self.travel_times = travel_times
        self.incidence_matrix = incidence_matrix
        self.turn_angle_mat = turn_angle_mat
        # TODO include uturn and left turn penalties when relevant
        #   I think that these should be done by some other preprocessing step before this class,
        # this should be a generic data class. Is not for now
        # On review, easiest to have this non generic for now, but could easily
        # move init to classmethods. Wait for real data to see.
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
        left_turn_dummy = get_left_turn_categorical_matrix(self.turn_angle_mat, left_turn_thresh,
                                                           u_turn_thresh)
        u_turn_dummy = get_uturn_categorical_matrix(self.turn_angle_mat, u_turn_thresh)
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
    def from_directory(cls, path, add_angles=True, angle_type='correct', delim=None):
        """Creates data set from specified folder, assuming standard file path names.
        Also returns obs mat to keep IO tidy and together"""
        if add_angles and angle_type not in ['correct', 'comparison']:
            raise KeyError("Angle type should be 'correct' or 'comparison'")
        INCIDENCE = "incidence.txt"
        TRAVEL_TIME = 'travelTime.txt'
        OBSERVATIONS = "observations.txt"
        TURN_ANGLE = "turnAngle.txt"
        file_incidence = os.path.join(path, INCIDENCE)
        file_travel_time = os.path.join(path, TRAVEL_TIME)
        file_turn_angle = os.path.join(path, TURN_ANGLE)
        file_obs = os.path.join(path, OBSERVATIONS)

        travel_times_mat = load_csv_to_sparse(file_travel_time, delim=delim,
                                              ).todok()
        fixed_dims = travel_times_mat.shape
        incidence_mat = load_csv_to_sparse(
            file_incidence, dtype='int', delim=delim).todok()
        # Get observations matrix - note: observation matrix is in sparse format, but is of the form
        #   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
        obs_mat = load_csv_to_sparse(
            file_obs, dtype='int', square_matrix=False, delim=delim).todok()

        if add_angles:
            turn_angle_mat = load_csv_to_sparse(file_turn_angle, delim=delim).todok()
            resize_to_dims(turn_angle_mat, fixed_dims, "Turn Angles")
            # print("qq", turn_angle_mat.shape, fixed_dims)
            out = RecursiveLogitDataStruct(travel_times_mat, incidence_mat, turn_angle_mat)
            if angle_type == 'correct':
                out.add_turn_categorical_variables()
            else:
                out.add_turn_categorical_variables()
                out.add_nonzero_arc_incidence()
        else:
            out = RecursiveLogitDataStruct(travel_times_mat, incidence_mat, turn_angle_mat=None)
        return out, obs_mat


class RecursiveLogitModel(object):
    """Abstraction of the linear algebra type relations on the recursive logit model to solve
    the matrix system and compute log likelihood.

    Doesn't handle optimisation directly (but does compute log likelihood), should be
    passed into optimisation algorithm in a clever way

    """

    def __init__(self, data_struct: RecursiveLogitDataStruct, optimiser: Optimiser, user_obs_mat,
                 initial_beta=-1.5, mu=1, ):
        self.network_data = data_struct  # all network attributes
        self.optimiser = optimiser  # optimisation alg wrapper class
        self.user_obs_mat = user_obs_mat  # matrix of observed trips
        self.data_array = data_struct.data_array
        self.n_dims = len(self.data_array)
        # TODO maybe have to handle more complex initialisations
        self.beta_vec = np.array([initial_beta for _ in range(self.n_dims)])
        self.mu = mu

        self._short_term_utility = None
        self._exponential_utility_matrix = None
        self._value_functions = None
        self._exp_value_functions = None

        self.flag_log_like_stored = False

        self.hessian = np.identity(data_struct.n_dims)  # TODO should this be on optimser instead?
        self.flag_exp_val_funcs_error = True

        self.update_beta_vec(self.beta_vec)  # to this to refresh dependent matrix quantitites
        self.n_log_like_calls = 0
        self.n_log_like_calls_non_redundant = 0

        # setup optimiser initialisation
        self.get_log_likelihood()  # need to compute starting LL for optimiser
        optimiser.set_beta_vec(self.beta_vec)
        optimiser.set_current_value(self.log_like_stored)

    def solve_for_optimal_beta(self, output_file=None):
        """Runs the line search optimisation algorithm until a termination condition is reached.
        Print output"""
        # print out iteration 0 information
        print(self.optimiser.get_iteration_log(self), file=None)
        n = 0
        while n <= 1000:
            if self.optimiser.METHOD_FLAG == OptimType.LINE_SEARCH:
                ok_flag, hessian, log_msg = self.optimiser.iterate_step(self, verbose=False,
                                                                        output_file=None)
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
        return self.beta_vec

    def update_beta_vec(self, new_beta_vec):
        """Change the current parameter vector beta and update intermediate results which depend
        on this"""
        # If beta has changed we need to refresh values
        if (new_beta_vec != self.beta_vec).any():
            self.flag_log_like_stored = False
        self.beta_vec = new_beta_vec

        # self._beta_changed = True
        # TODO delay this from happening until the update is needed - use flag

        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        # self._compute_value_function_matrix()
        # TODO make sure new stuff gets added here

    def _compute_short_term_utility(self):
        # print("beta", self.beta_vec.shape)
        # print("data dim", self.data_array)
        self.short_term_utility = np.sum(self.beta_vec * self.data_array)
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
        m_mat[nonzero_entries] = np.exp(1 / self.mu * m_mat[nonzero_entries].todense())
        self._exponential_utility_matrix = m_mat

    def get_exponential_utility_matrix(self):
        """ # TODO can cached this if I deem it handy.
        Returns M_{ka} matrix
        """

        return self._exponential_utility_matrix

    def _compute_value_function(self, m_tilde):
        """Solves the system Z = Mz+b and stores the output for future use.
        Has rudimentary flagging of errors but doesn't attempt to solve any problems"""
        error_flag = 0
        ncols = self.network_data.incidence_matrix.shape[1]
        rhs = scipy.sparse.lil_matrix((ncols, 1))  # supressing needless sparsity warning
        rhs[-1, 0] = 1
        # (I-M)z =b


        # TODO tien takes the absolute value for numerical safety here
        A = identity(ncols) - m_tilde
        # print("printing A")
        # print(A.toarray())
        # print("rhs", rhs.toarray())
        z_vec = splinalg.spsolve(A, rhs)
        z_vec = np.atleast_2d(z_vec).T  # want a (x,1) column vector so that linear alg
        # checks work
        # print("z_vec", z_vec)
        # check we aren't getting negative solutions for value functions
        if z_vec.min() <= min(-1e-10, Optimiser.NUMERICAL_ERROR_THRESH):
            # thresh on this?
            self.flag_exp_val_funcs_error = True
            return
            # handling via flag rather than exceptions for efficiency
            # raise ValueError("value function has too small entries")

        if np.any(z_vec < 0):
            raise ValueError("value function had negative solution, cannot take "
                             "logarithm") # TODO note tien mai code just takes abs value
        # Note the transpose here is not mathematical, it is scipy being
        # lax about row and column vectors

        # Norm of residual
            # Note: Scipy sparse norm doesn't have a 2 norm so we use this
            # note this is expensive, in practice we may want to switch to sparse frobenius norm
            # element wise norm so doable sparsely
        # note that z_vec is dense so this should be dense without explicit cae
        if linalg.norm(
                np.array(A @ z_vec - rhs)) > Optimiser.RESIDUAL:  # residual - i.e. ill
            # conditioned solution
            self.flag_exp_val_funcs_error = True
            print("W: Value function solution is not exact, has residual.")
            return
            # raise ValueError("value function solution does not satisfy system well.")
        z_vec =np.squeeze(z_vec.T) # not required but convenient at this point !TODO
        zeroes = z_vec[z_vec == 0]
        if len(zeroes) > 0:
            # print("'Soft Warning', Z contains zeros in it, so value functions are undefined")
            val_funcs_tmp = z_vec.copy()
            val_funcs_tmp[val_funcs_tmp == 0] = np.nan  # TODO propagate nan handling
            # in the cases where nans occur we might not actually need to deal with the numbers
            # that are nans

        else:
            val_funcs_tmp = z_vec
        self.flag_exp_val_funcs_error = False # execution finished as normal
        self._value_functions = np.log(val_funcs_tmp)
        self._exp_value_functions = z_vec

    # def get_value_functions(self, return_exponentiated=False):
    #     if return_exponentiated:
    #         return self._value_functions, self._exp_value_functions
    #     return self._value_functions

    def get_log_like_new_beta(self, beta_vec):
        """update beta vec and compute log likelihood in one step - used for lambdas
        Effectively a bad functools.partial"""
        self.update_beta_vec(beta_vec)
        return self.get_log_likelihood()

    def get_log_likelihood(self, n_obs_override=None):
        """Compute the log likelihood of the data with the current beta vec
                n_obs override is for debug purposes to artificially lower the number of observations"""
        self.n_log_like_calls += 1
        if self.flag_log_like_stored:
            return self.log_like_stored, self.grad_stored
        self.n_log_like_calls_non_redundant += 1
        # print_sparse(v_mat)

        # TODO majorly wrong, doesn't use the obs matrix!
        #   m and v are supposed to depend on current obs
        obs_mat = self.user_obs_mat
        num_obs, path_max_len = np.shape(obs_mat)
        if n_obs_override is not None:
            num_obs = n_obs_override
        # local references with idomatic names
        N = self.n_dims  # number of attributes in data
        mu = self.mu
        v_mat = self.get_short_term_utility()  # capital u in tien mai's code
        m_mat = self.get_exponential_utility_matrix()

        # value_funcs, exp_val_funcs = self.get_value_functions(return_exponentiated=True)
        # print("printing v_mat")
        # print_sparse(v_mat)
        #
        # print("printing m_mat")
        # print_sparse(m_mat)


        # grad = get_value_func_grad(m_mat, self, exp_val_funcs)

        log_like_cumulative = 0.0  # weighting of all observations
        grad_cumulative = np.zeros(N)  # gradient combined across all observations
        gradient_each_obs = np.zeros((num_obs, N))  # store gradient according to each obs
        # tODO this looks redundant to store these at
        #  the moment but this is a global variable in TIEN's code

        # iterate through observation number
        for n in range(num_obs):
            # print("obs num: ", n, "--------------------------------------")
            dest = obs_mat[n, 0]
            orig_index = obs_mat[n, 1] - 1  # subtract 1 for zero based python
            # first_action = obs_mat[n, 2]

            # Compute modified matrices for current dest
            old_row_shape = m_mat.shape[0]
            last_index_in_rows = old_row_shape-1


            m_tilde = m_mat[0:last_index_in_rows+1, 0:last_index_in_rows+1] # plus 1 for inclusive
            # print("m_mat shape", m_mat.shape, m_tilde.shape)
            m_tilde[:, last_index_in_rows,] = m_mat[:, dest-1]
            # force an extra final row

            # m_tilde.resize(old_row_shape+1, old_row_shape+1)

            # print("m mat")
            # print(m_mat.toarray())
            #
            # print(("m_tilde"))
            # print(m_tilde.toarray())

            # Now get exponentiated value funcs
            # TODO this is bad api
            self._compute_value_function(m_tilde)
            # IF we had numerical issues in computing value functions
            # TODO should this flag just be part of the the return?
            if self.flag_exp_val_funcs_error:
                self.log_like_stored = Optimiser.LL_ERROR_VALUE
                self.grad_stored = np.ones(num_obs)
                self.flag_log_like_stored = True
                return self.log_like_stored, self.grad_stored

            value_funcs, exp_val_funcs = self._value_functions, self._exp_value_functions
            grad_orig = self.get_value_func_grad_orig(orig_index, m_tilde, exp_val_funcs)
            # print("grad orig", grad_orig, orig_index, "\n\t", exp_val_funcs)
            # print("grad orig", grad_orig)
            orig_utility = value_funcs[orig_index]
            # note we are adding to this, this is a progress value
            log_like_orig = -1 * (1 / mu) * orig_utility  # log probability from orign

            sum_inst_util = 0.0  # keep sum of instantaneous utility
            sum_current_attr = np.zeros(N)  # sum of observed attributes

            # # TODO this is inefficient since we've just casted a whole bunch of dense zeros,
            #    but not sure
            # #  that there is a convenient better way that doesn't already ruin sparsity.
            # #  easiest solution would be to make path length appear in input file
            path = obs_mat[n, :].toarray().squeeze()  # kill off singleton 2d dim
            # know all zeros are at end so can just count nonempty
            path_len = np.count_nonzero(path)

            # TODO vectorise all of this
            # -1 again since loop is forward set, probably could reindex
            for node_index in np.arange(1, path_len - 1):
                # entry [k,a] is in matrix index [k-1, a-1] : -1  since zero based
                current_node_index = path[node_index] - 1
                next_node_index = path[node_index + 1] - 1

                final_index_in_data = np.shape(self.network_data.incidence_matrix)[0]
                if next_node_index > final_index_in_data:
                    # I can't see when this would happen
                    print("WARN, dodgy bounds indexing hack occur in path tracing,"
                          " changed next node")
                    next_node_index = final_index_in_data

                sum_inst_util += v_mat[current_node_index, next_node_index]

                # TODO this should directly unwrap, checking things line up first - not easy to do
                # is the first dim is kind of a numpy list around sparse matrices
                # current data attribute (travel time, turn angle, ...)
                for attr in range(N):
                    sum_current_attr[attr] += self.network_data.data_array[attr][
                        current_node_index, next_node_index]
            #
            # print("sum current_attr", sum_current_attr, grad_orig)

            gradient_each_obs[n, :] = sum_current_attr - grad_orig  # LogLikeGrad in Code doc
            # print("grad each obs", gradient_each_obs[n, :])
            log_like_obs = 1 / mu * sum_inst_util + log_like_orig  # LogLikeFn in Code Doc

            # Some kind of successive over relaxation/ momentum
            log_like_cumulative += (log_like_obs - log_like_cumulative) / (n + 1)
            pre = grad_cumulative.copy()
            new_grad = (gradient_each_obs[n, :] - grad_cumulative) / (n + 1)
            # print("current grad weighted", new_grad)
            a = gradient_each_obs[n, :]
            b = -grad_cumulative
            c = n + 1

            grad_cumulative += (gradient_each_obs[n, :] - grad_cumulative) / (n + 1)
            # print("current grad weighted", grad_cumulative)


            # TODO negation of gradient_each_obs is used as global var elsewhere
        self.log_like_stored = -log_like_cumulative
        self.grad_stored = -grad_cumulative
        self.flag_log_like_stored = True
        return self.log_like_stored, self.grad_stored

    def get_value_func_grad_orig(self, orig_index, m_tilde, exp_val_funcs):
        """Note relies on 'self' for the attribute data.
        Computes the gradient of the value function evaluated at the origin
        Named GradValFnOrig in companion notes, see also GradValFn2 for component part
        - this function applies the 1/z_{orig} * [rest]_{evaluated at orig}
        """

        partial_grad = self._get_value_func_incomplete_grad(m_tilde, exp_val_funcs)
        b = m_tilde.toarray()
        # print("partial grad\n", partial_grad)
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
