import scipy
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse import linalg as splinalg
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
import numpy as np

class RecursiveLogitDataSet(object):
    """Generic struct which stores all the arc attributes together in a convenient manner"""
    def __init__(self, travel_times, incidence_matrix, turn_angles=None):
        self.travel_times = travel_times
        self.incidence_matrix = incidence_matrix
        self.turn_angles = turn_angles
        # TODO include uturn and left turn penalties when relevant
        #   I think that these should be done by some other preprocessing step before this class,
        # this should be a generic data class. Is not for now
        self.data_array = np.array([self.travel_times, self.travel_times])
        self.n_dims = len(self.data_array)


class RecursiveLogitModel(object):
    """Abstraction of the linear algebra type relations on the recursive logit model to solve
    the matrix system and compute log likelihood.

    Doesn't handle optimisation directly (but does compute log likelihood), should be
    passed into optimisation algorithm in a clever way

    """

    def __init__(self, data_struct: RecursiveLogitDataSet, initial_beta=-1.5, mu=1):
        self.data = data_struct
        self.data_array = data_struct.data_array
        self.n_dims = len(self.data_array)
        # TODO maybe have to handle more complex initialisations
        self.beta_vec = np.array([initial_beta for _ in range(self.n_dims)])
        self.mu = mu

        self._short_term_utility = None
        self._exponential_utility_matrix = None
        self._value_functions = None
        self._exp_value_functions = None

        self.update_beta_vec(self.beta_vec) # to this to refresh default quantities

    def get_beta_vec(self):
        """Getter is purely to imply that beta vec is not a fixed field"""
        return self.beta_vec

    def update_beta_vec(self, new_beta_vec):
        """Change the current parameter vector beta and update intermediate results which depend
        on this"""
        self.beta_vec = new_beta_vec
        # self._beta_changed = True
        # TODO delay this from happening until the update is needed - use flag

        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        self._compute_value_function_matrix()
        # TODO make sure new stuff gets added here

    def _compute_short_term_utility(self):
        self.short_term_utility =np.sum(self.beta_vec * self.data_array)

    def get_short_term_utility(self):
        """Returns v(a|k)  for all (a,k) as 2D array,
        uses current value of beta"""

        return self.short_term_utility

    def _compute_exponential_utility_matrix(self):

        # explicitly do need this copy since we modify m_mat
        m_mat = self.get_short_term_utility().copy()
        # note we currently use incidence matrix here, since this distinguishes the
        # genuine zero arcs from the absent arcs
        # (since data format has zero arcs for silly reasons)
        nonzero_entries = np.nonzero(self.data.incidence_matrix)
        # TODO this throws a sparsity efficiency warning
        m_mat[nonzero_entries] = np.exp(1 / self.mu * m_mat[nonzero_entries])
        self._exponential_utility_matrix = m_mat



    def get_exponential_utility_matrix(self):
        """ # TODO can cached this if I deem it handy.
        Returns M_{ka} matrix
        """

        return self._exponential_utility_matrix

    def _compute_value_function_matrix(self):
        """Solves the system Z = Mz+b and stores the output for future use.
        Has rudimentary flagging of errors but doesn't attempt to solve any problems"""
        error_flag = 0
        ncols = np.shape(self.data.incidence_matrix)[1]
        rhs = scipy.sparse.lil_matrix((ncols, 1))  # supressing needless sparsity warning
        rhs[-1, 0] = 1
        # (I-M)z =b
        M = self.get_exponential_utility_matrix()

        # TODO tien takes the absolute value for numerical safety here
        A = identity(ncols) - M
        z_vec = splinalg.spsolve(A, rhs)

        if np.min(z_vec) <= 1e-10:  # TODO add an optional thresh on this?
            error_flag = 1
            raise ValueError("value function has too small entries")

        if np.any(z_vec) < 0:
            raise ValueError("value function had negative solution, cannot take "
                             "logarithm")
        # Note the transpose here is not mathematical, it is scipy being
        # lax about row and column vectors
        if linalg.norm(
                A @ z_vec - rhs.transpose()) > 1e-3:  # residual - i.e. ill conditioned solution
            raise ValueError("value function solution does not satisfy system well.")
        self._value_functions = np.log(z_vec)
        self._exp_value_functions = z_vec

    def get_value_functions(self, return_exponentiated=False):
        if return_exponentiated:
            return self._value_functions, self._exp_value_functions
        return self._value_functions

    def get_new_beta_log_like(self, beta_vec, obs_mat):
        self.update_beta_vec(beta_vec)
        return  self.get_log_likelihood(obs_mat)

    def get_log_likelihood(self, obs_mat):
        """Compute the log likelihood of the data with the current beta vec"""

        num_obs, path_max_len = np.shape(obs_mat)
        # local references with idomatic names
        N = self.n_dims # number of attributes in data
        mu = self.mu
        v_mat = self.get_short_term_utility()  # capital u in tien mai's code
        m_mat = self.get_exponential_utility_matrix()
        value_funcs, exp_val_funcs = self.get_value_functions(return_exponentiated=True)

        grad = get_value_func_grad(m_mat, self, exp_val_funcs)

        log_like_cumulative = 0.0  # weighting of all observations
        grad_cumulative = np.zeros(N) # gradient combined across all observations
        gradient_each_obs = np.zeros((num_obs, N)) # store gradient according to each obs
        # tODO this looks redundant to store these at
        #  the moment but this is a global variable in TIEN's code

        # iterate through observation number
        for n in range(num_obs):
            dest = obs_mat[n, 0]  # this was for adding extra cols, but we handle this without bad
            # fixes
            orig_index = obs_mat[n, 1] - 1   # subtract 1 for zero based python
            # first_action = obs_mat[n, 2]

            grad_orig = grad[:, orig_index] / exp_val_funcs[orig_index]
            orig_utility = value_funcs[orig_index]
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

                final_index_in_data = np.shape(self.data.incidence_matrix)[0]
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
                    sum_current_attr[attr] += self.data.data_array[attr][
                        current_node_index, next_node_index]
            #
            gradient_each_obs[n, :] = sum_current_attr - grad_orig  # GradEachObs in Code doc
            log_like_obs = 1 / mu * sum_inst_util + log_like_orig  # LogLikeObs in Code Doc

            # Some kind of successive over relaxation/ momentum
            log_like_cumulative += (log_like_obs - log_like_cumulative) / (n + 1)

            grad_cumulative += (gradient_each_obs[n, :] - grad_cumulative) / (n + 1)

            # TODO negation of gradient_each_obs is used as global var elsewhere

        return -log_like_cumulative, grad_cumulative


# tODO should these be methods, or just algorithmic pieces
def get_value_func_grad(M, data: RecursiveLogitModel, expV):
    """Tien mai method for returning the gradient of the value function,
    need to check consistency with maths.

    TODO Note tien my function returns 1/mu * this

    Returns a data.n_dims * shape(M)[0] matrix
    - gradient in each x component at every node"""
    # TODO check if dimensions should be transposed
    grad_v = np.zeros((data.n_dims, np.shape(M)[0]))
    I = identity(np.shape(M)[0])
    A = I - M

    for i in range(data.n_dims):
        current_attribute = data.data_array[i]
        # print("M")
        # print(M.toarray())
        # print("current attribute")
        # print(current_attribute.toarray())

        # note sparse matrices have * be matrix product for some reason
        u = M.multiply(current_attribute)
        # print("u")
        # print(u.toarray())
        v = u * expV
        # print("expV")
        # print(expV)
        # print("v")
        # print(v)
        grad_v[i,:] = splinalg.spsolve(A, v)  # this had the strange property that grad = v
        # in early testing
        # print(grad_v)

    return data.mu * grad_v









