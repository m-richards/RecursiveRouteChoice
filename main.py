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

class DataSet(object):
    """Really just a struct
    Added short term utility computation as well since this is handy in a bunch of places

    """

    def __init__(self, travel_times, incidence_matrix, turn_angles=None, initial_beta=-1.5, mu=1):

        self.travel_times = travel_times
        self.incidence_matrix = incidence_matrix
        self.turn_angles = turn_angles
        # TODO include uturn and left turn penalties when relevant
        self.data_array = np.array([self.travel_times, ])
        self.n_dims = len(self.data_array)
        self.beta_vec = np.array([initial_beta for i in range(self.n_dims)])

        self._short_term_utility = None
        self._exponential_utility_matrix = None
        self._value_functions = None
        self._exp_value_functions = None
        self.mu = mu

        self.update_beta(initial_beta) # to this to refresh default quantities

    def update_beta(self, new_beta_vec):
        self.beta_vec = new_beta_vec
        # self._beta_changed = True
        # TODO delay this from happening until the update is needed - use flag

        self._compute_short_term_utility()
        self._compute_exponential_utility_matrix()
        self._compute_value_matrix()
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
        nonzero_entries = np.nonzero(self.incidence_matrix)
        m_mat[nonzero_entries] = np.exp(1 / self.mu * m_mat[nonzero_entries])
        # This isn't a square matrix since in our data format,
        # dest row has no successors so is missing from incidence matrix
        # add a zero row at end to square matrix
        # ncols = np.shape(self.incidence_matrix)[1]
        # sparse_zeros = csr_matrix((1, ncols))
        # TODO can we drop a column in both directions instead?
        #   needs some more thinking
        # self._exponential_utility_matrix = scipy.sparse.vstack((m_mat, sparse_zeros))
        # updated input forma so that all matrices have the right dims
        self._exponential_utility_matrix = m_mat



    def get_exponential_utility_matrix(self):
        """ # TODO can cached this if I deem it handy"""

        return self._exponential_utility_matrix

    def _compute_value_matrix(self):
        pass
        error_flag = 0
        ncols = np.shape(self.incidence_matrix)[1]
        rhs = scipy.sparse.lil_matrix((ncols, 1)) # supressing needless sparsity warning
        rhs[-1, 0] = 1
        # (I-M)z =b
        M = self.get_exponential_utility_matrix()

        # TODO tien takes the absolute value for numerical safety here
        A = identity(ncols) - M
        z_vec = splinalg.spsolve(A, rhs)

        if np.min(z_vec) <= 1e-10:  # TODO add an optional thresh on this?
            error_flag = 1

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





def get_value_func_grad(M, data:DataSet, expV):
    """Tien mai method for returning the gradient of the value function,
    need to check consistency with maths.

    Should actually multiply this by mu.

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
        grad_v[i,:] = splinalg.spsolve(A ,v) # this had the strange property that grad = v
        # in early testing
        # print(grad_v)
    return grad_v



# get 1/mu * Grad(V0)
#%%



# def getGradV0(M, Att, op, expV, origin):
#     gradExpV = getGradExpV(M, Att, op, expV);
#     gradV0 = zeros(1,op.n);
#     for i = 1:op.n
#         gradV0(i) =  gradExpV(i).Value(origin)/expV(origin);
#     return gradV0






