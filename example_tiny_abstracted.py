# Figure 2 network from fosgerau - have value function of single param cost
import numpy as np
import scipy
from main import DataSet, get_value_func_grad
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix

import os
from os.path import join

# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network
subfolder = "ExampleTiny"  # big data from classical v2
folder = join("Datasets", subfolder)
INCIDENCE = "incidence.txt"
TRAVEL_TIME = 'travelTime.txt'
OBSERVATIONS = "observations.txt"
file_incidence = os.path.join(folder, INCIDENCE)
file_travel_time = os.path.join(folder, TRAVEL_TIME)
file_obs = os.path.join(folder, OBSERVATIONS)
row, col, data = np.loadtxt(file_travel_time, unpack=True)
incidence_data = np.ones(len(data))


def load_csv_to_sparse(fname, dtype=None, delim=" ", matrix_cast=None,
                       square_matrix=True):
    if matrix_cast is None:
        matrix_cast = csr_matrix
    row, col, data = np.loadtxt(fname, delimiter=delim, unpack=True, dtype=dtype)
    # print(row, col, data)
    # convert row and col to integers for coo_matrix
    # note we need this for float inputs since row cols still need to be ints to index
    rows_integer = row.astype(int)
    cols_integer = col.astype(int)
    if 0 not in rows_integer and 0 not in cols_integer:
        rows_integer = rows_integer - 1  # convert to zero based indexing if needed
        cols_integer = cols_integer - 1


    mat = matrix_cast(coo_matrix((data, (rows_integer, cols_integer)), dtype=dtype))
    if mat.shape[0] == mat.shape[1]-1 and square_matrix:
        # this means we have 1 less row than columns from our input data
        # i.e. missing the final k==d row with no successors
        ncols = np.shape(mat)[1]
        sparse_zeros = csr_matrix((1, ncols))
        mat = scipy.sparse.vstack((mat, sparse_zeros))
    return mat


travel_times_mat = load_csv_to_sparse(file_travel_time)
incidence_mat = load_csv_to_sparse(file_incidence, dtype='int')

# Get observations matrix - note: observation matrix is in sparse format, but is of the form
#   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
obs_mat = load_csv_to_sparse(file_obs, dtype='int',square_matrix=False)

data_struct = DataSet(travel_times=travel_times_mat, incidence_matrix=incidence_mat,
                      turn_angles=None)
np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 100

# from optimisers import log_likelihood
beta = np.array(-1.5)# default value, 1d for now
# log_likelihood(beta, data_struct, obs_mat)

beta_vec = beta
data = data_struct
obs = obs_mat
mu = 1

# temp func
# def log_likelihood(beta_vec, data:DataSet, obs, mu=1):
N = data.n_dims

# grad = np.zeros(N)

v_mat = data_struct.get_short_term_utility()# capital u in tien mai's code
data_struct.get_exponential_utility_matrix()
m_mat = data_struct.get_exponential_utility_matrix()
value_funcs, exp_val_funcs = data_struct.get_value_functions(return_exponentiated=True)

grad = get_value_func_grad(m_mat, data_struct, exp_val_funcs)

# iterate through observation number
num_obs = np.shape(obs)[0]
gradient_all_obs = np.zeros((num_obs, data_struct.n_dims))
for n in range(num_obs):
    dest = obs[n, 0]
    orig = obs[n, 1]

    orig_utility = value_funcs[orig]

    grad_orig = grad[:, orig]/ exp_val_funcs[orig]

    cumulative_inst_util = 0
    cumulative_attr_sum = np.zeros(data_struct.n_dims) # sum of observed attributes
    first_action = obs[n, 2]

    ln_pn = -1 * (1/mu) * orig_utility # log probability from orign

    gradient_all_obs[n, :] = - grad_orig
    path = obs[n, :]
    print("path is ", path.toarray())
    path_len = np.count_nonzero(path) # know all zeros are at end

    # TODO vectorise all of this
    for node_index in np.arange(1, path_len-1):
    # for node in path[1:]:
        max_index =path[node_index+1]
        if max_index> np.shape(incidence_mat)[0]:
            # I can't see when this would happen
            print("WARN, dodgy bounds indexing hack occur in path tracing")
            max_index = np.shape(incidence_mat)[0]

        cumulative_inst_util += v_mat[path[node_index], max_index]
        for j in range(data_struct.n_dims):
            cumulative_attr_sum[j, :] += data_struct.data_array[j][
                path[node_index], max_index]








