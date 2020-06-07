# Figure 2 network from fosgerau - have value function of single param cost
import numpy as np
import scipy
from main import DataSet
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


def load_csv_to_sparse(fname, dtype=None, delim=" ", matrix_cast=None):
    if matrix_cast is None:
        matrix_cast = csr_matrix
    row, col, data = np.loadtxt(fname, delimiter=delim, unpack=True, dtype=dtype)
    print(row, col, data)
    # convert row and col to integers for coo_matrix
    # note we need this for float inputs since row cols still need to be ints to index
    rows_integer = row.astype(int)
    cols_integer = col.astype(int)
    if 0 not in rows_integer and 0 not in cols_integer:
        rows_integer = rows_integer - 1  # convert to zero based indexing if needed
        cols_integer = cols_integer - 1

    return matrix_cast(coo_matrix((data, (rows_integer, cols_integer)), dtype=dtype))


travel_times_mat = load_csv_to_sparse(file_travel_time)
incidence_mat = load_csv_to_sparse(file_incidence, dtype='int')

# Get observations matrix - note: observation matrix is in sparse format, but is of the form
#   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
obs_mat = load_csv_to_sparse(file_obs, dtype='int')

data_struct = DataSet(travel_times=travel_times_mat, incidence_matrix=incidence_mat,
                      turn_angles=None)
