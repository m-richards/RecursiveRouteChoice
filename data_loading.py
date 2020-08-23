"""File containing IO data loading and standard preprocessing steps to construct
data matrices from input files"""
import os

import numpy as np
import scipy

from scipy.sparse import coo_matrix

INCIDENCE = "incidence.txt"
TRAVEL_TIME = 'travelTime.txt'
OBSERVATIONS = "observations.txt"
TURN_ANGLE = "turnAngle.txt"


def load_standard_path_format_csv(directory_path, delim=None, match_tt_shape=False,
                                  angles_included=True):
    """Expects standardised filenames in directory direction_name.
    Delim is csv delimiter, match tt_shape is for rescaling incidence and angle matrices
    so that dimensions are consistent"""
    file_incidence = os.path.join(directory_path, INCIDENCE)
    file_travel_time = os.path.join(directory_path, TRAVEL_TIME)
    file_turn_angle = os.path.join(directory_path, TURN_ANGLE)
    file_obs = os.path.join(directory_path, OBSERVATIONS)

    travel_times_mat = load_csv_to_sparse(file_travel_time, delim=delim,
                                          ).todok()
    if match_tt_shape:
        fixed_dims = travel_times_mat.shape
    else:
        fixed_dims = None

    incidence_mat = load_csv_to_sparse(
        file_incidence, dtype='int', delim=delim).todok()
    if fixed_dims is not None:
        resize_to_dims(incidence_mat, fixed_dims, matrix_name_debug="Incidence Mat")
    # Get observations matrix - note: observation matrix is in sparse format, but is of the form
    #   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
    obs_mat = load_csv_to_sparse(
        file_obs, dtype='int', square_matrix=False, delim=delim).todok()

    to_return_data = [incidence_mat, travel_times_mat]

    if angles_included:
        turn_angle_mat = load_csv_to_sparse(file_turn_angle, delim=delim).todok()
        if fixed_dims is not None:
            resize_to_dims(turn_angle_mat, fixed_dims, matrix_name_debug="Turn Angles")
        to_return_data.append(turn_angle_mat)

    return obs_mat, to_return_data



def load_csv_to_sparse(fname, dtype=None, delim=None, square_matrix=True, shape=None) -> coo_matrix:
    """IO function to load row, col, val CSV and return a sparse scipy matrix.
    :square_matix <bool> means that the input should be square and we will try to square it by
        adding a row (this is commonly required in data)
    :matrix_format_cast_function is the output format of the matrix to be returned. Given as a
    function to avoid having to specify string equivalents"""
    row, col, data = np.loadtxt(fname, delimiter=delim, unpack=True, dtype=dtype)
    # print(row, col, data)
    # convert row and col to integers for coo_matrix
    # note we need this for float inputs since row cols still need to be ints to index
    rows_integer = row.astype(int)
    cols_integer = col.astype(int)
    # print(fname, np.max(row), np.max(col))
    if 0 not in rows_integer and 0 not in cols_integer:
        rows_integer = rows_integer - 1  # convert to zero based indexing if needed
        cols_integer = cols_integer - 1
    if not square_matrix:
        mat = coo_matrix((data, (rows_integer, cols_integer)), dtype=dtype)
    else:
        if shape is None:
            # note we add one to counteract minus one above
            max_dim = max(np.max(rows_integer), np.max(cols_integer))+1
            # print(max_dim)
            shape = (max_dim, max_dim)
        mat = coo_matrix((data, (rows_integer, cols_integer)),
                         dtype=dtype)
        # print(mat.shape)
        mat.resize(shape) # trim cols

    # if mat.shape[0] == mat.shape[1] - 1 and square_matrix:
    #     # this means we have 1 less row than columns from our input data
    #     # i.e. missing the final k==d row with no successors
    #     ncols = np.shape(mat)[1]
    #     sparse_zeros = csr_matrix((1, ncols))
    #     mat = scipy.sparse.vstack((mat, sparse_zeros))
    return mat


def resize_to_dims(matrix: scipy.sparse.dok_matrix, expected_max_shape, matrix_name_debug="(Name not "
                                                                                    "provided)"):
    """Resizes matrix to specified dims, issues warning if this is losing data from the matrix.
    Application is more general than the current error message suggests.
    Note the fact that the matrix is sparse is essential, numpy resize behaves differently to
    scipy.
    Note also, this upcasts dimensions if too small

    Note we use this since it is easier to read in a too large or small matrix from file and
    correct than limit the size from IO - exceptions get thrown
    """
    if (matrix.shape[0] > expected_max_shape[0]) or (matrix.shape[1] > expected_max_shape[1]):
        # warnings.warn( note note using warnings since I'm trying to catch all warnings
        # but this is an expected warning (but issued so that I don't forget at a later date)
        print(f"Warning: '{matrix_name_debug}' Matrix has dimensions {matrix.shape} which exceeds "
              f"expected size "
              f"{expected_max_shape}. Matrix has been shrunk (default size inferred from "
              f"travel time matrix)", )
    # resize in any case
    matrix.resize(*expected_max_shape)





