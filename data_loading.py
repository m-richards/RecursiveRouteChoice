"""File containing IO data loading and standard preprocessing steps to construct
data matrices from input files"""

import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix
import warnings

LEFT_TURN_THRESH = -0.5236  # 30 degrees
U_TURN_THRESH = 3.1  # radians


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


def resize_to_dims(matrix: scipy.sparse.dok_matrix, expected_max_shape, matrix_name):
    """Resizes matrix to specified dims, issues warning if this is losing data from the matrix.
    Application is more general than the current error message suggests.
    Note the fact that the matrix is sparse is essential, numpy resize behaves differently to
    scipy.
    Note also, this upcasts dimensions if too small
    """
    if (matrix.shape[0] > expected_max_shape[0]) or (matrix.shape[1] > expected_max_shape[1]):
        # warnings.warn( note note using warnings since I'm trying to catch all warnings
        # but this is an expected warning (but issued so that I don't forget at a later date)
        print(f"Warning: '{matrix_name}' Matrix has dimensions {matrix.shape} which exceeds "
              f"expected size "
              f"{expected_max_shape}. Matrix has been shrunk (default size inferred from "
              f"travel time matrix)", )
    # resize in any case
    matrix.resize(*expected_max_shape)





def get_uturn_categorical_matrix(turn_angle_mat, u_turn_thresh=None):
    """Assumes that angles are between -pi and pi"""
    u_turn_thresh = u_turn_thresh if u_turn_thresh is not None else U_TURN_THRESH
    return (np.abs(turn_angle_mat) > u_turn_thresh).astype(int).todok()


def get_left_turn_categorical_matrix(turn_angle_mat, left_turn_thresh=None,
                                     u_turn_thresh=None):
    """Assumes that angles are between -pi and pi"""
    if left_turn_thresh is None:
        left_turn_thresh = LEFT_TURN_THRESH
    if u_turn_thresh is None:
        u_turn_thresh = U_TURN_THRESH
    # Note this is done strangely since scipy doesn't support & conditions on
    # sparse matrices. Also is more efficient to only do comparison on nonzero (since this is dense)
    nz_rows, nz_cols = turn_angle_mat.nonzero()

    nz_left_turns_mask = np.array(
        (turn_angle_mat[nz_rows, nz_cols].toarray() < left_turn_thresh) &  # turn is to the left
        (turn_angle_mat[nz_rows, nz_cols].toarray() > -u_turn_thresh))[0]  # turn is not a uturn
    # note testing todense suggests faster or at least not worse, supresses error
    masked_rows = nz_rows[nz_left_turns_mask]
    masked_cols = nz_cols[nz_left_turns_mask]
    vals = np.ones(len(masked_cols), dtype='int')
    left_turn_mat = scipy.sparse.coo_matrix(
        (vals, (masked_rows, masked_cols)), shape=turn_angle_mat.shape, dtype='int')

    return left_turn_mat.todok()


def get_incorrect_tien_turn_matrices(turn_angle_mat, left_turn_thresh=LEFT_TURN_THRESH,
                                     u_turn_thresh=U_TURN_THRESH):
    """A function to generate the turn matrices equivalently tien mai code for comparison
    purposes, note that this is logically incorrect though.
    Deliberately copies confusing logic since it is trying to be consistent.
    Computes turn angles correctly in a convoluted way.
    Skips computing leftTurn since this is overridden to be an incidence matrix"""
    # Angles between -pi and pi
    u_turn_mat = (np.abs(turn_angle_mat) > u_turn_thresh).astype(int)

    new_turn_angles = turn_angle_mat.copy()
    nonzero_turn_angles = np.nonzero(turn_angle_mat)
    for x, y in zip(*nonzero_turn_angles):
        i = (x, y)
        current_turn_angle = turn_angle_mat[i]
        if abs(current_turn_angle) < u_turn_thresh:  # not a uturn
            if current_turn_angle >= 0:  # turn to the right
                new_turn_angles[i] = 0  # remove right turns from matrix (not sure why)
            else:  # straight or to the left
                if current_turn_angle < left_turn_thresh:
                    new_turn_angles[i] = 1
                else:
                    new_turn_angles[i] = 0
        else:
            new_turn_angles[i] = 0

    return new_turn_angles, u_turn_mat
