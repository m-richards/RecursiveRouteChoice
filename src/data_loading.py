"""File containing IO data loading and standard preprocessing steps to construct
data matrices from input files"""
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse

INCIDENCE = "incidence.txt"
TRAVEL_TIME = 'travelTime.txt'
OBSERVATIONS = "observations.txt"
TURN_ANGLE = "turnAngle.txt"


def load_standard_path_format_csv(directory_path, delim=None, match_tt_shape=False,
                                  angles_included=True):
    """Returns the observations and list of matrices loaded from specified file directory
    :param directory_path: folder which contains files
    :type directory_path: str or os.PathLike[Any]
    :param delim: csv separator (i.e. ",", "\t", ...)
    :type delim: str
    :param match_tt_shape: trim all matrixes to have same shape as travel time mtrix
    :type match_tt_shape: bool
    :param angles_included: Boolean flag controlling whether angle file is expected or not.
    :type angles_included: bool
    :return: observations matrix and list of all data matrices
    :rtype: tuple[dok_matrix, list[dok_matrix]"""
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


def load_csv_to_sparse(fname, dtype=None, delim=None, square_matrix=True, shape=None) -> \
        sparse.coo_matrix:
    """IO function to load row, col, val CSV and return a sparse scipy matrix.
    :square_matix <bool> means that the input should be square and we will try to square it by
        adding a row (this is commonly required in data)
    :matrix_format_cast_function is the output format of the matrix to be returned. Given as a
    function to avoid having to specify string equivalents"""
    row, col, data = np.loadtxt(fname, delimiter=delim, unpack=True, dtype=dtype)
    # convert row and col to integers for coo_matrix
    # note we need this for float inputs since row cols still need to be ints to index
    rows_integer = row.astype(int)
    cols_integer = col.astype(int)
    if 0 not in rows_integer and 0 not in cols_integer:
        rows_integer = rows_integer - 1  # convert to zero based indexing if needed
        cols_integer = cols_integer - 1
    if not square_matrix:
        mat = sparse.coo_matrix((data, (rows_integer, cols_integer)), dtype=dtype)
    else:
        if shape is None:
            # note we add one to counteract minus one above
            max_dim = max(np.max(rows_integer), np.max(cols_integer)) + 1
            shape = (max_dim, max_dim)
        mat = sparse.coo_matrix((data, (rows_integer, cols_integer)),
                                dtype=dtype)
        mat.resize(shape)  # trim cols

    return mat


def resize_to_dims(matrix: sparse.dok_matrix, expected_max_shape,
                   matrix_name_debug="(Name not provided)"):
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


def load_tnpm_to_sparse(net_fpath, columns_to_extract=None, use_file_order_for_arc_numbers=True):
    """
    :param net_fpath path to network file
    :param columns_to_extract list of columns to keep. init_node and term_node are always kept
    # and form the basis of the arc-arc matrix.
    Currently only length is supported since the conversion from node to arc is not clear in
    this case.
    # Legal columns to extract are:
    #  capacity, length, free_flow_time, b, power, speed_limit, toll, link_type
    # Note that some of these will be constant across arcs and are redundant to include.

    :return
    :rtype [dict, scipy.sparse.coo_matrix, ...]


    """
    if columns_to_extract is None:
        columns_to_extract = ["length"]
    columns_to_extract = [i.lower() for i in columns_to_extract]
    if columns_to_extract[0] != "length":
        raise NotImplementedError("only support length, since other fields don't have natural "
                                  "averages.")
    net = pd.read_csv(net_fpath, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first and last columns
    net.drop(['~', ';'], axis=1, inplace=True)

    net2 = net[['init_node', 'term_node'] + columns_to_extract]
    node_set = set(net2['init_node'].unique()).union(set(net2['term_node'].unique()))
    nrows = net2.shape[0]
    arc_matrix = sparse.dok_matrix(sparse.coo_matrix((nrows, nrows)))

    arc_to_index_map = {}
    if use_file_order_for_arc_numbers:  # for consistency with any visuals
        for n, s, f in net2[['init_node', 'term_node']].itertuples():
            arc_to_index_map[(s, f)] = n

    print(arc_to_index_map)

    n = 0
    for first_arc_start in node_set:
        start_df = net2[net2['init_node'] == first_arc_start][['term_node', 'length']]
        for k in start_df.itertuples():
            first_arc_end = k.term_node
            start_len = k.length
            first_arc = first_arc_start, first_arc_end

            if first_arc not in arc_to_index_map:
                arc_to_index_map[first_arc] = n
                n += 1
            end_df = net2[net2['init_node'] == first_arc_end][['term_node', 'length']]
            for j in end_df.itertuples(index=False):
                end_arc_end = j.term_node
                end_len = j.length
                end_arc = (first_arc_end, end_arc_end)

                if end_arc not in arc_to_index_map:
                    arc_to_index_map[end_arc] = n
                    n += 1
                arc_matrix[arc_to_index_map[first_arc],
                           arc_to_index_map[end_arc]] = (start_len + end_len) / 2
    return arc_to_index_map, arc_matrix


def load_obs_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def write_obs_to_json(filename, obs, allow_rewrite=False):
    if os.path.exists(filename) and allow_rewrite is False:
        raise IOError("File already exists. Specify 'force_override=True' to enable re-writing.")
    with open(filename, 'w') as f:
        json.dump(obs, f)
