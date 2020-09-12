# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)
import os

import numpy as np
import time
import scipy
from scipy.sparse import linalg as splinalg
from data_loading import load_standard_path_format_csv
from data_processing import AngleProcessor
from recursive_route_choice import RecursiveLogitModelEstimation, ModelDataStruct, RecursiveLogitModel
from optimisers import LineSearchOptimiser, OptimHessianType
from scipy import sparse
# np.seterr(all='raise')  # all='print')
import warnings
# np.set_printoptions(precision=12, suppress=True)
np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500
"""Default assumptions for now
-   ban all uturns in the network
-   force going to dest if it is possible (remove all other choices from second to final arc)
        in m_tilde matrix (does this actually ban an option?)
-   assume v(k|d)=0 (standard assumption)

TODO need to investigate if these conditions imply that system has redundant dimension 
 Think it does so should be able to make it slightly smaller"""


def zero_pad_mat(mat, top=False, left=False, bottom=False, right=False):
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
        return mat.todok() # don't want to stay as coo
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




# import warnings
# warnings.simplefilter("error")

# DATA

Distances = np.array(
    [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
     [3.5, 3, 4, 0, 2.5, 3, 3, 0],
     [4.5, 4, 5, 0, 0, 0, 4, 3.5],
     [3, 0, 0, 2, 2, 2.5, 0, 2],
     [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
     [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
     [0, 3, 4, 0, 2.5, 3, 3, 2.5],
     [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])
# # TODO note angles need to be mixed with incidence to determine 0 angle from missing arc
# # or we could encode 0 as 360 - > probably better
Angles = np.array(
    [[180, -90, -45, 360, 90, 0, 0, 0],
     [90, 180, -135, 0, -90, -45, 360, 0],
     [45, 135, 180, 0, 0, 0, -90, 360],
     [360, 0, 0, 180, -90, 135, 0, 90],
     [-90, 90, 0, 90, 180, -135, -90, 0],
     [0, 45, 0, -135, 135, 180, -135, 135],
     [0, 360, 90, 0, 90, 135, 180, -90],
     [0, 0, 360, -90, 0, -135, 90, 180]])

# REduced - taking arcs 1, 2 and 5 of full eg
# x = 0.12
# Distances = np.array(
#     [[x, x/2+2],
#      [x/2+2, 2],
#      ])
#
#
#
#
#
#
# # TODO note angles need to be mixed with incidence to determine 0 angle from missing arc
# # or we could encode 0 as 360 - > probably better
# Angles = np.array(
#     [[180, 360,],
#      [360, 180,]])

# TODO ban u-turns:
# Distances = Distances - np.diag(Distances)
# Angles = Angles - np.diag(Angles)
# print(Distances)


# # Reduced only arc 1
#
# Distances = np.array(
#     [[4],
#
#      ])
# # TODO note angles need to be mixed with incidence to determine 0 angle from missing arc
# # or we could encode 0 as 360 - > probably better
# Angles = np.array(
#     [[180],
#      ])
#

# note dists are symmetric and angles minus main diag are antisymmetric - except for 360s which
# are zeros.
print("dists")
print(Distances)
print("angles")
print(Angles)


from scipy.sparse import dok_matrix, identity

incidence_mat = (Distances > 0).astype(int)

print("orig incidence\n", incidence_mat)

angles_rad = AngleProcessor.to_radians(Angles)
#
#
# time_io_start = time.time()
# # subfolder ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# # subfolder = "ExampleTiny"
# subfolder = "ExampleTinyModifiedObs"
# folder = os.path.join("../Datasets", subfolder)
#
# obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
# incidence_mat, travel_times_mat, angle_cts_mat = attrs
left, right, neutral, u_turn = AngleProcessor.get_turn_categorical_matrices(dok_matrix(
 angles_rad), dok_matrix(incidence_mat))
# incidence matrix which only has nonzero travel times - rather than what is specified in file
distances = dok_matrix(Distances)
# data_list = np.array([distances, left])
data_list = np.array([distances])
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances", "u_turn"))
m = -1
# beta_vec = np.array([-1, -1])
beta_vec = np.array([-1])
import optimisers as op
optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)
model = RecursiveLogitModel(network_struct,  user_obs_mat=None,
                                      initial_beta=beta_vec, mu=1)

# To generate obs we want to actually simulate and draw errors
# first we need to have all the deterministic utility
print(model.get_beta_vec())
model._compute_short_term_utility()
model._compute_exponential_utility_matrix()
short_term_utility = np.sum(model.get_beta_vec() * model.data_array)


print(left.toarray())
print("short util\n", model.get_short_term_utility().toarray())
exp_utility_matrix = model.get_exponential_utility_matrix()
print("m_orig = \n", exp_utility_matrix.toarray())
obs_per_pair = 2
O = range(distances.shape[0])
D = range(distances.shape[1])
# start at an arc, end at a node, fix ending at a node by appending an arc
# assume for simplicity that the nodes are the nodes at the centre of my arcs

# short term util has a fixed zero cost for the rotating (dest) col
# TODO change if nonzero
# TODO note don't need to padd bottom
short_term_utility = zero_pad_mat(short_term_utility, right=True)

seed = 1
rng = np.random.default_rng(seed)
from scipy import linalg

m, n = exp_utility_matrix.shape


print( 2 *( 80* "=" + "\n"))



assert m == n
for dest in [1,6]:  # D: # (0,2) specifies a node via an adjacency list
    print("dest = ", dest, "augmented dest col =", m)
    # for each dest need to compute value function
    m_tilde = exp_utility_matrix.copy()
    incidence_tilde = incidence_mat.copy() # TODO review if necessary

    # any arcs which connect to desk are zero
    # incidence must locally change too. Need to do a small pen and paper example
    # augmenting an arc means that stuff doesn't need to change as much
    len_with_aug_dest = m + 1  # index is this -1

    m_tilde = zero_pad_mat(m_tilde, bottom=True, right=True)
    incidence_tilde = zero_pad_mat(incidence_tilde, bottom=True, right=True)

    # print("modified incidence int\n", incidence_tilde, type(incidence_tilde))

    # m_tilde is always going to have same dims, this additional 1 is non destructive,
    # we would just need to reset this column back to zeros and we could reuse matrix
    # TODO this analysis is wrong in terms of current implementation - we have seperate incidence
    #  mat
    m_tilde[:-1, :-1] = exp_utility_matrix
    m_tilde[dest, :] = 0.0  # TODO do we do this? # enforces going to dest
    incidence_tilde[dest, :] = 0

    m_tilde[dest, -1] = 1  # exp(v(a|k)) = 1 when v(a|k) = 0 # try 0.2
    incidence_tilde[dest, -1] = 1
    print("modified incidence\n", incidence_tilde)
    # m_tilde[dest, -1] = 0.2  # exp(v(a|k)) = 1 when v(a|k) = 0 # try 0.2
    # Things to observe with x = 0.12 - can get value function which are positive and optimal at
    # the origin! really bad
    # if we enforce going to dest, still end up with orig positive when it shouldn't be
    # if we make v(d|k) negative can fix stuff to negative but still not the right order of values

    print("m tilde\n", m_tilde.toarray())
    a_mat = identity(len_with_aug_dest) - m_tilde
    print("amat\n", a_mat.toarray())
    print("det = ", linalg.det(a_mat.toarray()), linalg.det(exp_utility_matrix.toarray()))
    # implicitly we also get m+1th row being all zeros which is what we want since no succesors
    rhs = scipy.sparse.lil_matrix((len_with_aug_dest, 1))  # suppressing needless sparsity warning
    rhs[-1, 0] = 1
#
    z_vec = splinalg.spsolve(a_mat, rhs)  # rhs has to be (n,1)
    # print("a mat = \n", a_mat.toarray())
    # print("rhs = \n", rhs.toarray())
    z_vec = np.atleast_2d(z_vec).T  #
    print("z_vec = \n", z_vec)
    value_funcs = np.log(z_vec)
    if np.any(value_funcs > 0):
        warnings.warn(f"WARNING: Positive value functions: {value_funcs[value_funcs > 0]}",
                      category='error')
    print("raw val funcs\n", value_funcs)
    # print(" valf", value_funcs)
    # short_term_utility
    # print("stu")
    # print(short_term_utility)
    # v_op = short_term_utility[0, 1]
    # v_pp = short_term_utility[0, 0]
    # print("theory V(p) with dest correction", np.log(np.exp(v_op)/ (1- np.exp(v_pp))))
    # print("cond", np.linalg.cond(a_mat.toarray()))
    # continue
    for orig in [0,1,2,7]:  # O:
        # print(40 *"-")
        # print("Origin : ", orig, "targer = ", n+1)
        if orig ==dest: # silly case
            print("skipping o==d for o=", orig)
            continue
        for i in range(obs_per_pair):
            # while we haven't reached the dest
            current_arc = orig
            path_string = f"Start: {orig}"
            count = 0
            while current_arc != len_with_aug_dest-1:  # index of augmented dest arc
                count += 1
                if count > 10:
                    zzz
                current_incidence_col = incidence_tilde[current_arc, :]
                # all arcs current arc connects to
                neighbour_arcs = current_incidence_col.nonzero()[0]

                eps = rng.gumbel(loc=-np.euler_gamma, scale=1, size=len(neighbour_arcs))

                value_functions_observed = (short_term_utility[current_arc, neighbour_arcs]
                                            + value_funcs[neighbour_arcs].T
                                            + eps)

                if np.any(np.isnan(value_functions_observed)):
                    raise ValueError("beta vec is invalid, gives non real solution")
                # value_functions_observed = np.nan_to_num(value_functions_observed, nan=-np.Inf)
                # print(f"value functions observed at {current_arc}:\n arcs:\n{neighbour_arcs.T}\n "
                #       f"vals:\n\n{value_functions_observed.T}")

                next_arc_index = np.argmax(value_functions_observed)
                next_arc = neighbour_arcs[next_arc_index]
                # print(np.max(value_functions_observed), next_arc)
                path_string += f" -> {next_arc}"
                current_arc = next_arc


                # print(current_incidence_col, neighbour_arcs)
                # print("dims")
                # print(short_term_utility[current_arc, neighbour_arcs].shape,
                #       value_funcs[neighbour_arcs].shape, eps.shape)
                # print("out")
                # print(value_functions_observed)
                # print(path_string)
                # break
            print(path_string +": Fin")
            # break
#
#
#
#
#
#

# print(type(left), type(distances))

# print(type(model.get_beta_vec()), type(model.data_array))
#
# print(type(model.get_beta_vec()[0]), type(model.data_array[0]))
#
# print(model.data_array[0].toarray())
# # model._compute_short_term_utility()
# # model.get_short_term_utility()
#
#
# value_functions = model.get
