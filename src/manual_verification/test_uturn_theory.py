# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)
import os

import numpy as np
import time
import scipy
from scipy.sparse import linalg as splinalg
from data_loading import load_standard_path_format_csv
from data_processing import AngleProcessor
from main import RecursiveLogitModelEstimation, RecursiveLogitDataStruct, RecursiveLogitModel
from optimisers import LineSearchOptimiser, OptimHessianType

# np.seterr(all='raise')  # all='print')

# np.set_printoptions(precision=12, suppress=True)
np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500


# import warnings
# warnings.simplefilter("error")

# DATA

# Distances = np.array(
#     [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
#      [3.5, 3, 4, 0, 2.5, 3, 3, 0],
#      [4.5, 4, 5, 0, 0, 0, 4, 3.5],
#      [3, 0, 0, 2, 2, 2.5, 0, 2],
#      [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
#      [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
#      [0, 3, 4, 0, 2.5, 3, 3, 2.5],
#      [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])
# # TODO note angles need to be mixed with incidence to determine 0 angle from missing arc
# # or we could encode 0 as 360 - > probably better
# Angles = np.array(
#     [[180, -90, -45, 360, 90, 0, 0, 0],
#      [90, 180, -135, 0, -90, -45, 360, 0],
#      [45, 135, 180, 0, 0, 0, -90, 360],
#      [360, 0, 0, 180, -90, 135, 0, 90],
#      [-90, 90, 0, 90, 180, -135, -90, 0],
#      [0, 45, 0, -135, 135, 180, -135, 135],
#      [0, 360, 90, 0, 90, 135, 180, -90],
#      [0, 0, 360, -90, 0, -135, 90, 180]])

# REduced - taking arcs 1, 2 and 5 of full eg
x = 0.12
Distances = np.array(
    [[x, x/2+2],
     [x/2+2, 2],
     ])




# TODO note angles need to be mixed with incidence to determine 0 angle from missing arc
# or we could encode 0 as 360 - > probably better
Angles = np.array(
    [[180, 360,],
     [360, 180,]])

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
network_struct = RecursiveLogitDataStruct(data_list, incidence_mat,
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
print("m_orig = ", exp_utility_matrix.toarray())
obs_per_pair = 1
O = range(distances.shape[0])
D = range(distances.shape[1])
# start at an arc, end at a node, fix ending at a node by appending an arc
# assume for simplicity that the nodes are the nodes at the centre of my arcs

rng = np.random.default_rng()
from scipy import linalg

m, n = exp_utility_matrix.shape
assert m == n
for dest in [1]:  # D: # (0,2) specifies a node via an adjacency list
    print("dest = ", dest, "augmented dest col =", m)
    # for each dest need to compute value function
    m_tilde = exp_utility_matrix.copy()

    # any arcs which connect to desk are zero
    # incidence must locally change too. Need to do a small pen and paper example
    # augmenting an arc means that stuff doesn't need to change as much
    len_with_aug_dest = m + 1  # index is this -1
    m_tilde.resize(len_with_aug_dest, len_with_aug_dest)
    # m_tilde is always going to have same dims, this additional 1 is non destructive,
    # we would just need to reset this column back to zeros and we could reuse matrix
    m_tilde[dest, :] = 0.0 # TODO do we do this? # enforces going to dest
    m_tilde[dest, -1] = 1 # exp(v(a|k)) = 1 when v(a|k) = 0 # try 0.2
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
    print("raw val funcs\n", value_funcs)
    # print(" valf", value_funcs)
    # short_term_utility
    # print("stu")
    # print(short_term_utility)
    v_op=  short_term_utility[0,1]
    v_pp = short_term_utility[0,0]
    print("theory V(p) with dest correction", np.log(np.exp(v_op)/ (1- np.exp(v_pp))))
    print("cond", np.linalg.cond(a_mat.toarray()))
    # continue
#     for orig in [1]:  # O:
#         for i in range(obs_per_pair):
#             # while we haven't reached the dest
#             current_arc = orig
#             path_string = f"Start: {orig} -> "
#             count = 0
#             while current_arc != n + 1:  # index of augmented dest arc
#                 count += 1
#                 if count > 10:
#                     zzz
#
#                 current_incidence_col = incidence_mat[current_arc, :]
#                 # all arcs current arc connects to
#                 neighbour_arcs = current_incidence_col.nonzero()[0]
#
#                 eps = rng.gumbel(loc=-np.euler_gamma, scale=1, size=len(neighbour_arcs))
#
#                 value_functions_observed = (short_term_utility[current_arc, neighbour_arcs]
#                                             + value_funcs[neighbour_arcs].T
#                                             + eps)
#                 if np.any(np.isnan(value_functions_observed)):
#                     raise ValueError("beta vec is invalid, gives non real solution")
#                 # value_functions_observed = np.nan_to_num(value_functions_observed, nan=-np.Inf)
#                 print(f"value functions observed at {current_arc}:\n arcs:\n{neighbour_arcs.T}\n "
#                       f"vals:\n\n{value_funcs.T}")
#
#                 next_arc = np.argmax(value_functions_observed)
#                 path_string += f"{next_arc} -> "
#                 current_arc = next_arc
#
#
#                 # print(current_incidence_col, neighbour_arcs)
#                 # print("dims")
#                 # print(short_term_utility[current_arc, neighbour_arcs].shape,
#                 #       value_funcs[neighbour_arcs].shape, eps.shape)
#                 # print("out")
#                 # print(value_functions_observed)
#                 print(path_string)
#                 # break
#             print(path_string)
#             break
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
