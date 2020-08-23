# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)
import os

import numpy as np
import time
import scipy
from scipy.sparse import linalg as splinalg
from data_loading import load_standard_path_format_csv
from data_processing import AngleProcessor
from main import RecursiveLogitModelEstimation, RecursiveLogitDataStruct, RecursiveLogitModel, \
    RecursiveLogitModelPrediction
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

# # TODO note angles need to be mixed with incidence to determine 0 angle from missing arc
# # or we could encode 0 as 360 - > probably better
# Angles = np.array(
#     [[180, 360,],
#      [360, 180,]])

# TODO ban u-turns:
# Distances = Distances - np.diag(Distances)
# Angles = Angles - np.diag(Angles)
# print(Distances)


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
model = RecursiveLogitModelPrediction(network_struct,  user_obs_mat=None,
                                      initial_beta=beta_vec, mu=1)

obs = model.generate_observations(origin_indices=[0, 1, 2, 7], dest_indices=[1, 6, 3],
                                  num_obs_per_pair=4, iter_cap=15, rng_seed=1)

from pprint import pprint
pprint(obs)
print(obs)