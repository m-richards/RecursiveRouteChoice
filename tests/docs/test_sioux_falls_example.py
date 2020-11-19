"""Script which is presented in the Sphinx documentation, reading appropriate sections of this
file. If this file is updated, the line numbers in Sphinx (docs/source/involved_example.rst)
need to be updated too.


"""

import numpy as np

from recursiveRouteChoice.data_loading import load_tntp_node_formulation
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers

# DATA
import os
print("sys path is", os.getcwd(), os.listdir(os.getcwd()))
network_file = os.path.join("tests", "docs", "SiouxFalls_net.tntp")
node_max = 24  # from network file

data_list, data_list_names = load_tntp_node_formulation(
    network_file, columns_to_extract=["length", "capacity"], sparse_format=False)

# Convert entries to np.arrays since network is small so dense format is more efficient
distances = data_list[0]

incidence_mat = (distances > 0).astype(int)
network_struct = ModelDataStruct(data_list, incidence_mat)

beta_sim = np.array([-0.8, -0.00015])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_sim, mu=1)
print("Linear system size", model.get_exponential_utility_matrix().shape)

# sparse sample for quick running example
orig_indices = np.arange(0, node_max, 2)
dest_indices = (orig_indices + 5) % node_max
# sample every OD pair once
# orig_indices = np.arange(0, node_max, 1)
# dest_indices = np.arange(0, node_max, 1)
obs_per_pair = 1
print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
      f"configuration")
seed = 42
obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                  num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed)

optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
beta_est_init = [-5, -0.00001]
model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=beta_est_init, mu=1,
                                          optimiser=optimiser)

beta_est = model_est.solve_for_optimal_beta(verbose=False)

print(f"beta expected: [{beta_sim[0]:6.4f}, {beta_sim[1]:6.4f}],"
      f" beta_actual: [{beta_est[0]:6.4f}, {beta_est[1]:6.4f}]")


def test_est():
    assert np.allclose([-1.02129578, -0.000175462809], beta_est)


def test_sim():
    assert obs == [[5, 0, 1, 5],
                   [5, 2, 3, 4, 5],
                   [5, 4, 5],
                   [5, 6, 7, 5],
                   [5, 8, 4, 5],
                   [5, 10, 3, 4, 5],
                   [5, 12, 11, 10, 3, 4, 5],
                   [5, 14, 18, 16, 15, 7, 5],
                   [5, 16, 15, 7, 5],
                   [5, 18, 16, 15, 7, 5],
                   [5, 20, 21, 14, 18, 16, 15, 7, 5],
                   [5, 22, 13, 10, 3, 4, 5],
                   [7, 0, 1, 5, 7],
                   [7, 2, 3, 4, 5, 7],
                   [7, 4, 5, 7],
                   [7, 6, 7],
                   [7, 8, 7],
                   [7, 10, 9, 15, 7],
                   [7, 12, 11, 10, 3, 4, 5, 7],
                   [7, 14, 18, 16, 15, 7],
                   [7, 16, 15, 7],
                   [7, 18, 16, 15, 7],
                   [7, 20, 19, 18, 16, 15, 7],
                   [7, 22, 13, 10, 9, 15, 7],
                   [9, 0, 2, 11, 10, 9],
                   [9, 2, 3, 10, 9],
                   [9, 4, 8, 9],
                   [9, 6, 7, 15, 9],
                   [9, 8, 9],
                   [9, 10, 9],
                   [9, 12, 11, 10, 9],
                   [9, 14, 9],
                   [9, 16, 15, 9],
                   [9, 18, 16, 15, 9],
                   [9, 20, 21, 14, 9],
                   [9, 22, 13, 10, 9],
                   [11, 0, 2, 11],
                   [11, 2, 11],
                   [11, 4, 3, 2, 11],
                   [11, 6, 7, 15, 9, 10, 11],
                   [11, 8, 9, 10, 11],
                   [11, 10, 11],
                   [11, 12, 11],
                   [11, 14, 13, 10, 11],
                   [11, 16, 15, 9, 10, 11],
                   [11, 18, 16, 15, 9, 10, 11],
                   [11, 20, 23, 12, 11],
                   [11, 22, 23, 12, 11],
                   [13, 0, 2, 3, 10, 13],
                   [13, 2, 3, 10, 13],
                   [13, 4, 3, 10, 13],
                   [13, 6, 17, 15, 9, 10, 13],
                   [13, 8, 9, 10, 13],
                   [13, 10, 13],
                   [13, 12, 23, 22, 13],
                   [13, 14, 13],
                   [13, 16, 18, 14, 13],
                   [13, 18, 14, 13],
                   [13, 20, 23, 22, 13],
                   [13, 22, 13],
                   [15, 0, 1, 5, 7, 15],
                   [15, 2, 3, 4, 5, 7, 15],
                   [15, 4, 5, 7, 15],
                   [15, 6, 7, 15],
                   [15, 8, 9, 15],
                   [15, 10, 9, 15],
                   [15, 12, 23, 20, 19, 18, 16, 15],
                   [15, 14, 18, 16, 15],
                   [15, 16, 15],
                   [15, 18, 16, 15],
                   [15, 20, 19, 18, 16, 15],
                   [15, 22, 13, 14, 18, 16, 15],
                   [17, 0, 1, 5, 7, 6, 17],
                   [17, 2, 3, 10, 9, 15, 17],
                   [17, 4, 5, 7, 6, 17],
                   [17, 6, 17],
                   [17, 8, 9, 15, 17],
                   [17, 10, 9, 15, 17],
                   [17, 12, 23, 20, 19, 17],
                   [17, 14, 9, 15, 17],
                   [17, 16, 15, 17],
                   [17, 18, 16, 15, 17],
                   [17, 20, 19, 17],
                   [17, 22, 21, 19, 17],
                   [19, 0, 1, 5, 7, 15, 16, 18, 19],
                   [19, 2, 3, 10, 13, 14, 18, 19],
                   [19, 4, 5, 7, 15, 16, 18, 19],
                   [19, 6, 17, 19],
                   [19, 8, 9, 15, 17, 19],
                   [19, 10, 13, 14, 18, 19],
                   [19, 12, 23, 20, 19],
                   [19, 14, 21, 19],
                   [19, 16, 18, 19],
                   [19, 18, 19],
                   [19, 20, 19],
                   [19, 22, 21, 19],
                   [21, 0, 2, 11, 12, 23, 20, 21],
                   [21, 2, 3, 10, 13, 22, 21],
                   [21, 4, 8, 9, 14, 21],
                   [21, 6, 17, 19, 21],
                   [21, 8, 9, 14, 21],
                   [21, 10, 13, 22, 21],
                   [21, 12, 23, 20, 21],
                   [21, 14, 21],
                   [21, 16, 18, 19, 21],
                   [21, 18, 19, 21],
                   [21, 20, 21],
                   [21, 22, 21],
                   [23, 0, 2, 11, 12, 23],
                   [23, 2, 11, 12, 23],
                   [23, 4, 8, 9, 10, 13, 22, 23],
                   [23, 6, 17, 19, 20, 23],
                   [23, 8, 9, 10, 13, 22, 23],
                   [23, 10, 13, 22, 23],
                   [23, 12, 23],
                   [23, 14, 21, 20, 23],
                   [23, 16, 18, 19, 20, 23],
                   [23, 18, 19, 20, 23],
                   [23, 20, 23],
                   [23, 22, 23],
                   [1, 0, 1],
                   [1, 2, 0, 1],
                   [1, 4, 5, 1],
                   [1, 6, 7, 5, 1],
                   [1, 8, 4, 5, 1],
                   [1, 10, 3, 4, 5, 1],
                   [1, 12, 11, 2, 0, 1],
                   [1, 14, 18, 16, 15, 7, 5, 1],
                   [1, 16, 15, 7, 5, 1],
                   [1, 18, 16, 15, 7, 5, 1],
                   [1, 20, 19, 18, 16, 15, 7, 5, 1],
                   [1, 22, 13, 10, 3, 4, 5, 1],
                   [3, 0, 2, 3],
                   [3, 2, 3],
                   [3, 4, 3],
                   [3, 6, 7, 5, 4, 3],
                   [3, 8, 4, 3],
                   [3, 10, 3],
                   [3, 12, 11, 2, 3],
                   [3, 14, 13, 10, 3],
                   [3, 16, 15, 9, 10, 3],
                   [3, 18, 14, 13, 10, 3],
                   [3, 20, 23, 22, 13, 10, 3],
                   [3, 22, 13, 10, 3]] != [[1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 3, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 3, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [1, 3, 0, 1],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2],
                                           [2, 3, 0, 1, 2]]
