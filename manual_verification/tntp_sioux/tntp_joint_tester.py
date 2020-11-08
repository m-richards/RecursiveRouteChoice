import numpy as np
from scipy.sparse import dok_matrix
import awkward1 as ak

from data_loading import write_obs_to_json, load_obs_from_json, load_tntp_to_sparse_arc_formulation

from recursive_route_choice import ModelDataStruct



import optimisers as op

np.set_printoptions(edgeitems=10, linewidth=300)
from recursive_route_choice import (RecursiveLogitModelPrediction,
    RecursiveLogitModelEstimation)
# from recursive_logit_efficient_update import (RecursiveLogitModelEstimationSM as
#                                               RecursiveLogitModelEstimation)
# from recursive_logit_efficient_update import (RecursiveLogitModelPredictionSM as
#                                               RecursiveLogitModelPrediction)


# np.core.arrayprint._line_width = 500

# DATA
# network_file = "SiouxFalls_net.tntp"
# arcmaxp1 = 76
network_file = "EMA_net.tntp"
arcmaxp1 = 256
# network_file = "Anaheim_net.tntp"
# network_file = "berlin-mitte-center_net.tntp"
standardise='minmax'
# standardise=None
arc_to_index_map, distances = load_tntp_to_sparse_arc_formulation(network_file,
                                                                  columns_to_extract=["length"],
                                                                  standardise=standardise)
# print(arc_to_index_map)
index_node_pair_map = {v: k for (k, v) in arc_to_index_map.items()}

# distances = np.array(
#     [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
#      [3.5, 3, 4, 0, 2.5, 3, 3, 0],
#      [4.5, 4, 5, 0, 0, 0, 4, 3.5],
#      [3, 0, 0, 2, 2, 2.5, 0, 2],
#      [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
#      [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
#      [0, 3, 4, 0, 2.5, 3, 3, 2.5],
#      [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])
#

incidence_mat = (distances > 0).astype(int).A
distances = distances #/ np.mean(distances.A)
distances = distances.A
# distances += np.abs(np.min(distances))
# distances +=1
data_list = [distances]
nz_dist = distances.reshape(distances.shape[0] * distances.shape[1], 1)
nz_dist = nz_dist[nz_dist>0]
print("(max dist, min dist, mean dist) = ", (np.max(nz_dist), np.min(nz_dist),
                                             np.mean(nz_dist), np.std(nz_dist)))
network_struct = ModelDataStruct(data_list, incidence_mat,
                                 data_array_names_debug=("distances", "u_turn"))

beta_vec = np.array([-1])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
print("Linear system size", model.get_exponential_utility_matrix().shape)
orig_indices = np.arange(0, arcmaxp1, 30)
dest_indices = (orig_indices + 5) % arcmaxp1
# orig_indices = np.arange(0, 7, 1)
# dest_indices = np.arange(0, 7, 1)
obs_per_pair = 1


print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
      f"configuration")


def get_data(beta, seed=None):
    beta_vec_generate = np.array([beta])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec_generate, mu=1)
    obs = model.generate_observations(origin_indices=orig_indices,
                                      dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed,
                                      )
    return obs


# =======================================================
print(120 * "=", 'redo with scipy')
optimiser = op.ScipyOptimiser(method='l-bfgs-b')  # bfgs, l-bfgs-b


import time
a = time.time()
# test_range = np.arange(-0.1, -5, -0.5)
# test_range = np.arange(-3, -20, -1)
# test_range = np.arange(-0.000, -0.1, -0.03)
# test_range = np.array([-0.3, -0.4, -2, -3, -10, -15])
# test_range = np.array([ -20, -26, -26.5, -27, -28])
# test_range = np.array([-0.01, -0.1, -0.3, -0.4, -0.5, -1, -3, -4, -5, -10, -15, -20, -50, -100,
#                       -200, -250,
#                       -300, -2000, -100000])
# Checking Berlin mitte beta range based upon ||M||^infty
# Trying to normalising **10
# test_range = np.array([-0.01, -0.02, -0.03, -0.038, -0.0385, -0.039, -0.04, -0.05,  -0.1, ])
# try to nromalise **1000
# test_range = np.array([-0.01, -0.015, -0.016, -0.01625, -0.02])
# PARAMETER RANGES TEST
# Sioux Falls unstandardised
test_range = np.array([-0.35, -25.7,  -26.5, -26.6, -26.7, -26.8, -26.9])
# EMA unstandardised
# test_range = np.array([-0.5, -0.6, -2, -3, -6.1, -6.2, -6.3 ])
#Sioux standardised
# test_range = np.array([-5.8,-5.9,  -395, -400])

# test_range = np.array([-20,-21, -22, -23, -24, -25, -26, -27, -28, -29, -30])

# def test_obs(beta):
#     try:
#
#         obs =  get_data(beta, seed=2)
#         print("Legal obs with beta = ",beta)
#         return obs
#
#     except ValueError as e:
#         if "small" in str(e):
#             return "small"
#         else:
#             return "large"
# lower = -1
# upper = -500
# while True:
#     mid = lower + (lower - upper) / 2
#     print("L,M,U", (lower, mid, upper))
#     obs = test_obs(mid)
#
#     if obs =="small":
#         lower = mid
#     elif obs=="large":
#         upper=mid
#     else:
#         break
#
#     if lower- upper>0:
#         print("L, U", lower, upper)
#         break


#
#
# test_range = np.array([-26,-26.1, -26.2, -26.3, -26.4, -26.45, -26.5, -27,])
for n, beta_gen in enumerate(test_range, start=1):
    print("BETA GEN = ", beta_gen)
    try:
        obs = get_data(beta_gen, seed=None)
    except ValueError as e:
        print(f"beta = {beta_gen} failed in prediction, {e}")
        continue
    # print(obs)
    beta_init = -5

    model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=beta_init, mu=1,
                                          optimiser=optimiser)
    # try:
    beta = model.solve_for_optimal_beta(verbose=False)
    # except ValueError:
    #     print("beta_expected", beta_gen,"beta actual failed linesearch")
    #     continue
    print("beta_expected", beta_gen, "beta actual", beta)

b = time.time()
print("elapsed =", b-a, "s")
