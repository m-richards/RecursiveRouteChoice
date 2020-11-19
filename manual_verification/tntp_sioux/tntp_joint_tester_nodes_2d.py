import numpy as np

from recursiveRouteChoice.data_loading import load_tntp_node_formulation
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation, optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
network_file = "SiouxFalls_net.tntp"
node_max = 24
# network_file = "EMA_net.tntp"
# node_max = 77

data_list, data_list_names = load_tntp_node_formulation(network_file,
                                                        columns_to_extract=["length",
                                                                                      "capacity"
                                                                                     ],
                                                        )
# print(arc_to_index_map)
# print(data_list, data_list_names)
distances = data_list[0].A
data_list = [distances, data_list[1].A]
# data_list = [distances, distances]

incidence_mat = (distances > 0).astype(int)

network_struct = ModelDataStruct(data_list, incidence_mat,
                                 data_array_names_debug=("distances", "u_turn"))

beta_vec = np.array([-0.5, -0.00015])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
print("Linear system size", model.get_exponential_utility_matrix().shape)
orig_indices = np.arange(0, node_max,1)
dest_indices = (orig_indices + 5) % node_max
obs_per_pair = 1


print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
      f"configuration")


def get_data(beta_vec, seed=None):
    # beta_vec_generate = np.array([beta_vec])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec, mu=1)
    obs = model.generate_observations(origin_indices=orig_indices,
                                      dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed,
                                      )
    return obs


# =======================================================
print(120 * "=", 'redo with scipy')
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')  # bfgs, l-bfgs-b


#
# import time
# a = time.time()
# for n, beta_gen in enumerate(np.arange(-0.1, -2, -0.1), start=1):
#     try:
#         obs = get_data(beta_gen, seed=2)
#     except ValueError as e:
#         print(f"beta = {beta_gen} failed, {e}")
#         continue
#     # print(obs)
#     beta_init = -5
#     model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
#                                           initial_beta=beta_init, mu=1,
#                                           optimiser=optimiser)
#     beta = model.solve_for_optimal_beta(verbose=False)
#     print("beta_expected", beta_gen, "beta actual", beta)
#
# b = time.time()
# print("elapsed =", b-a, "s")


import time
import warnings
warnings.simplefilter("ignore")
import logging as logger
log = logger.getLogger()
log.setLevel(logger.CRITICAL+1)
logger.basicConfig(level=logger.CRITICAL)
a = time.time()
n = 0
for b1 in np.arange(-0.1, -2, -0.2):
    b2arr = [-0.0005, -0.001, -0.003]
    # b2arr = np.arange(-0.001, -0.009, -0.002)
    for b2 in b2arr:
        n += 1

        beta_gen = np.array([b1, b2])
        try:
            obs = get_data(beta_gen, seed=None)
        except ValueError as e:
            print(f"beta expected: {beta_gen}, beta_actual: [-  ,-  ]")
            # print(f"beta = {beta_gen} failed, {e}")
            continue
        # print(obs)
        beta_init = [-0.5, -0.00001]
        model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_init, mu=1,
                                              optimiser=optimiser)
        try:
            beta = model.solve_for_optimal_beta(verbose=False)
        except:
            print(f"beta expected: [{beta_gen[0]:6.4f}, {beta_gen[1]:6.4f}],"
                  f" beta_actual: [{'-':6}, {'-':6}]")
            continue
        # print("beta_expected", beta_gen, "beta actual", beta)

        print(f"beta expected: [{beta_gen[0]:6.4f}, {beta_gen[1]:6.4f}],"
              f" beta_actual: [{beta[0]:6.4f}, {beta[1]:6.4f}]")
        rel_errors = [np.abs((beta_gen[i] - beta[i])/beta_gen[i]) for i in range(2)]
        norm = np.linalg.norm(beta_gen - beta)
        print(f"\t {norm:6.4}, ({rel_errors[0]:6.3f}, {rel_errors[1]:6.3f})")

b = time.time()
print("elapsed =", b-a, "s")
