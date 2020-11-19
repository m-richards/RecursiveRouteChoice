import numpy as np
from scipy.sparse import dok_matrix

from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation, optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
# silly deterministic network
# distances = np.array(
#     [[0, 5, 0, 4],
#      [0, 0, 6, 0],
#      [0, 6, 0, 5],
#      [4, 0, 0, 0]])
# bigger silly - see phone photo
# distances = np.array(
#     [[0, 5, 0, 4, 0, 0, 0, 0, 0, 0],
#      [0, 0, 6, 0, 0, 0, 0, 0, 0, 6],
#      [0, 6, 0, 5, 0, 0, 0, 0, 0, 0],
#      [4, 0, 0, 0, 5, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
#      [5, 0, 0, 0, 6, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 6, 6, 0],
#      [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
#      [0, 0, 6, 0, 0, 0, 0, 0, 0, 6],
#      [0, 0, 0, 0, 0, 0, 0, 6, 6, 0]
#      ])



distances = np.array(
    [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
     [3.5, 3, 4, 0, 2.5, 3, 3, 0],
     [4.5, 4, 5, 0, 0, 0, 4, 3.5],
     [3, 0, 0, 2, 2, 2.5, 0, 2],
     [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
     [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
     [0, 3, 4, 0, 2.5, 3, 3, 2.5],
     [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances"))
beta_known = -1
beta_vec_generate = np.array([beta_known])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec_generate, mu=1)
#
# obs_indices = [0]
# obs = model.generate_observations(origin_indices=obs_indices,
#                                   dest_indices=[1],
#                                   num_obs_per_pair=40, iter_cap=2000, rng_seed=1,
#                                   )

obs_indices = [0, 3, 5]
dest_indices = [1, 2, 7]
obs_per_pair = 60
print(f"Generating {obs_per_pair * len(obs_indices) * len(dest_indices)} obs total")


def get_data(beta, seed=None):
    beta_vec_generate = np.array([beta])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec_generate, mu=1)
    obs = model.generate_observations(origin_indices=obs_indices,
                                      dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed,
                                      )
    return obs


#
# print(obs)
#
# print("\nPath in terms of arcs:")
# for path in obs:
#     string = "Orig: "
#     f = "Empty Path, should not happen"
#     for arc_index in path[1:]:
#         string += f"-{arc_index + 1}- => "
#     string += ": Dest"
#
#     print(string)
#
# obs_fname = "my_networks_obs2.json"
# write_obs_to_json(obs_fname, obs, allow_rewrite=True)
#
# np.set_printoptions(edgeitems=10, linewidth=300)
# # np.core.arrayprint._line_width = 500
# # obs_fname = 'my_networks_obs.json'
# obs_lil = load_obs_from_json(obs_fname)
# obs_ak = ak.from_json(obs_fname)
# print("len ", len(obs_ak))
#
#
# # silly levels of inefficiency but will fix later
#
# # obs = np.array(obs_lil)
# # obs = scipy.sparse.dok_matrix(obs_lil)
#
# #
# # DATA
# # distances = np.array(
# #     [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
# #      [3.5, 3, 4, 0, 2.5, 3, 3, 0],
# #      [4.5, 4, 5, 0, 0, 0, 4, 3.5],
# #      [3, 0, 0, 2, 2, 2.5, 0, 2],
# #      [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
# #      [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
# #      [0, 3, 4, 0, 2.5, 3, 3, 2.5],
# #      [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

distances = dok_matrix(distances)

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances",))

# =======================================================
print(120 * "=", 'redo with scipy')
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b') # bfgs, l-bfgs-b
beta = -0.4
beta_vec = np.array([beta])  # 4.96 diverges


import time
a = time.time()
for n, beta_gen in enumerate(np.arange(-0.1, -3, -0.1), start=1):
    try:
        obs = get_data(beta_gen, seed=1)
    except ValueError as e:
        print(f"beta = {beta_gen} failed, {e}")
        continue
    # print(obs)
    beta_init = -1.0
    model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=beta_init, mu=1,
                                          optimiser=optimiser)
    beta = model.solve_for_optimal_beta(verbose=False)
    print("beta_expected", beta_gen, "beta actual", beta)

b = time.time()
print("elapsed =", b-a, "s")

# 60 obs above test
