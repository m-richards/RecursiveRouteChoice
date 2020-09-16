
import numpy as np

from data_loading import write_obs_to_json, load_tnpm_to_sparse_node_formulation
from data_loading import load_tnpm_to_sparse_arc_formulation

from recursive_route_choice import ModelDataStruct, RecursiveLogitModelPrediction

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA


network_file = "SiouxFalls_net.tntp"
data_list, data_list_names = load_tnpm_to_sparse_node_formulation(network_file,
                                                                 columns_to_extract=["length",
                                                                                     'capacity'],
                                                                 )
distances, capacity = data_list
for i, j in zip(data_list, data_list_names):
    print(f"{j}:")
    # print(i.A)
# print(arc_to_index_map)
# index_node_pair_map = {v: k for (k, v) in arc_to_index_map.items()}
#
incidence_mat = (distances > 0).astype(int)

network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances", "capacity"))

beta_vec = np.array([-0.7, -0.0001])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
# orig_indices = np.arange(1, 70, 50)
orig_indices = [22, 18]
# dest_indices = np.arange(1, 8, 2)
dest_indices = [7, 13, 9]
obs_per_pair = 30
obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                  num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=1,
                                  )
# print(obs)

# Print paths in terms of nodes
print("\nNode Paths")
for path in obs:
    print([i + 1 for i in path[1:]])

#
# print(obs)
# write_obs_to_json("sioux_out_obs.json", obs, allow_rewrite=True)
