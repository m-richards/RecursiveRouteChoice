import numpy as np
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct

# DATA
# A trivial network
distances = np.array(
    [[0, 5, 0, 4],
     [0, 0, 6, 0],
     [0, 6, 0, 5],
     [4, 0, 0, 0]])

incidence_mat = (distances > 0).astype(int)

network_attribute_list = [distances]
network_struct = ModelDataStruct(network_attribute_list, incidence_mat,
                                 data_array_names_debug=("distances",))
model = RecursiveLogitModelPrediction(network_struct, initial_beta=[-0.4], mu=1)

obs_indices = [0, 3]
dest_indices = [1, 2]
obs_per_pair = 15
print(f"Generating {obs_per_pair * len(obs_indices) * len(dest_indices)} obs total")

obs = model.generate_observations(origin_indices=obs_indices, dest_indices=dest_indices,
                                  num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=1)
print(obs)
