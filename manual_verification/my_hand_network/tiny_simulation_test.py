import numpy as np
from data_loading import write_obs_to_json
from recursive_route_choice import ModelDataStruct, RecursiveLogitModelPrediction

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
# distances = np.array(
#     [[4, 3.5, 3],
#      [3.5, 3, 2.5],
#      [3, 2.5, 2],
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

beta_vec = np.array([-1])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
# obs_indices = [i for i in range(8)]
# obs = model.generate_observations(origin_indices=obs_indices,
#                                   dest_indices=obs_indices,
#                                   num_obs_per_pair=1, iter_cap=2000, rng_seed=1,
#                                   )
obs_indices = [1, 2, 3, 4, 5, 6, 7, 8]
obs = model.generate_observations(origin_indices=obs_indices,
                                  dest_indices=[7],
                                  num_obs_per_pair=1, iter_cap=2000, rng_seed=1,
                                  )

print(obs)

print("\nPath in terms of arcs:")
for path in obs:
    string = "Orig: "
    f = "Empty Path, should not happen"
    for arc_index in path:
        string += f"-{arc_index + 1}- => "
    string += ": Dest"

    print(string)

print(obs)
write_obs_to_json("my_networks_obs.json", obs, allow_rewrite=True)
