
import numpy as np

from data_loading import write_obs_to_json
from data_loading import load_tntp_to_sparse_arc_formulation

from recursive_route_choice import ModelDataStruct, RecursiveLogitModelPrediction

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


network_file = "SiouxFalls_net.tntp"
arc_to_index_map, distances = load_tntp_to_sparse_arc_formulation(network_file, columns_to_extract=["length"],
                                                                  )
print(arc_to_index_map)
index_node_pair_map = {v: k for (k, v) in arc_to_index_map.items()}

incidence_mat = (distances > 0).astype(int)

data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances", "u_turn"))

beta_vec = np.array([-1])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
# orig_indices = np.arange(1, 70, 50)
orig_indices = [22.0]
dest_indices = np.arange(2, 70, 50)
obs_per_pair = 2
obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                  num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=1,
                                  )
# print(obs)

# Print paths in terms of nodes
print("\nNode Paths")
for path in obs:
    string = "Orig Node: "
    f = "Empty Path, should not happen"
    for arc_index in path:
        print(arc_index)
        if arc_index == len(index_node_pair_map):
            # pass
            continue  # we are at the dummy dest arc
        s, f = index_node_pair_map[arc_index]
        string += f"({s}) -> "
    string += ": Dest"

    print(string)
print("\nreport arcs and nodes")
for path in obs:
    string = "Orig: "
    f = "Empty Path, should not happen"
    for arc_index in path:
        if arc_index == len(index_node_pair_map):
            # pass
            continue  # we are at the dummy dest arc
        s, f = index_node_pair_map[arc_index]
        string += f"({s}) -{arc_index + 1}- => "
    string += ": Dest"

    print(string)

print("\nPath in terms of arcs:")
for path in obs:
    string = "Orig: "
    f = "Empty Path, should not happen"
    for arc_index in path:
        string += f"-{arc_index + 1}- => "
    string += ": Dest"

    print(string)

print(obs)
write_obs_to_json("sioux_out_obs.json", obs, allow_rewrite=True)
