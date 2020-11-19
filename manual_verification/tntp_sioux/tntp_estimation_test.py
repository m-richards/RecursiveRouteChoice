
import numpy as np
import awkward1 as ak
from recursiveRouteChoice.data_loading import load_tntp_to_sparse_arc_formulation, \
    load_obs_from_json
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelEstimation, optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500
obs_fname = 'sioux_out_obs.json'
obs_lil = load_obs_from_json(obs_fname)
obs_ak = ak.from_json(obs_fname)
print("len ", len(obs_ak))


# DATA
network_file = "SiouxFalls_net.tntp"
arc_to_index_map, distances = load_tntp_to_sparse_arc_formulation(network_file,
                                                                  columns_to_extract=["length"])

index_node_pair_map = {v: k for (k, v) in arc_to_index_map.items()}

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances",))

beta_vec = np.array([-1])
optimiser = optimisers.LineSearchOptimiser(optimisers.OptimHessianType.BFGS, max_iter=4)

model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
                                      initial_beta=beta_vec, mu=1,
                                      optimiser=optimiser)
# log_like_out, grad_out = model.get_log_likelihood()

model.solve_for_optimal_beta()
